import torch
from torch import nn
import torch.distributions as dist
from torch.utils.data import Dataset
from typing import Callable, Iterable, List
from torchvision.datasets import MNIST
from pathlib import Path


def get_iters(data_path: Path, fetcher: Callable, **kwargs) -> Dataset:
    """Get a tuple of torch.utils.data.Dataset objects."""
    return fetcher(root=data_path, **kwargs)


def conv_red(
    in_dim: int, ker: int, pad: int = 0, dilation: int = 1, stride: int = 1
) -> int:
    """Calculate the dimension after a single convolution operation."""
    return (in_dim + 2 * pad - dilation * (ker - 1) - 1) // stride + 1


def get_dim_at_end(
    in_dim: int,
    kernels: Iterable[int],
    strides: Iterable[int],
    pools: Iterable[int],
    strides_pool: Iterable[int],
) -> int:
    """A function to calculate a dimension after multiple conv layers."""
    for i in range(len(kernels)):
        in_dim = conv_red(in_dim, kernels[i], stride=strides[i])
        # only pool if kernel size less than or equal to input size;
        if in_dim >= pools[i]:
            in_dim = conv_red(in_dim, pools[i], stride=strides_pool[i])
        # print(in_dim)
    return in_dim


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        ker_size,
        stride=1,
        batch_norm_dim=1,
        padding=0,
    ):
        """Conv block with BatchNorm2d -> Conv2d -> ReLU."""
        super(ConvBlock, self).__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(batch_norm_dim),
            nn.Conv2d(
                in_channels,
                out_channels,
                padding=padding,
                kernel_size=ker_size,
                stride=stride,
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):
    def __init__(
        self,
        D_latent: int,
        channels: List[int] = [3, 5],
        kers: List[int] = [5, 3],
        strides: List[int] = [1, 1],
        padding: List[int] = [0, 0],
    ):
        """
        Construct Encoder for inputs with dims: (batch_size, 1, 28, 28)
        to be encoded in dim: (batch_size, D_latent).
        """
        super(Encoder, self).__init__()
        self.D_latent = D_latent
        self.conv_blocks = nn.Sequential()
        assert len(kers) == len(strides) == len(channels)

        # add conv blocks;
        in_dim = 28
        for i in range(len(kers)):
            if in_dim < kers[i]:
                break
            # MNIST is grayscale so has single-channel images;
            in_channel = 1 if i == 0 else channels[i - 1]
            # add conv blocks of the type: BatchNorm -> Conv -> Relu;
            self.conv_blocks.append(
                ConvBlock(
                    in_channel,
                    channels[i],
                    padding=padding[i],
                    ker_size=kers[i],
                    stride=strides[i],
                    batch_norm_dim=in_channel,
                )
            )
            # calculate the remaining dim;
            in_dim = conv_red(
                in_dim, kers[i], stride=strides[i], pad=padding[i]
            )
            # add a maxpool with ker=2; stride=2;
            if in_dim >= 3:
                self.conv_blocks.append(nn.MaxPool2d(2, stride=2))
                # update remaining dim;
                in_dim = conv_red(in_dim, 2, stride=2)
        # print the input dimension for the linear layer;
        # print(f"dim * dim * channel: {in_dim * in_dim * channels[-1]}")
        # output D_latent * 2 things;
        # the first D_latent outputs are means;
        # the second D_latent outputs are log(std);
        self.linear = nn.Sequential(
            nn.BatchNorm2d(channels[-1]),
            nn.Flatten(1),
            nn.Linear(in_dim * in_dim * channels[-1], D_latent * 2),
        )

    def forward(self, X):
        """Return a distribution q(z | x).
        X contains zeros and ones; shape = (batch_size, 1, 28, 28)

        Return: a torch distribution instance,
                defined on values of shape = (batch_size, D_latent).
        """
        # take mus and sigmas;
        params = self.linear(self.conv_blocks(X))
        # first D_latent are mus;
        mus = params[:, : self.D_latent]
        # second D_latent are log sigmas;
        sigmas = torch.exp(params[:, self.D_latent :])
        assert mus.shape == (X.shape[0], self.D_latent)
        assert mus.shape == sigmas.shape
        # since the cov is diagonal for Gaussian dist,
        # each element of zi is indep of the other;
        # due to Gaussian assumption; so each zik~N(mus[i, k], sigma[i, k])
        # so I can use broadcast and just one dist.Normal;
        return dist.Normal(mus, sigmas)


class Decoder(nn.Module):
    def __init__(self, D_latent, final_latent=11):
        """
        Construct Decoder.

        This will operate on inputs of shape (batch_size, D_latent).
        """
        super().__init__()
        self.D_latent = D_latent
        # this is the reconstruction latent at
        # the final linear layer;
        self.final_latent = final_latent
        self.net = nn.Sequential(
            nn.BatchNorm1d(1),
            nn.Linear(self.D_latent, 53),
            nn.ReLU(),
            nn.BatchNorm1d(1),
            nn.Linear(53, 23),
            nn.ReLU(),
            nn.BatchNorm1d(1),
            nn.Linear(23, 28 * final_latent),
        )
        # to save some parameters, define a
        # matrix to map (28, final_layer) to (28, 28);
        # also use some good inits (Glorot and Bengio);
        # https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        self.projection = nn.init.xavier_normal_(
            nn.Parameter(torch.ones((self.final_latent, 28)))
        )
        self.diag_offset = nn.Parameter(torch.normal(0, std=0.1, size=(28,)))

    def forward(self, Z):
        """Return a distribution p(x | z)

        Z is real-valued, shape = (batch_size, D_latent).

        Return: a torch Bernoulli distribution instance,
                defined on values of shape = (batch_size, 1, 28, 28).
        """
        # get things in shape (batch_size, 28, final_latent)
        fz = self.net(Z.unsqueeze(1)).view((-1, 28, self.final_latent))
        # use the projection mat to map to
        # (batch_size, 28, 28)
        fz = fz @ self.projection + torch.diag(self.diag_offset)
        # unsqueeze to get (batch_size, 1, 28, 28) shape
        # and put through sigmoid for the bernoulli;
        return dist.Bernoulli(torch.sigmoid(fz.unsqueeze(1)))


def get_free_energy(
    enc: Encoder, dec: Decoder, X: torch.FloatTensor
) -> torch.Tensor:
    """
    Return the free energy in shape (batch_size, 1),
    based on the encoder and decoder nets.
    enc : Instance of `Encoder` class, which returns a distribution
          over Z when called on a batch of inputs X
    dec : Instance of `Decoder` class, which returns a distribution
          over X when called on a batch of inputs Z
    X   : A batch of datapoints, of shape = (batch_size, 1, 28, 28).
    """
    # get q(z|x) from the encoder;
    qzx = enc(X)
    # sample z values (X.shape[0], D_latent);
    z_samples = qzx.rsample()
    assert z_samples.shape == (X.shape[0], enc.D_latent)
    # get log prior;
    log_prior = dist.Normal(0, 1).log_prob(z_samples).sum(-1)
    # get likelihood;
    pxz = dec(z_samples)
    # return shape (batch_size, 1);
    return (
        log_prior
        + pxz.log_prob(X).sum([1, 2, 3])
        - qzx.log_prob(z_samples).sum(-1)
    )


def run_training(N_epochs, enc, dec, opt_vae, train_loader, data_size):
    """Train the VAE."""
    for epoch in range(N_epochs):
        train_loss = 0.0
        for (X, _) in train_loader:
            opt_vae.zero_grad()
            loss = -get_free_energy(enc, dec, X).mean()
            loss.backward()
            opt_vae.step()
            train_loss += loss.item() * X.shape[0] / data_size
        print(f"Epoch {epoch}, train loss = {train_loss:.4f}")


if __name__ == "__main__":
    # dim after conv with ker 5 and maxpool with ker=2;
    conv_pool1 = conv_red(conv_red(28, 5, pad=0), ker=2, stride=2)
    # dim after additional conv with ker=3 and maxpool with ker=2;
    assert 5 == conv_red(conv_red(conv_pool1, ker=3), ker=2, stride=2)
