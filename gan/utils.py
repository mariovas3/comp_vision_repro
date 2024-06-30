import math

import torch
from torch import nn


def conv_dim_formula(in_dim, kernels, paddings, strides, dilations=None):
    """
    Calculate final shape of in_dim after successive conv1d operations.

    Args:
        in_dim (int): input dimension.
        kernels (Sequence[int]): indexable seq of kernels.
        paddings (Sequence[int]): indexable seq of padding on both sides.
        strides (Sequence[int]): indexable seq of strides.
        dilations (Sequence[int]): indexable seq of dilations.
            If dilations is None, all dilations assumed to be 1.
    """
    assert len(kernels) == len(paddings) == len(strides)
    if dilations is not None:
        assert len(kernels) == len(dilations)

    out = in_dim
    for i in range(len(kernels)):
        offset = 2 * paddings[i] - kernels[i]
        if dilations is not None:
            offset -= (kernels[i] - 1) * (dilations[i] - 1)
        out = (out + offset) // strides[i] + 1
    return out


def get_height_width_conv(
    Hin, Win, kernels, paddings, strides, dilations=None
):
    Hout = conv_dim_formula(
        in_dim=Hin,
        kernels=kernels,
        paddings=paddings,
        strides=strides,
        dilations=dilations,
    )
    Wout = conv_dim_formula(
        in_dim=Win,
        kernels=kernels,
        paddings=paddings,
        strides=strides,
        dilations=dilations,
    )
    return Hout, Wout


def conv_transpose_dim_formula(
    in_dim, kernels, paddings, strides, dilations=None, output_paddings=None
):
    out = in_dim
    for i in range(len(kernels)):
        offset = -2 * paddings[i] + kernels[i]
        if dilations:
            offset += (kernels[i] - 1) * (dilations[i] - 1)
        if output_paddings:
            offset += output_paddings[i]
        out = (out - 1) * strides[i] + offset
    return out


def get_height_width_convtranspose(
    Hin, Win, kernels, paddings, strides, dilations=None, output_paddings=None
):
    Hout = conv_transpose_dim_formula(
        in_dim=Hin,
        kernels=kernels,
        paddings=paddings,
        strides=strides,
        dilations=dilations,
        output_paddings=output_paddings,
    )
    Wout = conv_transpose_dim_formula(
        in_dim=Win,
        kernels=kernels,
        paddings=paddings,
        strides=strides,
        dilations=dilations,
        output_paddings=output_paddings,
    )
    return Hout, Wout


class GeneratorConvTranspose(nn.Module):
    def __init__(
        self,
        Hout,
        Wout,
        latent_dim,
        out_channels,
        kernels,
        paddings,
        strides,
        dilations=None,
        output_paddings=None,
    ):
        """
        Expecting latent input of shape (..., latent_dim, 1, 1),
        and then ConvTranspose2d layers expand it to the needed dim.
        """
        super().__init__()
        self.Cout = out_channels[-1]
        self.Hout, self.Wout = Hout, Wout
        self.latent_dim = latent_dim
        out_channels = [self.latent_dim] + out_channels
        self.net = nn.Sequential()
        for i in range(len(kernels)):
            self.net.add_module(
                f"convt_{i}",
                nn.ConvTranspose2d(
                    in_channels=out_channels[i],
                    out_channels=out_channels[i + 1],
                    kernel_size=kernels[i],
                    stride=strides[i],
                    padding=paddings[i],
                    dilation=1 if dilations is None else dilations[i],
                    output_padding=0
                    if output_paddings is None
                    else output_paddings[i],
                ),
            )
            self.net.add_module(f"bn_{i}", nn.BatchNorm2d(out_channels[i + 1]))
            self.net.add_module(f"relu_{i}", nn.ReLU())
        H, W = get_height_width_convtranspose(
            Hin=1,
            Win=1,
            kernels=kernels,
            paddings=paddings,
            strides=strides,
            dilations=dilations,
            output_paddings=output_paddings,
        )
        # this is so I don't have to do too much calculations
        # for the shapes convtranspose;
        self.out_layer = nn.Sequential(
            nn.Flatten(-2),  # flatten the height and width dim;
            nn.Linear(H * W, Hout * Wout),
            nn.Tanh(),
        )

    def forward(self, z):
        if 4 - z.ndim == 2:
            z = z[:, :, None, None]
        assert z.ndim >= 3 and z.shape[-2:] == (
            1,
            1,
        ), "z should be of dim (..., latent_dim, 1, 1)"
        return self.out_layer(self.net(z)).view(
            -1, self.Cout, self.Hout, self.Wout
        )


class GeneratorMLP(nn.Module):
    def __init__(self, latent_dim, n_hidden, hidden_dim, out_shape):
        super().__init__()
        self.in_dim = latent_dim
        self.out_dim = math.prod(out_shape)
        self.n_channels, self.height, self.width = out_shape
        # add first hidden layer;
        self.net = nn.Sequential()
        self.net.add_module("hidden_0", nn.Linear(self.in_dim, hidden_dim))
        self.net.add_module("gelu_0", nn.GELU(approximate="tanh"))
        # add remaining hidden layers;
        for i in range(n_hidden - 1):
            self.net.add_module(
                f"hidden_{i+1}", nn.Linear(hidden_dim, hidden_dim)
            )
            self.net.add_module(f"gelu_{i+1}", nn.GELU(approximate="tanh"))
        self.net.add_module(f"out_layer", nn.Linear(hidden_dim, self.out_dim))

    def forward(self, x):
        return torch.tanh(self.net(x)).view(
            -1, self.n_channels, self.height, self.width
        )


class DiscriminatorCNN(nn.Module):
    def __init__(
        self,
        Cin,
        Hin,
        Win,
        out_channels,
        kernels,
        paddings,
        strides,
        dilations=None,
    ):
        """
        Expecting latent input of shape (..., latent_dim, 1, 1),
        and then ConvTranspose2d layers expand it to the needed dim.
        """
        super().__init__()
        self.Cin = Cin
        out_channels = [self.Cin] + out_channels
        self.net = nn.Sequential()
        for i in range(len(kernels)):
            self.net.add_module(
                f"conv_{i}",
                nn.Conv2d(
                    in_channels=out_channels[i],
                    out_channels=out_channels[i + 1],
                    kernel_size=kernels[i],
                    stride=strides[i],
                    padding=paddings[i],
                    dilation=1 if dilations is None else dilations[i],
                ),
            )
            self.net.add_module(f"bn_{i}", nn.BatchNorm2d(out_channels[i + 1]))
            self.net.add_module(
                f"leaky_relu_{i}", nn.LeakyReLU(0.2, inplace=True)
            )
        Hout, Wout = get_height_width_conv(
            Hin=Hin,
            Win=Win,
            kernels=kernels,
            paddings=paddings,
            strides=strides,
            dilations=dilations,
        )
        out_in = out_channels[-1] * Hout * Wout
        self.out_layer = nn.Sequential(
            nn.Flatten(), nn.Linear(out_in, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.out_layer(self.net(x))
