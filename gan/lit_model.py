import math

import lightning as L
import torch
import torch.nn.functional as F
from utils import DiscriminatorCNN, GeneratorMLP

import wandb

# this is for reproducibility if using stochastic components
# like ConvTranspose;
torch.use_deterministic_algorithms(True)


class LitGan(L.LightningModule):
    def __init__(
        self,
        latent_dim: int,
        lr: float,
        num_discriminator_grad_steps: int,
        cfg: dict,
        num_latents_to_sample: int = None,
        gradient_clip_val: float = None,
        gradient_clip_algorithm: str = "norm",
    ):
        super().__init__()
        # Important: set auto optim to False since will have more than
        # one optimiser and will be ambiguous how to automate it;
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.num_latents_to_sample = num_latents_to_sample
        self.num_discriminator_grad_steps = num_discriminator_grad_steps
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm
        self.G = GeneratorMLP(latent_dim=latent_dim, **cfg["generator"])
        self.D = DiscriminatorCNN()

    def configure_optimizers(self):
        return (
            torch.optim.Adam(self.G.parameters(), lr=self.hparams.lr),
            torch.optim.Adam(self.D.parameters(), lr=self.hparams.lr),
        )

    def forward(self, n_samples):
        z = torch.randn(
            (n_samples, self.hparams.latent_dim), device=self.device
        )
        return self.G(z)

    def on_train_batch_end(self, outputs, batch: torch.Any, batch_idx: int):
        k = self.num_discriminator_grad_steps
        # log this after every generator grad step;
        if (self.trainer.global_step + 1) % k == 0:
            x, _ = batch
            latent_batch_size = self.num_latents_to_sample or len(x)
            z = torch.randn(
                (latent_batch_size, self.hparams.latent_dim),
                device=self.device,
            )
            real_targets = torch.ones((len(x), 1), device=self.device)
            fake_targets = torch.zeros((len(z), 1), device=self.device)
            real_probs = self.D(x)
            fake_probs = self.D(self.G(z))
            real_loss = F.binary_cross_entropy(real_probs, real_targets)
            fake_loss = F.binary_cross_entropy(fake_probs, fake_targets)
            # at optimum, the gan objective should be log(1/4)
            # so this metric measures gan_obj - log(1/4) and this
            # should be >= 0 since log(1/4) should be min of minimax
            # if everything trains to convergence.
            dist_from_optimum = -(real_loss + fake_loss) - math.log(1 / 4)
            self.log("gan_obj_minus_log025", dist_from_optimum.item())

    def on_train_epoch_end(self):
        imgs = self(8).detach()
        imgs = wandb.Image(imgs, caption=f"end of epoch: {self.current_epoch}")
        wandb.log({"generator_samples": imgs})

    def training_step(self, batch):
        # the optims are wrapped in LightningOptimizer
        # to be able to handle all quirks of training;
        optimG, optimD = self.optimizers()
        x, _ = batch
        latent_batch_size = self.num_latents_to_sample or len(x)
        # sample latents from standard Gauss;
        z = torch.randn(
            (latent_batch_size, self.hparams.latent_dim), device=self.device
        )
        # train discriminator;
        # track only params of D;
        self.toggle_optimizer(optimizer=optimD)
        optimD.zero_grad()
        # get probs of x being real;
        out_probs_x = self.D(x)
        # get mean(-log(D(x)))
        real_targets = torch.ones((len(out_probs_x), 1), device=self.device)
        real_loss = F.binary_cross_entropy(
            input=out_probs_x,
            target=real_targets,
        )
        # get probs of G(z) being real;
        out_probs_fake = self.D(self.G(z).detach())
        # get mean(- log(1 - D(G(z))))
        fake_targets = torch.zeros(
            (len(out_probs_fake), 1), device=self.device
        )
        fake_loss = F.binary_cross_entropy(
            input=out_probs_fake,
            target=fake_targets,
        )
        # arithmetic avg the have similar scale to generator_loss;
        loss = (real_loss + fake_loss) / 2
        self.log("discriminator_loss", loss.item())
        self.manual_backward(loss)
        optimD.step()
        self.untoggle_optimizer(optimD)

        k = self.num_discriminator_grad_steps
        if (self.trainer.global_step + 1) % k == 0:
            # train generator;
            # tell lightning to only track grads for
            # the params of the optimiser passed to
            # toggle_optimizer;
            self.toggle_optimizer(optimizer=optimG)
            optimG.zero_grad()
            # get probs that G(z) is a real datapoint;
            out_probs = self.D(self.G(z))
            # want to trick D that G(z) are real;
            targets = torch.ones((len(out_probs), 1), device=self.device)
            # get Binary Cross Entropy = - log(D(G(z)))
            loss = F.binary_cross_entropy(input=out_probs, target=targets)
            self.log("generator_loss", loss.item())
            # safe way to do backward so that lightning
            # can handle mixed precision training;
            self.manual_backward(loss)
            # check if should clip grads of Generator;
            # here I do it manually bc configure_gradient_clipping()
            # won’t be called in Manual Optimization
            if self.gradient_clip_val is not None:
                self.clip_gradients(
                    optimG,
                    gradient_clip_val=self.gradient_clip_val,
                    gradient_clip_algorithm=self.gradient_clip_algorithm,
                )
            optimG.step()
            self.untoggle_optimizer(optimG)
