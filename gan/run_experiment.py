import lightning as L
from lightning.pytorch.cli import LightningCLI
from lit_mnist import MNISTDataModule
from lit_model import LitGan


def main():
    cli = LightningCLI(
        model_class=LitGan,
        datamodule_class=MNISTDataModule,
        seed_everything_default=0,
    )


if __name__ == "__main__":
    main()
