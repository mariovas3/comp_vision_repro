import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI
from lit_mnist import MNISTDataModule
from lit_model import LitGan


class MyLitCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        # make ModelCheckpoint callback configurable;
        parser.add_lightning_class_args(ModelCheckpoint, "my_model_checkpoint")
        parser.set_defaults({"my_model_checkpoint.every_n_epochs": 50})


def main():
    cli = MyLitCLI(
        model_class=LitGan,
        datamodule_class=MNISTDataModule,
        seed_everything_default=0,
    )


if __name__ == "__main__":
    main()
