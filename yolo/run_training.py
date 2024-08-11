from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI

from yolo.data.lit_data import LitVOCData
from yolo.model.lit_yolo_v2 import LitYoloV2


class MyLitCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.link_arguments(
            "trainer.accelerator",
            "data.pin_memory",
            compute_fn=lambda x: x == "gpu",
            apply_on="parse",
        )
        parser.link_arguments(
            "data.grid_dim", "model.grid_dim", apply_on="parse"
        )
        # make ModelCheckpoint callback configurable;
        parser.add_lightning_class_args(ModelCheckpoint, "my_model_checkpoint")
        parser.set_defaults(
            {
                "my_model_checkpoint.monitor": "val/loss",
                "my_model_checkpoint.mode": "min",
                "my_model_checkpoint.every_n_epochs": 50,
            }
        )


def main():
    cli = MyLitCLI(
        model_class=LitYoloV2,
        datamodule_class=LitVOCData,
        seed_everything_default=0,
    )


if __name__ == "__main__":
    main()
