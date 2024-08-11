from typing import Any, Mapping

import torchvision.transforms as T
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch import Tensor
from torchvision.datasets import VOCDetection

import wandb
from yolo.metadata import metadata
from yolo.model import utils

VAL_DATA = VOCDetection(
    root=metadata.DATA_DIR, year="2007", download=True, image_set="val"
)
VAL_DATA = [VAL_DATA[i] for i in range(9)]


class LogValDetectionsCallback(Callback):
    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if batch_idx == 0 and dataloader_idx == 0:
            logger = trainer.logger
            img_transform = trainer.val_dataloaders.dataset.img_transform
            resize_size = img_transform.transforms[0].size[0]
            crop_size = img_transform.transforms[1].size[0]
            plot_transform = T.Compose(
                [
                    T.Resize((resize_size,)),
                    T.CenterCrop((crop_size, crop_size)),
                ]
            )

            # plots a 3 by 3 grid of images with detections;
            fig = utils.plot_gt_yolo(
                one_bbox_per_grid=outputs,
                val_data=VAL_DATA,
                idx_to_label=metadata.IDX_TO_LABEL,
                resize_size=resize_size,
                crop_size=crop_size,
                plot_transform=plot_transform,
            )
            wandb.log({"val/detections": wandb.Image(fig)})
