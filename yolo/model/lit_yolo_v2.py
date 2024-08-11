import os

from yolo.metadata import metadata

os.environ["TORCH_HOME"] = str(metadata.SAVED_MODELS_PATH)
import torch
import torchvision.models as tv_models
from lightning import LightningModule

from yolo.data.utils import load_json
from yolo.model import eval_utils, utils


class LitYoloV2(LightningModule):
    def __init__(self, lr=1e-3, grid_dim=7, lam_noobj=0.5, lam_coord=5):
        super().__init__()
        self.save_hyperparameters()
        self.standard_img_dim = grid_dim * 32
        # instantiate loss;
        self.loss_fn = utils.YoloV2Loss(
            grid_dim=self.hparams["grid_dim"],
            num_bboxes=metadata.NUM_BBOXES,
            num_classes=len(metadata.LABEL_TO_IDX),
            lam_noobj=lam_noobj,
            lam_coord=lam_coord,
        )
        # get anchor boxes;
        anchor_boxes = load_json(metadata.DATA_DIR / "anchor_dims.json")
        self.anchor_boxes_wh = torch.cat(
            (
                torch.tensor(anchor_boxes["pw"]).view(1, -1),
                torch.tensor(anchor_boxes["ph"]).view(1, -1),
            ),
            0,
        )
        # instantiate model;
        weights = tv_models.ResNet50_Weights
        # the resnet compresses images by a factor of 32
        # so should work with dims that are multiples of 32;
        resnet50 = tv_models.resnet50(weights=weights.DEFAULT)
        self.model = utils.CombinedModel(
            resnet=resnet50,
            num_bboxes=metadata.NUM_BBOXES,
            num_bbox_elements=5,
            num_classes=len(metadata.LABEL_TO_IDX),
            anchor_boxes_wh=self.anchor_boxes_wh,
            standard_img_dim=self.standard_img_dim,
            grid_dim=grid_dim,
        )
        self.model.train()

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams["lr"])

    def _get_loss(self, img, targets, get_avg_iou=False):
        out = self.model.get_box_predictions(
            img, grid_dim=self.hparams["grid_dim"]
        )
        return self.loss_fn(
            out, targets, get_avg_iou=get_avg_iou, iou_box_selection=True
        )

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        img, label_matrices = batch
        loss = self._get_loss(img, label_matrices, get_avg_iou=False)
        self.log(
            "training/loss",
            value=loss.item(),
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        img, label_matrices = batch
        out = self.model.get_box_predictions(
            img, grid_dim=self.hparams["grid_dim"]
        )
        out = utils.greedy_confidence_box_selection(
            out, num_boxes=metadata.NUM_BBOXES
        )
        loss, avg_iou = self.loss_fn(
            out, label_matrices, get_avg_iou=True, iou_box_selection=False
        )
        self.log_dict(
            {
                "val/loss": loss.item(),
                "val/avg_iou": avg_iou.item(),
            },
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        return out

    def testing_step(self, batch, batch_idx):
        img, label_matrices = batch
        out = self.model.get_box_predictions(
            img, grid_dim=self.hparams["grid_dim"]
        )
        out = utils.greedy_confidence_box_selection(
            out, num_boxes=metadata.NUM_BBOXES
        )
        loss, avg_iou = self.loss_fn(
            out, label_matrices, get_avg_iou=True, iou_box_selection=False
        )
        self.log_dict(
            {
                "test/loss": loss.item(),
                "test/avg_iou": avg_iou.item(),
            },
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        return out
