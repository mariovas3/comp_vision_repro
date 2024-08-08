import os
import pickle
from itertools import chain
from pathlib import Path
from typing import List

import smart_open
import torch
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection

from yolo.data import utils
from yolo.metadata import metadata

os.environ["TORCH_HOME"] = str(metadata.SAVED_MODELS_PATH)


class LitVOCData(LightningDataModule):
    def __init__(
        self,
        batch_size=64,
        num_workers=4,
        pin_memory=True,
        years=("2007",),
        ignore_multibox=True,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.ignore_multibox = ignore_multibox
        self.years = years
        assert self.years == ("2007",) or self.years == ("2007", "2012")

    def prepare_data(self):
        if metadata.ANCHOR_DIMS_PATH.exists():
            print("ANCHOR DIMS PATH EXISTS!\nDATA PREP DONE!")
            return

        print(f"DOWNLOADING VOC DATA...")
        if not (metadata.DATA_DIR / "train_gt.pkl").exists():
            save_imgs_and_labels(
                split="train",
                years=self.years,
                ignore_multibox=self.ignore_multibox,
            )
            save_imgs_and_labels(
                split="val", years=self.years, ignore_multibox=False
            )
            save_imgs_and_labels(
                split="test", years=("2007",), ignore_multibox=False
            )

        save_anchor_box_dims(years=self.years)
        print("DATA PREP DONE!")

    def setup(self, stage="fit"):
        if stage == "fit":
            train_imgs = load_imgs(metadata.DATA_DIR / "train_imgs")
            train_gt = utils.load_pickle(metadata.DATA_DIR / "train_gt.pkl")
            val_imgs = load_imgs(metadata.DATA_DIR / "val_imgs")
            val_gt = utils.load_pickle(metadata.DATA_DIR / "val_gt.pkl")
            self.train_dataset = utils.VocDataset(
                imgs=train_imgs,
                label_matrices=train_gt,
                img_transform=utils.IMG_TRANSFORM_224,
            )
            self.val_dataset = utils.VocDataset(
                imgs=val_imgs,
                label_matrices=val_gt,
                img_transform=utils.IMG_TRANSFORM_224,
            )
        else:
            test_imgs = load_imgs(metadata.DATA_DIR / "test_imgs")
            test_gt = utils.load_pickle(metadata.DATA_DIR / "test_gt.pkl")
            self.test_dataset = utils.VocDataset(
                imgs=test_imgs,
                label_matrices=test_gt,
                img_transform=utils.IMG_TRANSFORM_224,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )


def save_imgs(imgs: List[Image.Image], dir_path: Path):
    dir_path.mkdir(parents=True, exist_ok=True)
    for idx, im in enumerate(imgs):
        im.save(dir_path / f"{idx}.jpg")


def read_pil(filename):
    with smart_open.open(filename, "rb") as f:
        with Image.open(f) as image:
            image = image.convert(image.mode)
            return image


def load_imgs(dir_path: Path):
    img_names = sorted(
        dir_path.glob("*.jpg"), key=lambda filename: int(Path(filename).stem)
    )
    return [read_pil(filename) for filename in img_names]


def save_imgs_and_labels(split="train", years=("2007",), ignore_multibox=True):
    imgs, labels = [], []
    for year in years:
        data = VOCDetection(
            root=metadata.DATA_DIR, year=year, download=True, image_set=split
        )
        ims, labs, _ = utils.get_all_img_label_matrices(
            data,
            grid_dim=metadata.GRID_DIM,
            num_bbox_elements=5,
            label_to_idx=metadata.LABEL_TO_IDX,
            ignore_multibox=ignore_multibox,
        )
        imgs.extend(ims)
        labels.extend(labs)
    del data
    print(f"SAVING {split} DATA...")
    # save imgs as {idx}.jpg in the dir_path;
    save_imgs(imgs, metadata.DATA_DIR / f"{split}_imgs")
    utils.save_to_pickle(labels, metadata.DATA_DIR / f"{split}_gt.pkl")


def save_anchor_box_dims(years=("2007",)):
    train_sets = []
    for year in years:
        train_data = VOCDetection(
            root=metadata.DATA_DIR, year=year, download=True, image_set="train"
        )
        train_sets.append(train_data)

    # get all boxes in x1, y1, x2, y2 format;
    all_boxes = torch.tensor(utils.get_all_boxes(chain(*train_sets)))

    print(f"RUNNING KMEANS FOR ANCHOR BOX PRIORS...")
    kmeans = utils.Kmeans(
        k=metadata.NUM_BBOXES,
        dist_metric="iou",
        data=all_boxes,
        num_runs=10,
        determ=True,
        verbose=False,
    )
    kmeans.fit()
    pw, ph = kmeans.get_centroids_width_and_height()

    print(f"SAVING ANCHOR BOX DIMS...")
    utils.save_to_json(
        {
            "pw": pw.tolist(),
            "ph": ph.tolist(),
        },
        filepath=metadata.DATA_DIR / "anchor_dims.json",
        indent=2,
    )


if __name__ == "__main__":
    dm = LitVOCData(
        batch_size=64,
        num_workers=1,
        pin_memory=False,
        years=("2007",),
        ignore_multibox=True,
    )
    dm.prepare_data()
    dm.setup("fit")
    tr_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    x, y = next(iter(tr_loader))
    assert x.shape == (64, 3, 224, 224)
    assert y.shape == (64, 7, 7, 25)
