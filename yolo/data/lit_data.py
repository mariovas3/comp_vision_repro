import os
from itertools import chain
from pathlib import Path
from typing import List

import torch
import torchvision.transforms as T
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
        grid_dim=7,
        pin_memory=True,
        years=("2007",),
        ignore_multibox=True,
    ):
        super().__init__()
        self.grid_dim = grid_dim
        assert grid_dim <= 13, f"{grid_dim=}, but should be <= 13"
        self.crop_size = grid_dim * 32
        self.resize_size = self.crop_size + 8
        self.img_transform = get_img_transform(
            resize_size=self.resize_size, crop_size=self.crop_size
        )
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
                grid_dim=self.grid_dim,
                resize_size=self.resize_size,
                split="train",
                years=self.years,
                ignore_multibox=self.ignore_multibox,
                crop_size=self.crop_size,
            )
            save_imgs_and_labels(
                grid_dim=self.grid_dim,
                resize_size=self.resize_size,
                split="val",
                years=self.years,
                ignore_multibox=False,
                crop_size=self.crop_size,
            )
            save_imgs_and_labels(
                grid_dim=self.grid_dim,
                resize_size=self.resize_size,
                split="test",
                years=("2007",),
                ignore_multibox=False,
                crop_size=self.crop_size,
            )

        save_anchor_box_dims(
            resize_size=self.resize_size,
            years=self.years,
            crop_size=self.crop_size,
        )
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
                img_transform=self.img_transform,
            )
            self.val_dataset = utils.VocDataset(
                imgs=val_imgs,
                label_matrices=val_gt,
                img_transform=self.img_transform,
            )
        else:
            test_imgs = load_imgs(metadata.DATA_DIR / "test_imgs")
            test_gt = utils.load_pickle(metadata.DATA_DIR / "test_gt.pkl")
            self.test_dataset = utils.VocDataset(
                imgs=test_imgs,
                label_matrices=test_gt,
                img_transform=self.img_transform,
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


def get_img_transform(resize_size, crop_size):
    return T.Compose(
        [
            T.Resize(resize_size),
            T.CenterCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def save_imgs(imgs: List[Image.Image], dir_path: Path):
    dir_path.mkdir(parents=True, exist_ok=True)
    for idx, im in enumerate(imgs):
        im.save(dir_path / f"{idx}.jpg")


def load_imgs(dir_path: Path):
    img_names = sorted(
        dir_path.glob("*.jpg"), key=lambda filename: int(Path(filename).stem)
    )
    return [utils.read_pil(filename) for filename in img_names]


def save_imgs_and_labels(
    grid_dim,
    resize_size,
    split="train",
    years=("2007",),
    ignore_multibox=True,
    crop_size=224,
):
    imgs, labels = [], []
    for year in years:
        data = VOCDetection(
            root=metadata.DATA_DIR, year=year, download=True, image_set=split
        )
        ims, labs, _ = utils.get_all_img_label_matrices(
            data,
            grid_dim=grid_dim,
            num_bbox_elements=5,
            label_to_idx=metadata.LABEL_TO_IDX,
            resize_size=resize_size,
            ignore_multibox=ignore_multibox,
            crop_size=crop_size,
        )
        imgs.extend(ims)
        labels.extend(labs)
    del data
    print(f"SAVING {split} DATA...")
    # save imgs as {idx}.jpg in the dir_path;
    save_imgs(imgs, metadata.DATA_DIR / f"{split}_imgs")
    utils.save_to_pickle(labels, metadata.DATA_DIR / f"{split}_gt.pkl")


def save_anchor_box_dims(resize_size, years=("2007",), crop_size=224):
    train_sets = []
    for year in years:
        train_data = VOCDetection(
            root=metadata.DATA_DIR, year=year, download=True, image_set="train"
        )
        train_sets.append(train_data)

    # get all boxes in x1, y1, x2, y2 format;
    all_boxes = torch.tensor(
        utils.get_all_boxes(
            chain(*train_sets),
            resize_size=resize_size,
            scale_box_dims=True,
            crop_size=crop_size,
        ),
        dtype=torch.float32,
    )

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
        grid_dim=7,
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
    y = y.view(-1, 25)
    xmin = y[:, 1].min().item()
    xmax = y[:, 1].max().item()
    ymin = y[:, 2].min().item()
    ymax = y[:, 2].max().item()
    wmin = y[:, 3].min().item()
    wmax = y[:, 3].max().item()
    hmin = y[:, 4].min().item()
    hmax = y[:, 4].max().item()
    print(
        f"{xmin=}, {xmax=}, {ymin=}, {ymax=}, {wmin=}, {wmax=}, {hmin=}, {hmax=}"
    )
