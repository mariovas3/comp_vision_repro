import json
import pickle
import random
from pathlib import Path
from typing import Literal, MutableSequence, Tuple

import smart_open
import torch
from numpy import clip
from PIL import Image

from yolo.metadata import metadata
from yolo.model import eval_utils


def get_labels_and_boxes_and_size(voc_annotation: dict):
    labels, boxes = [], []
    img_size = {_: int(val) for _, val in voc_annotation["size"].items()}
    for item in voc_annotation["object"]:
        labels.append(item["name"])
        boxes.append(
            [
                float(item["bndbox"]["xmin"]),
                float(item["bndbox"]["ymin"]),
                float(item["bndbox"]["xmax"]),
                float(item["bndbox"]["ymax"]),
            ]
        )
    return labels, boxes, img_size


class Kmeans:
    def __init__(
        self,
        k,
        dist_metric: Literal["iou", "euclid"],
        data: torch.Tensor,
        num_runs=10,
        maxiter=100,
        tol=1e-4,
        determ=False,
        verbose=False,
    ):
        """data expected to be in corner format (xmin, ymin, xmax, ymax)."""
        self.data = data
        self.k = k
        self.num_inits = 10
        if determ:
            random.seed(0)
        self.num_runs = num_runs
        self.means = None
        self.dist_metric = dist_metric
        self.assignments = None
        self.maxiter = maxiter
        self.tol = tol
        self.best_score = 1e5
        self.verbose = verbose

    def get_centroids_width_and_height(self):
        # means are of format (xmin, ymin, xmax, ymax)
        pw = self.means[:, -2] - self.means[:, 0]
        ph = self.means[:, -1] - self.means[:, 1]
        assert torch.all(ph > 0) and torch.all(pw > 0)
        return pw, ph

    def get_iou_dist(self, means) -> torch.Tensor:
        ious = torch.zeros((len(self.data), len(means)))
        for i, m in enumerate(means):
            m = m.view(1, -1).expand(len(self.data), -1)
            ious[:, i] = eval_utils.get_IoU(self.data, m, midpoint=False)
        return 1 - ious

    def update_means_(self, means, assignments):
        for i in range(len(means)):
            mask = assignments == i
            if torch.any(mask):
                means[i] = self.data[mask, :].mean(0)

    def fit(self):
        while self.num_runs:
            means = random.sample(range(len(self.data)), k=self.k)
            means = self.data[means]
            curr_score = -1
            for it in range(self.maxiter):
                if self.dist_metric == "iou":
                    dists = self.get_iou_dist(means)
                else:
                    dists = torch.norm(
                        self.data.unsqueeze(1) - means.unsqueeze(0),
                        dim=-1,
                        p=2,
                    )
                vals, idxs = dists.min(-1)
                score = vals.mean()
                if abs(curr_score - score) < self.tol:
                    if self.verbose:
                        print(
                            f"{self.dist_metric} k means converged step: {it}, score: {score}"
                        )
                    break
                if it < 10:
                    if self.verbose:
                        print(score)
                    # assert score <= curr_score
                curr_score = score
                assignments = idxs
                self.update_means_(means, assignments)
            if self.verbose:
                print(f"{self.dist_metric} k means max iter reached")
            if self.best_score > score:
                self.best_score = score
                self.means = means
                self.assignments = idxs
            self.num_runs -= 1


def get_all_boxes(dataset, resize_size, scale_box_dims=False, crop_size=224):
    all_boxes = []
    for _, info in dataset:
        _, boxes, img_size = get_labels_and_boxes_and_size(info["annotation"])
        if scale_box_dims:
            new_boxes = []
            for i, box in enumerate(boxes):
                # get XYXY after T.Resize and T.CenterCrop applied
                # to image.
                box, bad_crop = resize_and_crop_bbox(
                    bbox=box,
                    old_w=img_size["width"],
                    old_h=img_size["height"],
                    resize_size=resize_size,
                    crop_size=crop_size,
                )
                if bad_crop:
                    continue
                new_boxes.append(box)
            boxes = new_boxes
        all_boxes.extend(boxes)
    return all_boxes


def load_json(filepath: Path):
    with open(filepath, "r") as file:
        obj = json.load(file)
    return obj


def save_to_json(obj, filepath: Path, **kwargs):
    parent_dir = filepath.parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as file:
        json.dump(obj, file, **kwargs)


def save_to_pickle(obj, filepath: Path):
    parent_dir = filepath.parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(filepath: Path):
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    return obj


def read_pil(filename):
    with smart_open.open(filename, "rb") as f:
        with Image.open(f) as image:
            image = image.convert(image.mode)
            return image


def get_unique_labels(dataset) -> set[str]:
    labels = set()
    for data in dataset:
        voc_annotation = data[-1]["annotation"]
        for item in voc_annotation["object"]:
            labels.add(item["name"])
    return labels


def get_targets(
    voc_annotation: dict,
    grid_dim,
    num_bbox_elements,
    label_to_idx: dict,
    resize_size: int,
    crop_size: int,
):
    """
    Given VOC annotation return grid_dim x grid_dim x out_channels label matrix.

    The out_channels are num_bbox_elements + num_classes.
    """
    img_size = {_: int(val) for _, val in voc_annotation["size"].items()}
    assert img_size["depth"] == 3
    num_classes = len(label_to_idx)
    multiple_boxes_in_grid_cell = False
    new_img_size = {"width": crop_size, "height": crop_size}
    ignore_bad_crop = True

    # assume only one target box per grid cell;
    label_matrix = torch.zeros(
        (grid_dim, grid_dim, num_bbox_elements + num_classes)
    )
    for item in voc_annotation["object"]:
        label_idx = label_to_idx[item["name"]]
        coords = [
            float(item["bndbox"]["xmin"]),
            float(item["bndbox"]["ymin"]),
            float(item["bndbox"]["xmax"]),
            float(item["bndbox"]["ymax"]),
        ]
        # calculates new bbox XYXY coords after
        # T.Compose([T.Resize(resize_size), T.CenterCrop(crop_size)])
        # are applied to img;
        coords, bad_crop = resize_and_crop_bbox(
            bbox=coords,
            old_w=img_size["width"],
            old_h=img_size["height"],
            resize_size=resize_size,
            crop_size=crop_size,
        )
        # don't add this box if cropping left too much out;
        if bad_crop:
            continue
        else:
            ignore_bad_crop = False
        # get center coords
        coords = corners_to_midpoint(coords)

        # i and j are index of cell that contains the target box center.
        i, j = get_grid_cell_idxs(
            xcenter=coords[0],
            ycenter=coords[1],
            img_size=new_img_size,
            grid_dim=grid_dim,
        )
        # will also keep track if we have multiple target boxes in
        # single grid cell for the current image.
        if label_matrix[i, j, 0] == 1:
            multiple_boxes_in_grid_cell = True
        # add label to relevant grid cell;
        if label_matrix[i, j, -num_classes + label_idx] == 0:
            label_matrix[i, j, -num_classes + label_idx] = 1  # one-hot class;
            label_matrix[i, j, 0] = 1  # object exists;
            label_matrix[i, j, 1:num_bbox_elements] = torch.tensor(coords)
    return label_matrix, multiple_boxes_in_grid_cell, ignore_bad_crop


def get_resized_wh(old_w: int, old_h: int, size: int):
    """
    Sets the smaller dim to size and the other one
    to int(size * bigger / smaller)
    """
    if old_w < old_h:
        new_w = size
        new_h = old_h / old_w * size
    else:
        new_h = size
        new_w = old_w / old_h * size
    return int(new_w), int(new_h)


def resize_and_crop_bbox(
    bbox: MutableSequence,
    old_w: int,
    old_h: int,
    resize_size: int,
    crop_size: int,
) -> Tuple[MutableSequence, bool]:
    """
    Get the coords of bbox after resize and crop and bool indicator
    if bbox is usable.

    bbox should be in xmin, ymin, xmax, ymax format.

    Return: (bbox, bad_crop)

    Since we are resizing first and then cropping, it is possible
    we lose a big part of the bbox if it was close to the edge
    of the image. If the width or height decrease by more than
    40%, we return bad_crop=True alongside the bbox, otherwise
    bad_crop=False.
    """
    assert len(bbox) == 4
    new_w, new_h = get_resized_wh(old_w, old_h, size=resize_size)
    w_mult = new_w / old_w
    # shifting is for the crop operation since
    # we adjust coords by subtracting half ignored region
    # after the crop;
    w_shift = (new_w - crop_size) / 2
    h_mult = new_h / old_h
    h_shift = (new_h - crop_size) / 2
    # adjust for resizing;
    bbox[0], bbox[2] = bbox[0] * w_mult, bbox[2] * w_mult
    bbox[1], bbox[3] = bbox[1] * h_mult, bbox[3] * h_mult
    old_bw, old_bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    # adjust for cropping;
    for i, c in enumerate(bbox):
        if i % 2 == 0:
            bbox[i] = clip(0, int(c - w_shift), crop_size)
            assert (
                bbox[i] <= crop_size
            ), f"{c=}, {bbox[i]=}, {crop_size=}, {new_w=}, {old_w=}, {old_h=}"
        else:
            bbox[i] = clip(0, int(c - h_shift), crop_size)
            assert (
                bbox[i] <= crop_size
            ), f"{c=}, {bbox[i]=}, {crop_size=}, {new_h=}, {old_w=}, {old_h=}"
    new_bw, new_bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    # don't add this box if cropping left too much out;
    bad_crop = new_bw < old_bw * 0.6 or new_bh < old_bh * 0.6
    return bbox, bad_crop


def corners_to_midpoint(coords):
    xmin, ymin, xmax, ymax = coords
    centerx = (xmax + xmin) // 2
    centery = (ymax + ymin) // 2
    boxwidth = xmax - xmin
    boxheight = ymax - ymin
    return centerx, centery, boxwidth, boxheight


def get_grid_cell_idxs(xcenter, ycenter, img_size, grid_dim) -> tuple[int]:
    """returns grid cell (i, j) coords."""
    # i and j are in (0, grid_dim)
    # i,j represents the cell row and cell column
    i = int(grid_dim * xcenter / img_size["width"])
    j = int(grid_dim * ycenter / img_size["height"])
    return i, j


def get_all_img_label_matrices(
    dataset,
    grid_dim,
    num_bbox_elements,
    label_to_idx,
    resize_size,
    ignore_multibox=False,
    crop_size=224,
):
    """
    Returns list of PIL imgs, list of label_matrix tensors
    and list of idxs of images where multibox labels were spotted.

    if ignore_multibox set to True, we don't return imgs or label_matrices
    for such examples.
    """
    label_matrices = []
    multi_box_idxs = []
    imgs = []
    bad_crop_count = 0

    for i, (img, info) in enumerate(dataset):
        # since we first resize and then center crop
        # it might be possible to decrease
        label_matrix, multi_box, ignore_bad_crop = get_targets(
            info["annotation"],
            grid_dim=grid_dim,
            num_bbox_elements=num_bbox_elements,
            label_to_idx=label_to_idx,
            resize_size=resize_size,
            crop_size=crop_size,
        )

        if multi_box:
            multi_box_idxs.append(i)
            if ignore_multibox:
                continue
        if ignore_bad_crop:
            bad_crop_count += 1
            continue
        label_matrices.append(label_matrix)
        imgs.append(img)
    print(f"Ignored bad crops: {bad_crop_count}")
    return imgs, label_matrices, multi_box_idxs


class VocDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        imgs,
        label_matrices,
        img_transform,
    ):
        super().__init__()
        self.imgs = imgs
        self.label_matrices = label_matrices
        self.img_transform = img_transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        label_matrix = self.label_matrices[idx]
        # label matrix is of shape
        # (grid_dim, grid_dim, num_bbox_elements + num_classes)
        return self.img_transform(img), label_matrix
