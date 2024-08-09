import json
import pickle
import random
from pathlib import Path
from typing import Literal

import torch

from yolo.metadata import metadata
from yolo.model import eval_utils


def get_labels_and_boxes_and_size(voc_annotation: dict):
    labels, boxes = [], []
    img_size = {_: int(val) for _, val in voc_annotation["size"].items()}
    for item in voc_annotation["object"]:
        labels.append(item["name"])
        boxes.append(
            (
                float(item["bndbox"]["xmin"]),
                float(item["bndbox"]["ymin"]),
                float(item["bndbox"]["xmax"]),
                float(item["bndbox"]["ymax"]),
            )
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


def get_all_boxes(dataset, scale_box_dims=False, standard_img_dim=224):
    all_boxes = []
    for _, info in dataset:
        _, boxes, img_size = get_labels_and_boxes_and_size(info["annotation"])
        if scale_box_dims:
            for i, box in enumerate(boxes):
                box = [
                    c
                    * standard_img_dim
                    / (
                        img_size["width"]
                        if c_i % 2 == 0
                        else img_size["height"]
                    )
                    for c_i, c in enumerate(box)
                ]
                boxes[i] = box
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


def get_unique_labels(dataset) -> set[str]:
    labels = set()
    for data in dataset:
        voc_annotation = data[-1]["annotation"]
        for item in voc_annotation["object"]:
            labels.add(item["name"])
    return labels


def corners_to_midpoint(*coords):
    xmin, ymin, xmax, ymax = coords
    centerx = (xmax + xmin) / 2
    centery = (ymax + ymin) / 2
    boxwidth = xmax - xmin
    boxheight = ymax - ymin
    return centerx, centery, boxwidth, boxheight


def midpoint_relative_to_grid(
    *coords, img_size, grid_dim, standard_img_dim=224
) -> tuple[tuple]:
    """returns Tuple of tuples - coords and grid cell (i, j) coords."""
    x, y, w, h = coords
    # x and y are in (0, grid_dim)
    x = grid_dim * x / img_size["width"]
    y = grid_dim * y / img_size["height"]
    # i,j represents the cell row and cell column
    i, j = int(y), int(x)
    # make x and y in pixel coords in standard image;
    x, y = x * standard_img_dim / grid_dim, y * standard_img_dim / grid_dim
    # w and h are in (0, standard_img_dim)
    # with standard_img_dim=224 for resnet50;
    w = w / img_size["width"] * standard_img_dim
    h = h / img_size["height"] * standard_img_dim
    return (x, y, w, h), (i, j)


def get_targets(
    voc_annotation: dict,
    grid_dim,
    num_bbox_elements,
    label_to_idx: dict,
    standard_img_dim=224,
):
    img_size = {_: int(val) for _, val in voc_annotation["size"].items()}
    assert img_size["depth"] == 3
    num_classes = len(label_to_idx)
    multiple_boxes_in_grid_cell = False

    # assume only one target box per grid cell;
    label_matrix = torch.zeros(
        (grid_dim, grid_dim, num_bbox_elements + num_classes)
    )
    for item in voc_annotation["object"]:
        label_idx = label_to_idx[item["name"]]
        coords = (
            float(item["bndbox"]["xmin"]),
            float(item["bndbox"]["ymin"]),
            float(item["bndbox"]["xmax"]),
            float(item["bndbox"]["ymax"]),
        )
        # relative x and y center coords are in (0, grid_dim)
        # the w and h are in (0, STANDARDISED_IMG_DIM)
        # i and j are index of cell that contains the target box center.
        coords, (i, j) = midpoint_relative_to_grid(
            *corners_to_midpoint(*coords),
            img_size=img_size,
            grid_dim=grid_dim,
            standard_img_dim=standard_img_dim,
        )
        # will also keep track if we have multiple target boxes in
        # single grid cell for the current image.
        if label_matrix[i, j, 0] == 1:
            multiple_boxes_in_grid_cell = True

        if label_matrix[i, j, -num_classes + label_idx] == 0:
            label_matrix[i, j, -num_classes + label_idx] = 1  # one-hot class;
            label_matrix[i, j, 0] = 1  # object exists;
            label_matrix[i, j, 1:num_bbox_elements] = torch.tensor(coords)
    return label_matrix, multiple_boxes_in_grid_cell


def get_all_img_label_matrices(
    dataset,
    grid_dim,
    num_bbox_elements,
    label_to_idx,
    ignore_multibox=False,
    standard_img_dim=224,
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

    for i, (img, info) in enumerate(dataset):
        label_matrix, multi_box = get_targets(
            info["annotation"],
            grid_dim=grid_dim,
            num_bbox_elements=num_bbox_elements,
            label_to_idx=label_to_idx,
            standard_img_dim=standard_img_dim,
        )

        if multi_box:
            multi_box_idxs.append(i)
            if ignore_multibox:
                continue
        label_matrices.append(label_matrix)
        imgs.append(img)
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
