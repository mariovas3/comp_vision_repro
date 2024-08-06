import random
from typing import Literal

import torch

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

    def get_centroids_height_and_width(self):
        # means are of format (xmin, ymin, xmax, ymax)
        pw = self.means[:, -2] - self.means[:, 0]
        ph = self.means[:, -1] - self.means[:, 1]
        assert torch.all(ph > 0) and torch.all(pw > 0)
        return ph, pw

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


def get_all_boxes(dataset):
    all_boxes = []
    for _, info in dataset:
        _, boxes, _ = get_labels_and_boxes_and_size(info["annotation"])
        all_boxes.extend(boxes)
    return all_boxes
