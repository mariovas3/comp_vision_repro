from collections import deque

import matplotlib.pyplot as plt
import torch
from PIL.Image import Image
from torchvision.ops import nms as tv_nms


def vis_boxes(img: Image, boxes: list, labels: list, show_labels=True):
    ax = plt.gca()
    ax.imshow(img)
    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box
        ax.vlines(x=[xmin, xmax], ymin=ymin, ymax=ymax, color="r")
        ax.hlines(y=[ymin, ymax], xmin=xmin, xmax=xmax, color="r")
        if show_labels:
            # offset text bbox by 3;
            ax.text(
                xmin + 3,
                ymin + 3,
                label,
                color="white",
                fontsize="small",
                backgroundcolor="red",
                ha="left",
                va="top",
            )


def get_nms_boxes(boxes: list, iou_thres: float, conf_thres: float):
    """
    boxes (list): each element is (prob_score, x1, y1, x2, y2, class_pred)
    """
    boxes = [box for box in boxes if box[0] > conf_thres]
    # in descending order of prob_score;
    boxes = deque(sorted(boxes, key=lambda x: x[0], reverse=True))
    nms_boxes = []

    while boxes:
        # popleft on deque;
        curr = boxes.popleft()

        # keep boxes from different class or if iou with them is low;
        boxes = deque(
            box
            for box in boxes
            if (
                box[-1] != curr[-1]
                or get_IoU(
                    torch.tensor((curr[1:-1],)),
                    torch.tensor((box[1:-1],)),
                    midpoint=False,
                )
                < iou_thres
            )
        )
        nms_boxes.append(curr)
    return nms_boxes


def get_IoU(boxes1, boxes2, midpoint=True):
    """
    boxes1 (Tensor): is (batch_size, 4) shape;
    boxes2 (Tensor): is (batch_size, 4) shape;
    midpoint (bool): If True, the boxes' columns
    should be (x, y, w, h), otherwise they
    should be (xmin, ymin, xmax, ymax);
    """
    if midpoint:
        boxes1 = midpoint_box_to_corners(boxes1)
        boxes2 = midpoint_box_to_corners(boxes2)

    # Determine the coordinates of the intersection rectangle
    inter_tl_x = torch.maximum(boxes1[:, 0], boxes2[:, 0])
    inter_tl_y = torch.maximum(boxes1[:, 1], boxes2[:, 1])
    inter_br_x = torch.minimum(boxes1[:, 2], boxes2[:, 2])
    inter_br_y = torch.minimum(boxes1[:, 3], boxes2[:, 3])

    # Compute the width and height of the intersection rectangle
    inter_w = torch.maximum(torch.tensor(0), inter_br_x - inter_tl_x)
    inter_h = torch.maximum(torch.tensor(0), inter_br_y - inter_tl_y)

    # Compute the area of the intersection rectangle
    inter_area = inter_w * inter_h

    # Compute the area of each bounding box
    boxes1_area = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    boxes2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    union_area = boxes1_area + boxes2_area - inter_area
    return inter_area / union_area


def midpoint_box_to_corners(boxes: torch.Tensor):
    """
    Assumes boxes columns are (x_ceneter, y_center, width, height)
    and returns new tensor with (xmin, ymin, xmax, ymax) columns;
    """
    xmin = boxes[..., 0:1] - boxes[..., 2:3] / 2
    xmax = boxes[..., 0:1] + boxes[..., 2:3] / 2
    ymin = boxes[..., 1:2] - boxes[..., 3:4] / 2
    ymax = boxes[..., 1:2] + boxes[..., 3:4] / 2
    return torch.cat((xmin, ymin, xmax, ymax), -1)
