from collections import deque
from typing import Optional

import matplotlib.pyplot as plt
import torch
from PIL.Image import Image

import yolo.data.utils as dutils


def vis_boxes_labels(
    boxes, labels=None, title=None, box_color="r", is_gt=False
):
    ax = plt.gca()
    for i in range(len(boxes)):
        ax.vlines(
            x=[boxes[i][0], boxes[i][2]],
            ymin=boxes[i][1],
            ymax=boxes[i][-1],
            color=box_color,
        )
        ax.hlines(
            y=[boxes[i][1], boxes[i][3]],
            xmin=boxes[i][0],
            xmax=boxes[i][2],
            colors=box_color,
        )
        if labels is not None:
            text_x_coord = (boxes[i][0] + 3) if is_gt else (boxes[i][2] - 4)
            ax.text(
                text_x_coord,  # xmax
                boxes[i][1] + 3,  # ymin;
                labels[i],
                color="white",
                fontsize="small",
                backgroundcolor=box_color,
                ha="left" if is_gt else "right",
                va="top",
            )
    if title is not None:
        ax.set_title(title)


def vis_boxes(
    img: Image,
    boxes: list,
    labels: Optional[list] = None,
    title: Optional[str] = None,
    box_color: str = "r",
    is_gt=True,
):
    ax = plt.gca()
    ax.imshow(img)
    vis_boxes_labels(
        boxes=boxes,
        labels=labels,
        title=title,
        box_color=box_color,
        is_gt=is_gt,
    )


def see_grid_and_centers(img, voc_annotation, grid_dim):
    labels, boxes, size = dutils.get_labels_and_boxes_and_size(voc_annotation)
    print(labels)
    ax = plt.gca()
    vis_boxes(img, boxes, labels, show_labels=False)
    # add the 7x7 grid;
    ax.hlines(
        y=[size["height"] / grid_dim * i for i in range(1, grid_dim)],
        xmin=0,
        xmax=size["width"],
    )
    ax.vlines(
        x=[size["width"] / grid_dim * i for i in range(1, grid_dim)],
        ymin=0,
        ymax=size["height"],
    )
    # get the centers;
    for box in boxes:
        x = (box[0] + box[2]) / 2
        y = (box[1] + box[3]) / 2
        ax.scatter((x,), (y,), marker="*", s=50, color="green")


def get_nms_boxes(
    boxes: torch.Tensor,
    iou_thres: float,
    conf_thres: Optional[float] = None,
    midpoint=True,
) -> torch.Tensor:
    """
    conf_thres: keep boxes who have objects in them with confidence > conf_thres.
        It is possible to get no boxes if conf_thres is too high, so you should
        handle the case of no nms boxes being returned in the rest of your
        codebase.
    midpoint: if True, each element of boxes is (prob_score, x, y, w, h, class_pred)
        otherwise it is (prob_score, x1, y1, x2, y2, class_pred)
    """
    if conf_thres is not None:
        assert 0 <= conf_thres <= 1
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
                    curr[1:-1].unsqueeze(0),
                    box[1:-1].unsqueeze(0),
                    midpoint=midpoint,
                )
                .squeeze()
                .item()
                < iou_thres
            )
        )
        nms_boxes.append(curr)
    return torch.stack(nms_boxes)


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
    inter_tl_x = torch.maximum(boxes1[..., 0], boxes2[..., 0])
    inter_tl_y = torch.maximum(boxes1[..., 1], boxes2[..., 1])
    inter_br_x = torch.minimum(boxes1[..., 2], boxes2[..., 2])
    inter_br_y = torch.minimum(boxes1[..., 3], boxes2[..., 3])

    # Compute the width and height of the intersection rectangle
    inter_w = torch.maximum(torch.tensor(0), inter_br_x - inter_tl_x)
    inter_h = torch.maximum(torch.tensor(0), inter_br_y - inter_tl_y)

    # Compute the area of the intersection rectangle
    inter_area = inter_w * inter_h

    # Compute the area of each bounding box
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (
        boxes1[..., 3] - boxes1[..., 1]
    )
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (
        boxes2[..., 3] - boxes2[..., 1]
    )

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
