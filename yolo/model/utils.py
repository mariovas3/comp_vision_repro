import math

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from torch import nn

import yolo.data.utils as dutils
from yolo.model import eval_utils


class CombinedModel(nn.Module):
    def __init__(
        self,
        resnet,
        num_bboxes,
        num_bbox_elements,
        num_classes,
        anchor_boxes_wh: torch.Tensor,
        standard_img_dim: int = 224,
    ):
        super().__init__()
        self.resnet = resnet
        self.register_buffer("anchor_boxes_wh", anchor_boxes_wh)
        # change from yolov1 output, now predict classes
        # in each bounding box;
        self.out_channels = num_bboxes * (num_bbox_elements + num_classes)
        self.conv_head = nn.Conv2d(2048, self.out_channels, kernel_size=1)
        self.num_bboxes = num_bboxes
        self.num_bbox_elements = num_bbox_elements
        self.num_classes = num_classes
        self.standard_img_dim = standard_img_dim

    def forward(self, x):
        # feats should be of size (batch, 2048, H / 32, W / 32)
        # we treat the grid is 7x7 so that each cell
        # of feats corresponds to the feats from the
        # relevant cell of the grid over the input image.
        feats = get_resnet_feats(self.resnet, x)
        # conv_head has kernel = (1, 1) and just remaps
        # the 2048 in channels to however many out channels
        # are needed - giving (B, out_channels, 7, 7)
        # out_channels is
        # num_bboxes * (has_object_entry + bbox_coords + num_classes)
        # permute dims to get output of (B, grid_dim, grid_dim, out_channels)
        return self.conv_head(feats).permute(0, -2, -1, 1)

    def get_box_predictions(self, x, grid_dim):
        out = self(x)
        offset = self.num_bbox_elements + self.num_classes
        # get confidence prob;
        out[..., ::offset] = torch.sigmoid(out[..., ::offset])
        # get x and y offsets from top left corner of grid cell;
        # if grid cell has idx (i, j) in the grid, the anchor box
        # center coords are bx, by = i + sigmoid(x), j + sigmoid(y)
        # then to remap to image pixels you do
        # img_x, img_y = bx * img_height / S, by * img_width / S
        out[..., 1::offset] = torch.sigmoid(out[..., 1::offset])
        out[..., 2::offset] = torch.sigmoid(out[..., 2::offset])
        # exponentiate width and height entries;
        # the bounding box width and height are then
        # bw, bh = pw * exp(w_entry), ph * exp(h_entry)
        # where pw and ph are the prior width and height of the
        # anchor box.
        out[..., 3::offset] = torch.exp(out[..., 3::offset])
        out[..., 4::offset] = torch.exp(out[..., 4::offset])
        # get softmax for classes
        B, S, _, _ = out.shape
        out = out.view(B, S, S, self.num_bboxes, -1)
        out[..., self.num_bbox_elements :] = torch.softmax(
            out[..., self.num_bbox_elements :], -1
        )
        out = out.view(B, S, S, -1)
        # map x and y to pixel values in a
        # (standard_img_dim x standard_img_dim) image
        output_to_bounding_boxes_xywh_(
            yolo_output=out,
            grid_dim=grid_dim,
            num_bboxes=self.num_bboxes,
            anchor_boxes_wh=self.anchor_boxes_wh,
            standard_img_dim=self.standard_img_dim,
        )
        return out


def output_to_bounding_boxes_xywh_(
    yolo_output: torch.Tensor,
    grid_dim: int,
    num_bboxes: int,
    anchor_boxes_wh: torch.Tensor,
    standard_img_dim: int = 224,
):
    """
    Inplace modify yolo output to be
    prob_obj, bx, by, bw, bh, softmax_over_classes as per yolov2 paper.
    bx and by will be in pixel coords of a standard_img_dim x standard_img_dim
    image.

    yolo_output: tensor of size (batch, grid_dim, grid_dim, out_len)
        where out_len is num_bboxes * (has_object + (x, y, w, h) + num_classes)
    anchor_boxes: tensor of size (2, num_anchor_boxes).
    """
    assert num_bboxes == anchor_boxes_wh.shape[-1]
    # grid_dim, grid_dim, 1
    width_grid_coords = (
        torch.arange(grid_dim).expand(grid_dim, -1).unsqueeze(-1)
    )
    # the 5 corresponds to (has_object, x, y, w, h)
    num_classes = yolo_output.shape[-1] // num_bboxes - 5
    offset = 5 + num_classes
    # get predicted center of bounding boxes;
    # x and y are in (0, grid_dim) * standard_img_dim / grid_dim;
    yolo_output[..., 1::offset] = (
        (yolo_output[..., 1::offset] + width_grid_coords)
        * standard_img_dim
        / grid_dim
    )
    yolo_output[..., 2::offset] = (
        (yolo_output[..., 2::offset] + width_grid_coords.permute(1, 0, 2))
        * standard_img_dim
        / grid_dim
    )
    # width and height are in anchor_width * exp(logit_w) and anchor_height * exp(logit_h)
    yolo_output[..., 3::offset] = (
        yolo_output[..., 3::offset] * anchor_boxes_wh[0]
    )
    yolo_output[..., 4::offset] = (
        yolo_output[..., 4::offset] * anchor_boxes_wh[1]
    )


def get_resnet_feats(model, x):
    """
    Got this from
    https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L266
    """
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    return x


def single_yolo_to_nms(
    one_bbox_per_grid, iou_thres=0.1, conf_thres=0.5, midpoint=True
):
    """The input must be a single image."""
    assert one_bbox_per_grid.ndim == 3
    class_preds = one_bbox_per_grid[..., 5:].argmax(-1, keepdim=True)
    nms_input = torch.cat((one_bbox_per_grid[..., :5], class_preds), -1)
    nms_dim = nms_input.shape[-1]
    nms_boxes = eval_utils.get_nms_boxes(
        nms_input.view(-1, nms_dim),
        iou_thres=iou_thres,
        conf_thres=conf_thres,
        midpoint=midpoint,
    )
    return nms_boxes


def yolo_to_one_bbox_per_grid(img_batch, model: CombinedModel, grid_dim=7):
    yolo_out = model.get_box_predictions(img_batch, grid_dim=grid_dim)
    return greedy_confidence_box_selection(yolo_out, num_boxes=5)


def plot_gt_yolo(
    one_bbox_per_grid,
    val_data,
    idx_to_label,
    resize_size,
    crop_size,
    plot_transform,
):
    fig = plt.figure(figsize=(12, 12))
    for i in range(9):
        labels, boxes, img_size = dutils.get_labels_and_boxes_and_size(
            val_data[i][-1]["annotation"]
        )
        boxes, bads = list(
            zip(
                *[
                    dutils.resize_and_crop_bbox(
                        box,
                        old_w=img_size["width"],
                        old_h=img_size["height"],
                        resize_size=resize_size,
                        crop_size=crop_size,
                    )
                    for box in boxes
                ]
            )
        )
        labels = [_ for i, _ in enumerate(labels) if not bads[i]]
        boxes = [box for i, box in enumerate(boxes) if not bads[i]]
        img = plot_transform(val_data[i][0])

        # get pred boxes;
        nms_boxes_val = single_yolo_to_nms(one_bbox_per_grid[i]).detach()
        corners = eval_utils.midpoint_box_to_corners(
            nms_boxes_val[: len(boxes), 1:-1]
        )
        nms_labels = [
            idx_to_label[_.int().item()] for _ in nms_boxes_val[..., -1]
        ]
        title = f"nms boxes returned: {len(nms_boxes_val)}"

        plt.subplot(3, 3, i + 1)
        eval_utils.vis_boxes(
            img=img,
            boxes=boxes,
            labels=labels,
            title=None,
            box_color="r",
            is_gt=True,
        )
        eval_utils.vis_boxes_labels(
            boxes=corners.tolist(),
            labels=nms_labels,
            title=title,
            box_color="b",
            is_gt=False,
        )

        plt.axis("off")
    fig.tight_layout()
    return fig


def conv_dim_formula(in_dim, kernels, paddings, strides, dilations=None):
    """
    Calculate final shape of in_dim after successive conv1d operations.

    Args:
        in_dim (int): input dimension.
        kernels (Sequence[int]): indexable seq of kernels.
        paddings (Sequence[int]): indexable seq of padding on both sides.
        strides (Sequence[int]): indexable seq of strides.
        dilations (Sequence[int]): indexable seq of dilations.
            If dilations is None, all dilations assumed to be 1.
    """
    assert len(kernels) == len(paddings) == len(strides)
    if dilations is not None:
        assert len(kernels) == len(dilations)

    out = in_dim
    for i in range(len(kernels)):
        offset = 2 * paddings[i] - kernels[i]
        if dilations is not None:
            offset -= (kernels[i] - 1) * (dilations[i] - 1)
        out = (out + offset) // strides[i] + 1
    return out


class YoloV2Loss(nn.Module):
    def __init__(
        self, grid_dim, num_bboxes, num_classes, lam_noobj=0.5, lam_coord=5
    ):
        super().__init__()
        self.grid_dim = grid_dim
        self.num_bboxes = num_bboxes
        self.num_classes = num_classes
        self.lam_noobj = lam_noobj
        self.lam_coord = lam_coord
        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, pred, target, get_avg_iou=False, iou_box_selection=True):
        """
        iou_box_selection: if True, will select the greedy box in each cell
            w.r.t. target box iou. Otherwise selection needs to already have been
            performed so that pred.shape == target.shape
        pred: should be of shape (batch, grid_dim, grid_dim, num_boxes * (num_box_elements + num_classes))
        while target should be of size (batch, grid_dim, grid_dim, num_box_elements + num_classes)

        both should be in midpoint format - x, y, w, h
        """
        # selects the best box based on max iou;
        if iou_box_selection:
            if get_avg_iou:
                bestbox, vals = greedy_iou_box_selection(
                    pred, target, get_avg_iou
                )
            else:
                bestbox = greedy_iou_box_selection(pred, target, get_avg_iou)
        # otherwise select based on object confidence of boxes
        else:
            bestbox = pred
            assert bestbox.shape == target.shape
            if get_avg_iou:
                # is batch, grid_dim, grid_dim shape;
                vals = eval_utils.get_IoU(bestbox, target, midpoint=True)
        # mask for which grid_cells to count in loss;
        exists_box = target[..., 0].unsqueeze(-1)
        num_existing_boxes = exists_box.sum()

        # coord loss;
        # divide by crop dim;
        box_predictions = exists_box * bestbox[..., 1:5] / math.sqrt(224)
        box_targets = exists_box * target[..., 1:5] / math.sqrt(224)

        # Take sqrt of width and height of boxes
        # for more box size invariance;
        # box_predictions[..., 2:4] = torch.sqrt(box_predictions[..., 2:4])
        # box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        # get loss;
        box_loss = (
            self.mse(
                box_predictions,
                box_targets,
            )
            / num_existing_boxes
        )

        # object detection loss;
        object_loss = (
            self.mse(
                exists_box * bestbox[..., 0:1],
                exists_box * target[..., 0:1],
            )
            / num_existing_boxes
        )

        # no object loss;
        no_object_loss = self.mse(
            (1 - exists_box) * bestbox[..., 0:1],
            (1 - exists_box) * target[..., 0:1],
        ) / (exists_box.numel() - num_existing_boxes)

        # class loss;
        # I cringe when I see mse, so I used CELoss;
        class_loss = (
            nn.functional.cross_entropy(
                (exists_box * bestbox[..., -self.num_classes :]).view(-1),
                (exists_box * target[..., -self.num_classes :]).view(-1),
                reduction="sum",
            )
            / num_existing_boxes
        )

        # overall loss;
        loss = (
            self.lam_coord * box_loss  # first two rows in paper
            + object_loss  # third row in paper
            + self.lam_noobj * no_object_loss  # forth row
            + class_loss  # fifth row
        )
        print(
            f"{box_loss.item()=}, {object_loss.item()=}, {no_object_loss.item()=}, {class_loss.item()=}"
        )
        if get_avg_iou:
            avg_iou = (
                vals.unsqueeze(-1) * exists_box
            ).sum() / exists_box.sum()
            return loss, avg_iou
        return loss


def greedy_iou_box_selection(pred, target, get_avg_iou=False):
    """
    pred: of shape (B, grid_dim, grid_dim, num_boxes * (5 + num_classes))
    """
    batch_size, grid_dim, _, out_channels = target.shape
    num_boxes = pred.shape[-1] // out_channels
    assert num_boxes * out_channels == pred.shape[-1]
    # Calculate IoU for the predicted bounding boxes with target bbox
    # using broadcasting to make the targets of shape [..., 1, 4]
    # while the predictions have shape [..., num_boxes, 4];
    ious = eval_utils.get_IoU(
        target[..., 1:5].unsqueeze(-2),  # [..., 1, 4] shape;
        pred.view(batch_size, grid_dim, grid_dim, num_boxes, -1)[
            ..., 1:5
        ],  # [..., num_boxes, 4] shape
        midpoint=True,
    ).detach()
    # ious should be (batch_size, grid_dim, grid_dim, num_bboxes)
    # selects the best box based on max iou;
    if not get_avg_iou:
        bestbox = ious.argmax(-1)
    else:
        vals, bestbox = ious.max(-1)
    bestbox = pred.view(-1, num_boxes, out_channels)[
        [torch.arange(batch_size * grid_dim * grid_dim), bestbox.view(-1)]
    ].view(batch_size, grid_dim, grid_dim, -1)
    if get_avg_iou:
        return bestbox, vals
    return bestbox


def greedy_confidence_box_selection(label_matrices: torch.Tensor, num_boxes):
    """
    Selects one bbox per grid cell based on max confidence.

    label_matrices: is of 3 or 4 dims with the last 3 dims being
        (grid_dim, grid_dim, out_channels);
    """
    if label_matrices.ndim < 4:
        label_matrices = label_matrices.unsqueeze(0)
    assert label_matrices.ndim == 4
    batch_size, grid_dim, _, out_channels = label_matrices.shape
    box_dim = out_channels // num_boxes
    assert num_boxes * box_dim == out_channels
    label_matrices = label_matrices.view(-1, num_boxes, box_dim)
    idxs = label_matrices[..., 0].argmax(-1).detach()
    label_matrices = label_matrices[
        [torch.arange(batch_size * grid_dim * grid_dim), idxs.view(-1)]
    ].view(batch_size, grid_dim, grid_dim, -1)
    assert label_matrices.shape[-1] == box_dim
    return label_matrices
