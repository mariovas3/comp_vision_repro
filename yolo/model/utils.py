import torch
import torchvision.transforms as T
from torch import nn

from yolo.model import eval_utils


class CombinedModel(nn.Module):
    def __init__(
        self,
        resnet,
        num_bboxes,
        num_bbox_elements,
        num_classes,
    ):
        super().__init__()
        self.resnet = resnet
        # change from yolov1 output, now predict classes
        # in each bounding box;
        self.out_channels = num_bboxes * (num_bbox_elements + num_classes)
        self.conv_head = nn.Conv2d(2048, self.out_channels, kernel_size=1)
        self.num_boxes = num_bboxes
        self.num_bbox_elements = num_bbox_elements
        self.num_classes = num_classes

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

    def get_yolo9000_output(self, x):
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
        out = out.view(B, S, S, self.num_boxes, -1)
        out[..., self.num_bbox_elements :] = torch.softmax(
            out[..., self.num_bbox_elements :], -1
        )
        return out.view(B, S, S, -1)


def output_to_bounding_boxes_xywh_(
    yolo_output: torch.Tensor,
    grid_dim: int,
    num_boxes: int,
    anchor_boxes_wh: torch.Tensor,
):
    """
    Inplace modify yolo output to be
    prob_obj, bx, by, bw, bh, softmax_over_classes as per yolov2 paper.

    yolo_output: tensor of size (batch, grid_dim, grid_dim, out_len)
        where out_len is num_bboxes * (has_object + (x, y, w, h) + num_classes)
    anchor_boxes: tensor of size (2, num_anchor_boxes).
    """
    assert num_boxes == anchor_boxes_wh.shape[-1]
    # grid_dim, grid_dim, 1
    width_grid_coords = (
        torch.arange(grid_dim).expand(grid_dim, -1).unsqueeze(-1)
    )
    # the 5 corresponds to (has_object, x, y, w, h)
    num_classes = yolo_output.shape[-1] // num_boxes - 5
    offset = 5 + num_classes
    # get predicted center of bounding boxes;
    # x and y are in (0, grid_dim);
    yolo_output[..., 1::offset] = (
        yolo_output[..., 1::offset] + width_grid_coords
    )
    yolo_output[..., 2::offset] = yolo_output[
        ..., 2::offset
    ] + width_grid_coords.permute(1, 0, 2)
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
    def __init__(self, grid_dim, num_bboxes, num_classes):
        super().__init__()
        self.grid_dim = grid_dim
        self.num_bboxes = num_bboxes
        self.num_classes = num_classes
        self.lambda_noobj = 0.5
        self.lambda_coord = 5
        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, pred, target, get_avg_iou=False):
        """
        pred should be of shape (batch, grid_dim, grid_dim, num_boxes * (num_box_elements + num_classes))
        while target should be of size (batch, grid_dim, grid_dim, num_box_elements + num_classes)

        both should be in midpoint format - x, y, w, h
        """
        batch_size, grid_dim, _, _ = target.shape
        # Calculate IoU for the predicted bounding boxes with target bbox
        ious = eval_utils.get_IoU(
            target[..., 1:5].unsqueeze(-2),
            pred.view(batch_size, grid_dim, grid_dim, self.num_bboxes, -1)[
                ..., 1:5
            ],
            midpoint=True,
        )
        # ious should be (batch_size, grid_dim, grid_dim, num_bboxes)
        dim_size = self.num_classes + 5

        # selects the best box based on max iou;
        if not get_avg_iou:
            bestbox = ious.argmax(-1)
        else:
            vals, bestbox = ious.max(-1)
        bestbox = pred.view(-1, self.num_bboxes, dim_size)[
            [torch.arange(batch_size * grid_dim * grid_dim), bestbox.view(-1)]
        ].view(batch_size, grid_dim, grid_dim, -1)

        # mask for which grid_cells to count in loss;
        exists_box = target[..., 0].unsqueeze(-1)

        # coord loss;
        # Set cells with no object in them to 0
        box_predictions = exists_box * bestbox[..., 1:5]

        box_targets = exists_box * target[..., 1:5]

        # Take sqrt of width and height of boxes
        # for more box size invariance;
        box_predictions[..., 2:4] = torch.sqrt(box_predictions[..., 2:4])
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            box_predictions.view(-1),
            box_targets.view(-1),
        )

        # object detection loss;
        # pred_box is the confidence score for the bbox with highest IoU
        pred_box = bestbox[..., 0]

        object_loss = self.mse(
            (exists_box * pred_box).view(-1),
            (exists_box * target[..., 0]).view(-1),
        )

        # no object loss;
        no_object_loss = self.mse(
            ((1 - exists_box) * bestbox[..., 0]).view(-1),
            ((1 - exists_box) * target[..., 0]).view(-1),
        )

        # class loss;
        class_loss = self.mse(
            (exists_box * bestbox[..., -self.num_classes :]).view(-1),
            (exists_box * target[..., -self.num_classes :]).view(-1),
        )

        # overall loss;
        loss = (
            self.lambda_coord * box_loss  # first two rows in paper
            + object_loss  # third row in paper
            + self.lambda_noobj * no_object_loss  # forth row
            + class_loss  # fifth row
        )
        if get_avg_iou:
            avg_iou = (vals * exists_box).sum() / exists_box.sum()
            return loss, avg_iou
        return loss
