import torch
import torchvision.transforms as T
from torch import nn


class CombinedModel(nn.Module):
    def __init__(
        self,
        resnet,
        num_boxes,
        num_bbox_elements,
        num_classes,
        img_transforms=None,
    ):
        super().__init__()
        self.resnet = resnet
        self.out_channels = num_boxes * num_bbox_elements + num_classes
        self.conv_head = nn.Conv2d(2048, self.out_channels, kernel_size=1)
        self.num_boxes = num_boxes
        self.num_bbox_elements = num_bbox_elements
        self.num_classes = num_classes
        self.img_transforms = img_transforms

    def forward(self, x):
        # by default, transform input image to have H = W = 224;
        if self.img_transforms is not None:
            x = self.img_transforms(x)
        # feats should be of size (batch, 2048, H / 32, W / 32)
        # we treat the grid is 7x7 so that each cell
        # of feats corresponds to the feats from the
        # relevant cell of the grid over the input image.
        feats = get_resnet_feats(self.resnet, x)
        # conv_head has kernel = (1, 1) and just remaps
        # the 2048 in channels to however many out channels
        # are needed - giving (B, out_channels, 7, 7)
        # out_channels is
        # num_bboxes * (has_object_entry + bbox_coords) + num_classes
        # permute dims to get output of (B, grid_dim, grid_dim, out_channels)
        return self.conv_head(feats).permute(0, -2, -1, 1)

    def get_yolo9000_output(self, x):
        out = self(x)
        # get confidence prob;
        out[..., : -self.num_classes : self.num_bbox_elements] = torch.sigmoid(
            out[..., : -self.num_classes : self.num_bbox_elements]
        )
        # get x and y offsets from top left corner of grid cell;
        # if grid cell has idx (i, j) in the grid, the anchor box
        # center coords are bx, by = i + sigmoid(x), j + sigmoid(y)
        # then to remap to image pixels you do
        # img_x, img_y = bx * img_height / S, by * img_width / S
        out[
            ..., 1 : -self.num_classes : self.num_bbox_elements
        ] = torch.sigmoid(
            out[..., 1 : -self.num_classes : self.num_bbox_elements]
        )
        out[
            ..., 2 : -self.num_classes : self.num_bbox_elements
        ] = torch.sigmoid(
            out[..., 2 : -self.num_classes : self.num_bbox_elements]
        )
        # exponentiate width and height entries;
        # the bounding box width and height are then
        # bw, bh = pw * exp(w_entry), ph * exp(h_entry)
        # where pw and ph are the prior width and height of the
        # anchor box.
        out[..., 3 : -self.num_classes : self.num_bbox_elements] = torch.exp(
            out[..., 3 : -self.num_classes : self.num_bbox_elements]
        )
        out[..., 4 : -self.num_classes : self.num_bbox_elements] = torch.exp(
            out[..., 4 : -self.num_classes : self.num_bbox_elements]
        )
        return out


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
