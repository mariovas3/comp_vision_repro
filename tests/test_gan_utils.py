import torch
from torch import nn

from gan.utils import get_height_width_convtranspose


def test_conv_traspose_dim_formula():
    kernels, paddings, strides = [3, 3], [1, 1], [2, 1]
    convt = nn.Sequential(
        nn.ConvTranspose2d(
            in_channels=1,
            out_channels=3,
            kernel_size=kernels[0],
            stride=strides[0],
            padding=paddings[0],
        ),
        nn.ConvTranspose2d(
            in_channels=3,
            out_channels=5,
            kernel_size=kernels[1],
            stride=strides[1],
            padding=paddings[1],
        ),
    )
    x = torch.randn((1, 28, 28))
    out = convt(x)
    Hout, Wout = get_height_width_convtranspose(
        Hin=28,
        Win=28,
        kernels=kernels,
        paddings=paddings,
        strides=strides,
        dilations=None,
        output_paddings=None,
    )
    assert out.shape[-2:] == (Hout, Wout)


def test_conv_traspose_dim_formula_with_dil():
    kernels, paddings, strides = [3, 3], [1, 1], [2, 1]
    dilations = [1, 2]
    convt = nn.Sequential(
        nn.ConvTranspose2d(
            in_channels=1,
            out_channels=3,
            kernel_size=kernels[0],
            stride=strides[0],
            padding=paddings[0],
            dilation=dilations[0],
        ),
        nn.ConvTranspose2d(
            in_channels=3,
            out_channels=5,
            kernel_size=kernels[1],
            stride=strides[1],
            padding=paddings[1],
            dilation=dilations[1],
        ),
    )
    x = torch.randn((1, 28, 28))
    out = convt(x)
    Hout, Wout = get_height_width_convtranspose(
        Hin=28,
        Win=28,
        kernels=kernels,
        paddings=paddings,
        strides=strides,
        dilations=dilations,
        output_paddings=None,
    )
    assert out.shape[-2:] == (Hout, Wout)
