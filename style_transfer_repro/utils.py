import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.io import read_image
from PIL.Image import Image
import os
np.random.seed(42)
torch.manual_seed(42)


# standardise RGB images using the channel-wise quantites below;
MAGIC_NUMBERS = {"mean": torch.tensor([0.485, 0.456, 0.406]),
        "std": torch.tensor([0.229, 0.224, 0.225])}  # as per https://github.com/pytorch/vision/issues/1439
# to my understanding the MAGIC_NUMBERS are calculated on a subset of ImageNet;
# the exact subset seems to be lost;
# Nevertheless the docs (https://pytorch.org/vision/0.11/models.html) suggest that
# the pretrained models expect the input images to be standardised using the above quantities;


CUR_DIR = os.path.dirname(__file__)
CACHE_DIR = os.path.join(CUR_DIR, "data/")
# set cache directory for saving pretrained model;
torch.hub.set_dir(CACHE_DIR)


def load_image(path: str, transform: nn.Module=None, apply_transform: bool=False, shape: tuple=None) -> torch.Tensor:
    img = read_image(path)  # gives torch.uint8 type tensor;
    img_dims = img.shape  # dimensions for tensor are in (C, H, W) format;
    if apply_transform:
        if transform is None:
            shape = (720, 1080) if shape is None else shape
            transform = nn.Sequential(
                        T.Resize(shape),
                        T.ConvertImageDtype(torch.float32),  # pretrained models take tensor images with values in [0, 1];
                        T.Normalize(**MAGIC_NUMBERS)  # then standardise to be consistent with ImageNet input;
                    )
        return transform(img)
    return img


def tensor_to_pil(img: torch.Tensor, back_transform: nn.Module=None, apply_transform: bool=False) -> Image:
    img = img.to("cpu").detach().squeeze().clone()  # don't touch the actual image as it may be still training;
    if apply_transform:
        if back_transform is None:
            back_transform = nn.Sequential(
                        T.Normalize(- MAGIC_NUMBERS["mean"] / MAGIC_NUMBERS["std"], 1 / MAGIC_NUMBERS["std"]),
                        T.ConvertImageDtype(torch.uint8)
                    )
        return T.ToPILImage()(back_transform(img))
    return T.ToPILImage()(img)

