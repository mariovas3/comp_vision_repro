from pathlib import Path

ROOT_DIR = Path(__file__).absolute().parents[2]
SAVED_MODELS_PATH = ROOT_DIR / "saved_models/yolo"
DATA_DIR = ROOT_DIR / "data"
ANCHOR_DIMS_PATH = DATA_DIR / "anchor_dims.json"
STANDARDISED_IMG_DIM = 224

# resnet50 pretrained details:
# Accepts PIL.Image,
# batched (B, C, H, W) and single (C, H, W) image torch.Tensor objects.
# The images are resized to resize_size=[232] using
# interpolation=InterpolationMode.BILINEAR,
# followed by a central crop of crop_size=[224].
# Finally the values are first rescaled to [0.0, 1.0] and
# then normalized using mean=[0.485, 0.456, 0.406]
# and std=[0.229, 0.224, 0.225].
