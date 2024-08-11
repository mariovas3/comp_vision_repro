from pathlib import Path

ROOT_DIR = Path(__file__).absolute().parents[2]
SAVED_MODELS_PATH = ROOT_DIR / "saved_models/yolo"
DATA_DIR = ROOT_DIR / "data"
ANCHOR_DIMS_PATH = DATA_DIR / "anchor_dims.json"
DEFAULT_STANDARD_IMG_DIM = 224
DEFAULT_RESIZE_SIZE = DEFAULT_STANDARD_IMG_DIM + 8
NUM_BBOXES = 5
DEFAULT_GRID_DIM = 7
LABEL_TO_IDX = {
    "aeroplane": 0,
    "bicycle": 1,
    "bird": 2,
    "boat": 3,
    "bottle": 4,
    "bus": 5,
    "car": 6,
    "cat": 7,
    "chair": 8,
    "cow": 9,
    "diningtable": 10,
    "dog": 11,
    "horse": 12,
    "motorbike": 13,
    "person": 14,
    "pottedplant": 15,
    "sheep": 16,
    "sofa": 17,
    "train": 18,
    "tvmonitor": 19,
}

IDX_TO_LABEL = sorted(LABEL_TO_IDX.keys())
# resnet50 pretrained details:
# Accepts PIL.Image,
# batched (B, C, H, W) and single (C, H, W) image torch.Tensor objects.
# The images are resized to resize_size=[232] using
# interpolation=InterpolationMode.BILINEAR,
# followed by a central crop of crop_size=[224].
# Finally the values are first rescaled to [0.0, 1.0] and
# then normalized using mean=[0.485, 0.456, 0.406]
# and std=[0.229, 0.224, 0.225].
