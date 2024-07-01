from typing import Optional

import lightning as L
import metadata
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

# non-scriptable transform;
# scriptable transforms are combined with nn.Sequential
# and all subtransforms operate on torch.Tensor (not PIL.Image or others);
MNIST_TRANSFORM = T.Compose([T.ToTensor(), T.Lambda(lambda x: x * 2 - 1)])


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, batch_size, num_workers):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dims = (1, 28, 28)
        self.num_classes = 10

    def prepare_data(self):
        # download the data;
        MNIST(root=metadata.DATA_PATH, train=True, download=True)

    def setup(self, stage: Optional[str] = None):
        self.dataset = MNIST(
            root=metadata.DATA_PATH,
            train=True,
            download=False,
            transform=MNIST_TRANSFORM,
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
