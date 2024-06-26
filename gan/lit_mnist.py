from typing import Optional

import lightning as L
import metadata
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


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
            transform=T.ToTensor(),
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
