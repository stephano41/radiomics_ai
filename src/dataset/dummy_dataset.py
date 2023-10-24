from skorch.dataset import Dataset
import torch


class DummyDataset(Dataset):
    def __init__(self, X, y=None, length=None, image_shape=(16,16,16)):
        self.image_shape = image_shape
        super().__init__(X, y, length)

    def __getitem__(self, item):
        return torch.randn([1, *self.image_shape]), 0
