from torchio import SubjectsDataset
import torch

class SkorchSubjectsDataset(SubjectsDataset):
    def __init__(self, subjects, y=None, transform=None, load_getitem=True):
        super().__init__(subjects, transform, load_getitem)

    def __getitem__(self, item):
        return torch.concatenate([i.data for i in super().__getitem__(item).get_images()]), 0