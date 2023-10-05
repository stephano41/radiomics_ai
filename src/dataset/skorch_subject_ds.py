import pandas as pd
import torch
from torchio import SubjectsDataset


class SkorchSubjectsDataset(SubjectsDataset):
    def __init__(self, subjects, y=None, transform=None, load_getitem=True):
        if isinstance(subjects, pd.DataFrame):
            subjects = subjects.values.flatten().tolist()
        super().__init__(subjects, transform, load_getitem)

    def __getitem__(self, item):
        return torch.concatenate([i.data for i in super().__getitem__(item).get_images()]), 0
