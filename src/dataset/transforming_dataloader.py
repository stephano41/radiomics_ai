from copy import copy

from torch.utils.data import DataLoader
from torchio import SubjectsDataset, Compose


class TransformingDataLoader(DataLoader):
    def __init__(self, dataset, augment_transforms, *args, **kwargs):
        _dataset = copy(dataset)

        if isinstance(_dataset, SubjectsDataset):
            existing_transforms = _dataset._transform
            if existing_transforms is not None:
                _dataset.set_transform(Compose([existing_transforms, augment_transforms]))

        super().__init__(_dataset, *args, **kwargs)
