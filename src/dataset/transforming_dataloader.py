from copy import copy, deepcopy

from torch.utils.data import DataLoader, Subset
from torchio import SubjectsDataset, Compose, Queue


class TransformingDataLoader(DataLoader):
    def __init__(self, dataset, augment_transforms, *args, **kwargs):
        self.augment_transforms = augment_transforms

        _dataset = copy(dataset)

        if isinstance(dataset, Subset):
            _dataset.dataset = copy(_dataset.dataset)
            _dataset.dataset = self._add_augmentation(_dataset.dataset)
        else:
            _dataset = self._add_augmentation(_dataset)

        super().__init__(_dataset, *args, **kwargs)


    def _add_augmentation(self, dataset):
        if isinstance(dataset, SubjectsDataset):
            existing_transforms = dataset._transform
            _dataset = deepcopy(dataset)
            if existing_transforms is not None:
                _dataset.set_transform(Compose([existing_transforms, self.augment_transforms]))
            else:
                _dataset.set_transform(self.augment_transforms)
            return _dataset

        if isinstance(dataset, Queue):
            existing_transforms = dataset.subjects_dataset._transform
            _dataset = deepcopy(dataset.subjects_dataset)
            if existing_transforms is not None:
                _dataset.set_transform(Compose([existing_transforms, self.augment_transforms]))
            else:
                _dataset.set_transform(self.augment_transforms)
            dataset.subjects_dataset = _dataset
        
        return dataset