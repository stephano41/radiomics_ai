from typing import Tuple

import pandas as pd
import torch
from torchio import SubjectsDataset
from skorch.dataset import Dataset
import torchio as tio
import numpy as np
from src.utils.dataset import get_multi_paths_with_separate_folder_per_case


#
# class SkorchSubjectsDataset(SubjectsDataset):
#     def __init__(self, subjects, y=None, transform=None, load_getitem=True):
#         if isinstance(subjects, pd.DataFrame):
#             subjects = subjects.values.flatten().tolist()
#         super().__init__(subjects, transform, load_getitem)
#
#     def __getitem__(self, item):
#         return torch.concatenate([i.data for i in super().__getitem__(item).get_images()]), 0


class SkorchSubjectsDataset(SubjectsDataset):
    def __init__(self, X, y, data_dir, image_stems: Tuple[str, ...]=('image'), mask_stem='mask', transform=None,
                 load_getitem=True):
        """
        mask name is 'mask'
        each subject as ID attribute as well
        :param X:
        :param y:
        :param data_dir:
        :param image_stems:
        :param mask_stem:
        :param transform:
        :param load_getitem:
        """
        self.data_dir = data_dir
        self.mask_stem = mask_stem
        self.image_stems = image_stems
        self.paths_df = get_multi_paths_with_separate_folder_per_case(data_dir=data_dir,
                                                                      image_stems=image_stems,
                                                                      mask_stem=mask_stem,
                                                                      relative=False)
        self.X = X
        
        subject_list= self.get_subjects_list(X)
        
        if y is None:
            y=[torch.Tensor([0])]*len(X)
        else:
            identity_matrix = np.eye(len(np.unique(y)))
            y = identity_matrix[y]
        
        self.y= y

        # self.identity_matrix = np.eye(len(max(np.unique(y), 2)))
        self.id_map = dict(zip(X, y))
        super().__init__(subject_list, transform, load_getitem=load_getitem)

    def __getitem__(self, item):
        subject = super().__getitem__(item)
        out_X = torch.concatenate([i.data for i in subject.get_images()]).type(torch.FloatTensor)
        # print(out_X.dtype)
        return out_X, torch.Tensor(self.id_map[subject.ID]).float()

    def get_subjects_list(self, X):
        subjects = []

        for _, row in self.paths_df.iterrows():
            images = row.loc[row.index.isin(self.image_column_names)].to_dict()

            subjects.append(tio.Subject(
                ID=row.ID,
                mask=tio.LabelMap(row.segmentation_path),
                **{k: tio.ScalarImage(v) for k, v in images.items()}
            ))

        subjects_df = pd.DataFrame({'ID': [s.ID for s in subjects], 'subjects': subjects})
        X_df = pd.DataFrame({'ID': X})

        filtered_df = X_df.merge(subjects_df, on='ID', how='left')

        return filtered_df.loc[:, filtered_df.columns=='subjects'].values.flatten().tolist()

    @property
    def image_column_names(self):
        return [f"image_{name}" for name in self.image_stems]