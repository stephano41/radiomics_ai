import os
import pickle
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import SimpleITK as sitk
from radiomics import imageoperations
from radiomics.imageoperations import _checkROI
import numpy as np
from pqdm.processes import pqdm
import torchio as tio
from torchio import SubjectsDataset
from tqdm import tqdm

from ..utils.prepro_utils import get_multi_paths_with_separate_folder_per_case



class SitkImageProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, data_dir, image_stems: Tuple[str, ...] = ('image'), mask_stem='mask'):
        self.data_dir = data_dir
        self.mask_stem = mask_stem
        self.image_stems = image_stems
        self.paths_df = get_multi_paths_with_separate_folder_per_case(data_dir=data_dir,
                                                                      image_stems=image_stems,
                                                                      mask_stem=mask_stem,
                                                                      relative=False)

        self.subject_list = self.get_subjects_list()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Expects X as list of id's to get or extract
        subjects_df = pd.DataFrame({'ID': [s.ID for s in self.subject_list], 'subjects': self.subject_list})
        X_df = pd.DataFrame({'ID': X})

        filtered_df = X_df.merge(subjects_df, on='ID', how='left')

        return filtered_df.loc[:, filtered_df.columns != 'ID']

    @property
    def image_column_names(self):
        return [f"image_{name}" for name in self.image_stems]

    def get_subjects_list(self):
        subjects = []

        for _, row in tqdm(self.paths_df.iterrows()):
            images = row.loc[row.index.isin(self.image_column_names)].to_dict()

            subjects.append(tio.Subject(
                ID=row.ID,
                mask=tio.LabelMap(row.segmentation_path),
                **{k: tio.ScalarImage(v) for k, v in images.items()}
            ))

        return subjects

    def get_feature_names_out(self):
        # see https://stackoverflow.com/questions/75026592/how-to-create-pandas-output-for-custom-transformers
        pass

# https://github.com/skorch-dev/skorch/blob/master/notebooks/Transfer_Learning.ipynb
# https://github.com/skorch-dev/skorch/blob/master/examples/nuclei_image_segmentation/Nuclei_Image_Segmentation.ipynb
