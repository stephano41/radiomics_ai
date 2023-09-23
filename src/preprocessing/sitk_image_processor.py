from typing import Tuple

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import torchio as tio

from src.utils.prepro_utils import get_multi_paths_with_separate_folder_per_case


class SitkImageProcessor(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer for processing medical image data stored as SimpleITK images.

    This class is designed to work with medical image data organized in a directory structure where each subject has its
    own folder containing image and mask files. It provides methods for loading, processing, and transforming image data.

    Parameters:
        data_dir (str): The root directory containing the subject folders with image and mask data.
        image_stems (tuple of str, optional): A tuple of image stem names to identify different image modalities.
            Default is ('image').
        mask_stem (str, optional): The stem name to identify mask files. Default is 'mask'.

    Attributes:
        data_dir (str): The root directory containing the subject folders.
        mask_stem (str): The stem name for mask files.
        image_stems (tuple of str): Tuple of image stem names.
        paths_df (DataFrame): A DataFrame containing paths to image and mask files for each subject.
        subject_list (list of Subject): A list of Subject objects, each representing a subject with associated image
            and mask data.

    Methods:
        fit(X, y=None): Fit method for scikit-learn compatibility. This transformer is stateless and does not require fitting.
        transform(X, y=None): Transforms a list of subject IDs into a DataFrame containing image and mask data.

    Properties:
        image_column_names: Returns a list of column names for the image data in the transformed DataFrame.

    Example:
        processor = SitkImageProcessor(data_dir='/path/to/data')
        subjects_data = processor.transform(['subject1', 'subject2'])"""
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

        for _, row in self.paths_df.iterrows():
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
