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

from .sitk_flip import flip_image3D
from .sitk_rotate import sitk_transform3D
from ..utils.prepro_utils import get_multi_paths_with_separate_folder_per_case


class SitkImageTransformer:
    def __init__(self, transform_kwargs=None):
        if transform_kwargs is None:
            self.transform_kwargs = dict(aug_transform=sitk.Similarity3DTransform(),
                                         thetaX=(0, 0),
                                         thetaY=(0, 0),
                                         thetaZ=(-np.pi / 2, np.pi / 2),
                                         tx=(0, 0),
                                         ty=(0, 0),
                                         tz=(0, 0),
                                         scale=(1, 1),
                                         n=2)

    def transform(self, sitk_images: pd.DataFrame, y=None):
        generated_images = []
        for row in sitk_images.iterrows():
            sitk_list = row[1].values
            generated_images.append(sitk_list)
            generated_images.append([flip_image3D(img) for img in sitk_list])
            augmented_images = [sitk_transform3D(img, **self.transform_kwargs) for img in sitk_list]

            generated_images.extend([list(i) for i in zip(*augmented_images)])
        return pd.DataFrame(data=generated_images, columns=sitk_images.columns)


class SitkImageProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, result_dir, data_dir, image_stems: Tuple[str,...]=('image'), mask_stem='mask', n_jobs=2, target_size=(16, 16, 16), **settings):
        self.n_jobs = n_jobs
        self.target_size = target_size
        self.data_dir = data_dir
        self.mask_stem = mask_stem
        self.result_dir = result_dir
        self.image_stems = image_stems
        # self.set_output = set_output

        self.paths_df = get_multi_paths_with_separate_folder_per_case(data_dir=data_dir,
                                                                      image_stems=image_stems,
                                                                      mask_stem=mask_stem,
                                                                      relative=False)

        self._settings = settings.copy()

        self.label = settings.get('label', 1)
        self.label_channel = settings.get('label_channel', 0)
        self.interpolator = settings.get('interpolator', sitk.sitkBSpline)
        self.padDistance = settings.get('padDistance', 5)

        self.saved_df = None

        if not self.get_cache_path.exists():
            self.saved_df = self.extract_data(self.paths_df)

    @property
    def get_cache_path(self):
        return Path(self.result_dir) / 'processed_sitk.pkl'

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Expects X as list of id's to get or extract
        assert self.get_cache_path.exists(), f"{self.__class__} hasn't been initialised yet"

        with open(self.get_cache_path, 'rb') as f:
            self.saved_df = pickle.load(f)

        X_df = pd.DataFrame({'ID': X})
        filtered_df = X_df.merge(self.saved_df, on='ID', how='left')

        return filtered_df.loc[:, filtered_df.columns!='ID']

    def extract_data(self, df):
        print(f"extracting at {self.get_cache_path} cached path")
        image_column_names = [f"image_{name}" for name in self.image_stems]

        image_paths = df.loc[:, df.columns.isin(image_column_names)]
        mask_paths = df.loc[:, df.columns == 'segmentation_path']
        ID_list = df.loc[:, df.columns == 'ID']

        X = pqdm(
            ({"image_paths": val[0][1].values,
              "mask_path": val[1][1].values[0],
              'id': val[2][1].values[0]}
             for val in zip(image_paths.iterrows(), mask_paths.iterrows(), ID_list.iterrows())),
            self.read_crop_resample,
            n_jobs=self.n_jobs,
            argument_type='kwargs'
        )

        # column_names = pd.Series(df.columns).loc[df.columns.str.startswith(self.image_column_prefix)].values
        # column_names = np.insert(column_names, 0, 'ID')

        x_df = pd.DataFrame(data=X, columns=["ID"] + image_column_names)

        with open(os.path.join(self.get_cache_path), 'wb') as f:
            pickle.dump(x_df, f)

        return x_df

    def read_crop_resample(self, image_paths, mask_path, id):
        resampled_images = [id]
        try:
            for image_path in image_paths:
                image, mask = self.load_nifti(image_path, mask_path)

                cropped_image, cropped_mask = self.crop_to_mask(image, mask)

                resampled_image, _ = self.resample_image_mask(cropped_image, cropped_mask, self.target_size)

                resampled_images.append(resampled_image)
        except Exception as e:
            print(e)
            raise e
            pass

        return resampled_images

    def crop_to_mask(self, image, mask):
        boundingBox, correctedMask = imageoperations.checkMask(image, mask, **self._settings)
        if correctedMask is not None:
            mask = correctedMask

        if boundingBox is None:
            raise ValueError('Mask checks failed during pre-crop')
        cropped_image, cropped_mask = imageoperations.cropToTumorMask(image, mask, boundingBox)
        return cropped_image, cropped_mask

    def load_nifti(self, image_filename, mask_filename):
        image = sitk.ReadImage(image_filename)
        mask = imageoperations.getMask(sitk.ReadImage(mask_filename), label=self.label,
                                       label_channel=self.label_channel)
        return image, mask

    def resample_image_mask(self, image, mask, new_size):
        new_spacing = [curr * orig / new for curr, orig, new in
                       zip(image.GetSpacing(), image.GetSize(), new_size)]

        bb = _checkROI(image, mask, label=self.label)

        direction = np.array(mask.GetDirection())

        maskSpacing = np.array(mask.GetSpacing())
        Nd_mask = len(maskSpacing)

        spacingRatio = maskSpacing / new_spacing

        # Determine bounds of cropped volume in terms of new Index coordinate space,
        # round down for lowerbound and up for upperbound to ensure entire segmentation is captured (prevent data loss)
        # Pad with an extra .5 to prevent data loss in case of upsampling. For Ubound this is (-1 + 0.5 = -0.5)
        bbNewLBound = np.floor((bb[:Nd_mask] - 0.5) * spacingRatio - self.padDistance)

        # Ensure resampling is not performed outside bounds of original image
        bbNewLBound = np.where(bbNewLBound < 0, 0, bbNewLBound)

        bbOriginalLBound = bbNewLBound / spacingRatio
        newOriginIndex = np.array(.5 * (new_spacing - maskSpacing) / maskSpacing)
        newCroppedOriginIndex = newOriginIndex + bbOriginalLBound
        newOrigin = mask.TransformContinuousIndexToPhysicalPoint(newCroppedOriginIndex)

        imagePixelType = image.GetPixelID()
        maskPixelType = mask.GetPixelID()

        rif = sitk.ResampleImageFilter()

        rif.SetOutputSpacing(new_spacing)
        rif.SetOutputDirection(direction)
        rif.SetSize(new_size)
        rif.SetOutputOrigin(newOrigin)

        rif.SetOutputPixelType(imagePixelType)
        rif.SetInterpolator(self.interpolator)
        resampledImageNode = rif.Execute(image)

        rif.SetOutputPixelType(maskPixelType)
        rif.SetInterpolator(sitk.sitkNearestNeighbor)
        resampledMaskNode = rif.Execute(mask)

        return resampledImageNode, resampledMaskNode

    def get_feature_names_out(self):
        # see https://stackoverflow.com/questions/75026592/how-to-create-pandas-output-for-custom-transformers
        pass
