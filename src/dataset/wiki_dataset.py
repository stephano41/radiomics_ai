import os
from functools import partial

import SimpleITK
import numpy as np
import pandas as pd

import six
from dcmrtstruct2nii import dcmrtstruct2nii
from dcmrtstruct2nii.exceptions import InvalidFileFormatException
from radiomics import featureextractor
from tqdm import tqdm
from pathlib import Path
from multiprocessing.pool import Pool


def _match1row(df, column_name, value):
    return df.loc[df[column_name] == value]

# TODO take out extraction class out of dataset
class WikiSarcoma:
    """
    takes in data_path to nii files of wiki sarcoma from https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=21266533#21266533715ed75afd8744ec84c7d0b7daa64456
    does feature extraction with pyradiomics, generating x
    maps data with clinical info csv for histological grades
    """

    def __init__(self, data_path, clinical_data_path, radiomic_params, extract_label=255, num_cpu=2):
        self.clinical_csv = pd.read_csv(clinical_data_path, encoding='cp1252')
        if not isinstance(data_path, Path):
            self.data_path = Path(data_path)
        else:
            self.data_path = data_path

        partial_data_extractor = partial(_one_extraction, clinical_csv=self.clinical_csv,
                                         extract_label=extract_label)
        with Pool(num_cpu) as pool:
            self._x = []
            self._y = []
            for feature, target in tqdm(pool.imap(partial_data_extractor, list(self.data_path.glob('*/')))):
                self._x.append(feature)
                self._y.append(target)

        self._x = np.array(self._x)
        self._y = np.array(self._y)

        # convert y values from strings to 0, 1, 2
        self._y = np.searchsorted(np.unique(self._y), self._y).astype(int)

    @property
    def x(self) -> np.ndarray:
        """
        x shape is sample size x feature size
        :return:
        """
        return self._x

    @property
    def y(self) -> np.ndarray:
        """
        y shape is sample size
        :return:
        """
        return self._y


def _one_extraction(pt_path, clinical_csv, extract_label):
    pt_id = pt_path.name
    target = _match1row(clinical_csv, "Patient ID", pt_id)["Grade"].iloc[0]
    # _y = np.append(_y, target)

    image = SimpleITK.ReadImage(pt_path / 'image.nii.gz')
    mask = SimpleITK.ReadImage(pt_path / 'mask_GTV_Mass.nii.gz')
    result = extractor.execute(image, mask, extract_label)

    feature = np.array([])

    for key, value in six.iteritems(result):
        if key.startswith("original_"):
            feature = np.append(feature, result[key])
    # self._x.append(feature)
    return feature, target


def convert_wiki2nii(csv_path, data_path, output_path, modality="MR"):
    csv = pd.read_csv(csv_path)

    pt_ids = np.unique(csv["Subject ID"])

    for pt_id in tqdm(pt_ids):
        pt_rows = csv.loc[csv["Subject ID"] == pt_id]
        struct_file = pt_rows.loc[pt_rows["Modality"] == "RTSTRUCT"]["File Location"].iloc[0]
        struct_file = os.path.join(data_path, struct_file)
        struct_file = os.path.join(struct_file, os.listdir(struct_file)[0])

        image_file = pt_rows.loc[pt_rows["Modality"] == modality]["File Location"].iloc[0]
        image_file = os.path.join(data_path, image_file)

        pt_path = os.path.join(output_path, pt_id)
        if not os.path.exists(pt_path):
            os.mkdir(pt_path)
        try:
            dcmrtstruct2nii(struct_file, image_file, pt_path)
        except InvalidFileFormatException as e:
            print(e)
            os.rmdir(pt_path)
            pass
