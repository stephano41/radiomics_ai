from __future__ import annotations

from typing import Union, List, Dict
import SimpleITK as sitk
import numpy as np
from pydicom import dcmread
import os
from pathlib import Path
import dicom2nifti
import pandas as pd
import re
import logging
from tqdm import tqdm


logger = logging.getLogger(__name__)

def get_dicomdir_meta(dicomdir_path: str, dicom_meta_params: Dict[str, str] = None)-> pd.DataFrame:
    """
    Extracts metadata from DICOMDIR file.

    Parameters:
        dicomdir_path (str): Path to DICOMDIR file.
        dicom_meta_params (Dict[str, str], optional): Additional DICOM metadata parameters to extract. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame containing DICOM metadata.
    """
    dicom_dir = dcmread(dicomdir_path)
    assert len(dicom_dir.patient_records) == 1
    assert len(dicom_dir.patient_records[0].children) == 1

    study_meta = []
    for series in dicom_dir.patient_records[0].children[0].children:
        instance_meta = {}
        for instance in series.children:

            instance_path = Path(dicom_dir.filename).parent / os.path.join(*instance.ReferencedFileID)
            series_path = instance_path.parent
            if not os.path.exists(series_path/'SeriesHeader.zip'):
                logger.warning(f"{series_path} had no SeriesHeader.zip file, assuming is a localiser so skipping")
                break
            instance_meta = get_dicom_meta(str(instance_path), dicom_meta_params)
            instance_meta['series_path'] = str(instance_path.parent)
            break
        if instance_meta:
            study_meta.append(instance_meta)
    return pd.DataFrame(study_meta)


def dicom_meta2nii(study_meta: pd.DataFrame, output_folder: str,
                   series_description_filter: Union[None, List[str]] = None,
                   reorient: bool = False, codependent: bool = False, codependent_tolerance: float = 0.1, patient_id='')-> pd.DataFrame:
    """
    Converts DICOM metadata to NIfTI format.

    Parameters:
        study_meta (pd.DataFrame): DataFrame containing DICOM metadata.
        output_folder (str): Output folder for NIfTI files.
        series_description_filter (Union[None, List[str]], optional): List of series description filters. Defaults to None.
        reorient (bool, optional): Whether to reorient NIfTI files. Defaults to False.
        codependent (bool, optional): Whether to consider codependent series. Defaults to False.
        codependent_tolerance (float, optional): Tolerance for codependent series. Defaults to 0.1.

    Returns:
        pd.DataFrame: DataFrame containing filtered DICOM metadata.
    """
    if series_description_filter is None:
        series_description_filter = ['.*']

    Path(output_folder).mkdir(exist_ok=True)

    filtered_df = []

    patient_image_position = None
    for filter in series_description_filter:
        match_rows = study_meta[study_meta['SeriesDescription'].str.contains(filter, regex=True)]
        match_rows = match_rows.sort_values(by='AcquisitionDateTime', ascending=False)
        if match_rows.empty:
            raise ValueError("No matching rows!")

        if codependent and patient_image_position is not None:
            euclidean_distances = match_rows['ImagePositionPatient'].apply(lambda x:
                                                                           np.linalg.norm(
                                                                               np.array(x.split("\\")).astype(
                                                                                   float) - patient_image_position))
            under_tolerance_row = match_rows[euclidean_distances/3.0 <= codependent_tolerance]
            if under_tolerance_row.empty:
                raise ValueError(f"No rows found under tolerance of {codependent_tolerance} for {match_rows}, and euclidean_distance averages {euclidean_distances/3.0}")
            match_row = under_tolerance_row.iloc[0]
        else:
            match_row = match_rows.iloc[0]
            # store this for the second round and if codependent is enabled
            if match_row['ImagePositionPatient']:
                # if no data in image position then can only infer from acquisition date
                patient_image_position = np.array(match_row['ImagePositionPatient'].split("\\")).astype(float)
            else:
                logger.warning(f"No ImagePositionPatient, can only use AcquisitionDateTime for {match_row}")

                patient_image_position=None

        series_path = match_row['series_path']
        series_name = re.sub(r'[^\w\s]', '', filter)
        dicom2nifti.convert_dicom.dicom_series_to_nifti(original_dicom_directory=series_path,
                                                        output_file=os.path.join(output_folder,
                                                                                 f"{series_name}_{patient_id}"),
                                                        reorient_nifti=reorient)

        filtered_df.append(match_row)

    return pd.concat(filtered_df, axis=1).T.reset_index(drop=True)


def get_dicom_meta(instance_path: str, dicom_meta_params: Dict[str, str] = None):
    """
       Extracts metadata from a DICOM instance file.

       Parameters:
           instance_path (str): Path to DICOM instance file.
           dicom_meta_params (Dict[str, str], optional): Additional DICOM metadata parameters to extract. Defaults to None.

       Returns:
           Dict[str, str]: Dictionary containing DICOM metadata.
       """
    if dicom_meta_params is None:
        dicom_meta_params = {}
    _dicom_meta_params = {
        'PatientsSex': '0010|0040',
        'PatientsBirthDate': '0010|0030',
        'PatientsAge': '0010|1010',
        'SeriesDate': '0008|0021',
        'SeriesDescription': '0008|103e',
        'BodyPartExamined': '0018|0015',
        'ManufacturersModelName': '0008|1090',
        'Manufacturer': '0008|0070',
        'PatientID': '0010|0020',
        'PatientsSize': '0010|1020',
        'PatientsWeight': '0010|1030',
        'SliceThickness': '0018|0050',
        'RepetitionTime': '0018|0080',
        'EchoTime': '0018|0081',
        'MagneticFieldStrength': '0018|0087',
        'SpacingBetweenSlices': '0018|0088',
        'DeviceSerialNumber': '0018|1000',
        'SeriesNumber': '0020|0011',
        'PixelSpacing': '0028|0030',
        'AcquisitionDateTime': '0008|002a',
        'ImagePositionPatient': '0020|0032'
    }
    _dicom_meta_params.update(dicom_meta_params)

    reader = sitk.ImageFileReader()

    reader.SetFileName(instance_path)
    reader.ReadImageInformation()

    meta = {}
    for k, v in _dicom_meta_params.items():
        if reader.HasMetaDataKey(v):
            meta[k] = (reader.GetMetaData(v)).strip()
        else:
            logger.warning(f'missing data key ({k}): {v} in {reader.GetFileName()}')
            meta[k]=""

    return meta


def dicomdir2nii(dicomdir_ds_folder: str, output_folder: str, dicom_meta_params: Dict[str, str] = None,
                 series_description_filter: Union[None, List[str]] = None, codependent: bool = False,
                 codependent_tolerance: float = 0.1):
    """
    Converts DICOMDIR files to NIfTI format.

    Parameters:
        dicomdir_ds_folder (str): Path to the folder containing DICOMDIR files. Expects it in the format: ./Patent_ID/DICOMDIR
        output_folder (str): Output folder for NIfTI files.
        dicom_meta_params (Dict[str, str], optional): Additional DICOM metadata parameters to extract. Defaults to None.
        series_description_filter (Union[None, List[str]], optional): List of series description filters. Defaults to None.
        codependent (bool, optional): Whether to consider codependent series. Defaults to False.
        codependent_tolerance (float, optional): Tolerance for codependent series. Defaults to 0.1.

    Returns:
        pd.DataFrame: DataFrame containing DICOM metadata.
    """
    dicom_metas = []
    for dicomdir_file in tqdm(list(Path(dicomdir_ds_folder).rglob("DICOMDIR"))):
        dicom_meta = get_dicomdir_meta(str(dicomdir_file), dicom_meta_params=dicom_meta_params)
        filtered_meta = dicom_meta2nii(dicom_meta,
                                       output_folder=output_folder,
                                       series_description_filter=series_description_filter,
                                       codependent=codependent,
                                       codependent_tolerance=codependent_tolerance,
                                       patient_id=dicomdir_file.parent.name)
        dicom_metas.append(filtered_meta)
    return pd.concat(dicom_metas, axis=0).reset_index(drop=True)
