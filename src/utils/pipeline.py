from typing import Optional, Sequence

import pandas as pd
import yaml
from autorad.data import ImageDataset, FeatureDataset
from autorad.feature_extraction import FeatureExtractor
from autorad.utils.extraction_utils import filter_pyradiomics_names
from autorad.utils.preprocessing import get_paths_with_separate_folder_per_case
from hydra.utils import instantiate

from src.utils.dataset import get_multi_paths_with_separate_folder_per_case


def get_data(data_dir, image_stem='image', mask_stem='mask') -> ImageDataset:
    """
       Loads and prepares an image dataset from a specified directory where images and their corresponding masks are
       stored in separate subfolders for each case.

       This function uses the utility `get_paths_with_separate_folder_per_case` to create a DataFrame that contains
       the paths to the image files and their corresponding mask files, relative to the given data directory.
       The DataFrame is then used to instantiate an `ImageDataset` object, which facilitates the loading and handling
       of medical image data for further processing or analysis.

       Parameters:
       ----------
       data_dir : str
           The root directory where image and mask subfolders are stored. Each subfolder represents a separate case
           and contains image files and corresponding mask files named according to the `image_stem` and `mask_stem` parameters.
       image_stem : str, optional
           The filename stem used to identify image files within each case subfolder (default is 'image').
       mask_stem : str, optional
           The filename stem used to identify mask files within each case subfolder (default is 'mask').

       Returns:
       -------
       ImageDataset
           An object of type `ImageDataset` that includes methods for accessing and manipulating the medical image
           and mask data. This object contains the paths to the images and masks, and provides an interface for
           further processing such as feature extraction or image transformations.

       Example:
       --------
       >>> data_directory = '/path/to/data'
       >>> image_dataset = get_data(data_directory)
       >>> print(image_dataset)
       ImageDataset containing data paths for images and masks with root directory at /path/to/data

       Notes:
       -----
       - The directory structure expected by this function should have separate subdirectories for each case,
         each containing images and masks with filenames starting with the specified stems.
       - This function is tailored to be used with medical image datasets where it is common to have separate
         image and mask files for each case.
       """
    paths_df = get_paths_with_separate_folder_per_case(data_dir,
                                                       relative=True,
                                                       image_stem=image_stem,
                                                       mask_stem=mask_stem)
    image_dataset = ImageDataset(
        paths_df,
        ID_colname='ID',
        root_dir=data_dir
    )
    return image_dataset


def get_feature_dataset(target_column: str, image_dataset=None, label_csv_path=None, extraction_params="mr_default.yml",
                        n_jobs=-1, label_csv_encoding=None, feature_df_merger=None,
                        existing_feature_df=None) -> FeatureDataset:
    if existing_feature_df is None:
        assert image_dataset is not None, f"param image_dataset cannot be None when existing_feature_df is not provided"
        extractor = FeatureExtractor(image_dataset, extraction_params=extraction_params, n_jobs=n_jobs)

        feature_df = extractor.run()

        label_df = pd.read_csv(label_csv_path, encoding=label_csv_encoding)

        merged_feature_df = instantiate(feature_df_merger, label_df=label_df, feature_df=feature_df)

        return FeatureDataset(merged_feature_df, target=target_column, ID_colname='ID')

    else:
        return FeatureDataset(pd.read_csv(existing_feature_df),
                              target=target_column,
                              ID_colname='ID')


def load_existing_features(existing_feature_path: str, target_column: str, additional_features) -> FeatureDataset:
    """ Load existing feature dataset from a CSV file. """
    return FeatureDataset(pd.read_csv(existing_feature_path), target=target_column, ID_colname='ID', additional_features=additional_features)


def extract_features(image_stems: Sequence[str], paths_df: pd.DataFrame, data_dir: str, extraction_params: str,
                     n_jobs: int) -> pd.DataFrame:
    """ Extract features for each image stem and concatenate them into a single DataFrame. """
    feature_dfs = []
    mlflow_run_id = None
    for image_stem in image_stems:
        image_dataset = ImageDataset(
            paths_df,
            ID_colname='ID',
            image_colname=f'image_{image_stem}',
            root_dir=data_dir
        )
        # TODO featureextractor to log the same mlflow run id
        extractor = FeatureExtractor(image_dataset, extraction_params=extraction_params, n_jobs=n_jobs)
        feature_df = extractor.run(run_id=mlflow_run_id)
        mlflow_run_id = extractor.run_id_
        feature_df = feature_df.rename(columns={
            name: f'{name}_{image_stem}' for name in filter_pyradiomics_names(feature_df.columns)
        })
        feature_dfs.append(feature_df)

    return pd.concat(feature_dfs, axis=1).loc[:, ~pd.concat(feature_dfs, axis=1).columns.duplicated()]


def merge_labels(feature_df: pd.DataFrame, label_csv_path: str, label_csv_encoding: Optional[str],
                 merger) -> pd.DataFrame:
    """ Merge feature DataFrame with labels. """
    label_df = pd.read_csv(label_csv_path, encoding=label_csv_encoding)
    return instantiate(merger, label_df=label_df, feature_df=feature_df)


def get_multimodal_feature_dataset(
        target_column: str,
        data_dir: Optional[str] = None,
        label_csv_path: Optional[str] = None,
        image_stems: Sequence[str] = ('image',),
        mask_stem: str = 'mask',
        label_csv_encoding: Optional[str] = None,
        additional_features: Sequence[str] = None,
        extraction_params: str = "mr_default.yml",
        n_jobs: int = -1,
        feature_df_merger=None,
        existing_feature_df: Optional[str] = None
) -> FeatureDataset:
    """ Orchestrates the creation of a multimodal feature dataset. """
    if additional_features is None:
        additional_features = [] 
    
    if existing_feature_df:
        return load_existing_features(existing_feature_df, target_column, additional_features)

    paths_df = get_multi_paths_with_separate_folder_per_case(
        data_dir, relative=True, image_stems=image_stems, mask_stem=mask_stem
    )
    all_feature_df = extract_features(image_stems, paths_df, data_dir, extraction_params, n_jobs)
    merged_feature_df = merge_labels(all_feature_df, label_csv_path, label_csv_encoding, feature_df_merger)


    return FeatureDataset(merged_feature_df, target=target_column, ID_colname='ID',
                          additional_features=additional_features)


def split_feature_dataset(feature_dataset: FeatureDataset, existing_split=None, save_path=None,
                          method='train_with_cross_validation_test', split_on=None, test_size=0.2, *args, **kwargs
                          ):
    if existing_split is None:
        feature_dataset.split(save_path=save_path, method=method, split_on=split_on, test_size=test_size, *args,
                              **kwargs)
    else:
        with open(existing_split, 'r') as f:
            feature_dataset.load_splits((yaml.safe_load(f)))

    return feature_dataset
