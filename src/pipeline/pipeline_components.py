from typing import Tuple

import pandas as pd
import yaml
from autorad.data import ImageDataset, FeatureDataset
from autorad.feature_extraction import FeatureExtractor
from autorad.utils.extraction_utils import filter_pyradiomics_names
from autorad.utils.preprocessing import get_paths_with_separate_folder_per_case
from hydra.utils import instantiate

from src.utils.prepro_utils import get_multi_paths_with_separate_folder_per_case


def get_data(data_dir, image_stem='image', mask_stem='mask') -> ImageDataset:
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


def get_multimodal_feature_dataset(data_dir, label_csv_path, target_column, image_stems: Tuple[str, ...] = 'image',
                                   mask_stem='mask', label_csv_encoding=None, additional_features=[],
                                   extraction_params="mr_default.yml", n_jobs=-1,
                                   feature_df_merger=None, existing_feature_df=None) -> FeatureDataset:
    if existing_feature_df is not None:
        return FeatureDataset(pd.read_csv(existing_feature_df), target=target_column, ID_colname='ID',
                              additional_features=additional_features)

    paths_df = get_multi_paths_with_separate_folder_per_case(data_dir,
                                                             relative=True,
                                                             image_stems=image_stems,
                                                             mask_stem=mask_stem
                                                             )

    feature_dfs = []
    for image_stem in image_stems:
        image_dataset = ImageDataset(paths_df,
                                     ID_colname='ID',
                                     image_colname=f'image_{image_stem}',
                                     root_dir=data_dir)

        extractor = FeatureExtractor(image_dataset, extraction_params=extraction_params, n_jobs=n_jobs)

        feature_df = extractor.run()

        feature_df = feature_df.rename(columns=
                                       {feature_name: f'{feature_name}_{image_stem}' for feature_name in
                                        filter_pyradiomics_names(feature_df.columns.tolist())})

        feature_dfs.append(feature_df)

    # feature_dfs.insert(0, feature_df[exclusion_columns])
    df = pd.concat(feature_dfs, axis=1)

    all_feature_df = df.loc[:, ~df.columns.duplicated()]

    label_df = pd.read_csv(label_csv_path, encoding=label_csv_encoding)

    merged_feature_df = instantiate(feature_df_merger, label_df=label_df, feature_df=all_feature_df)

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
