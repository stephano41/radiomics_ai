import pandas as pd
from autorad.data import FeatureDataset
from autorad.utils.preprocessing import get_paths_with_separate_folder_per_case
from hydra.utils import instantiate

from src.dataset import ImageDataset
from src.feature_extraction import FeatureExtractor


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
                        n_jobs=None, label_csv_encoding=None, feature_df_merger=None,
                        existing_feature_df=None) -> FeatureDataset:
    if existing_feature_df is None:
        extractor = FeatureExtractor(image_dataset, extraction_params=extraction_params, n_jobs=n_jobs)

        feature_df = extractor.run()

        label_df = pd.read_csv(label_csv_path, encoding=label_csv_encoding)

        merged_feature_df = instantiate(feature_df_merger, label_df=label_df, feature_df=feature_df)

        return FeatureDataset(merged_feature_df, target=target_column, ID_colname='ID')

    else:
        return FeatureDataset(pd.read_csv(existing_feature_df),
                              target=target_column,
                              ID_colname='ID')
