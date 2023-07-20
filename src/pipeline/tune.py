import os
from datetime import datetime
from pathlib import Path
import logging

import hydra
import pandas as pd
import yaml
from autorad.data import FeatureDataset
from autorad.models import MLClassifier
from autorad.preprocessing import run_auto_preprocessing
from autorad.utils.preprocessing import get_paths_with_separate_folder_per_case
from hydra.utils import instantiate
from autorad import evaluation

from src.dataset import ImageDataset
from src.feature_extraction import FeatureExtractor
from src.training import Trainer
from src.inference import get_artifacts_from_last_run

logger = logging.getLogger(__name__)


def draft_pipeline(config):
    # setup dataset, extract features, split the data
    dataset = get_data(**config.dataset)

    feature_dataset = get_feature_dataset(dataset,
                                          **config.feature_dataset)

    output_dir = hydra.utils.HydraConfig.get().run.dir
    # save the feature_dataset
    feature_dataset.df.to_csv(os.path.join(output_dir, 'extracted_features.csv'))

    if config.split.existing_split is None:
        feature_dataset.split(method=config.split.method, save_path=os.path.join(output_dir, 'splits.yml'))
    else:
        with open(config.split.existing_split, 'r') as f:
            feature_dataset.load_splits((yaml.safe_load(f)))

    # initialise models
    if config.models is None:
        models = MLClassifier.initialize_default_sklearn_models()
    else:
        models = [MLClassifier.from_sklearn(model_name) for model_name in config.models]

    # run auto preprocessing
    run_auto_preprocessing(data=feature_dataset.data,
                           result_dir=Path(output_dir),
                           **config.preprocessing)

    # start training
    trainer = Trainer(
        dataset=feature_dataset,
        models=models,
        result_dir=output_dir,
        multi_class=config.multi_class,
        labels=config.labels
    )
    if config.name is None:
        experiment_name = datetime.now().strftime('%Y%m%d%H%M%S')
    else:
        experiment_name = config.name

    trainer.set_optimizer('optuna', n_trials=config.optimizer.n_trials)
    trainer.run(auto_preprocess=True, experiment_name=experiment_name)

    # start evaluation
    artifacts = get_artifacts_from_last_run(experiment_name)

    result_df = evaluation.evaluate_feature_dataset(dataset=feature_dataset,
                                                    model=artifacts['model'],
                                                    preprocessor=artifacts["preprocessor"],
                                                    split="test")
    print(result_df)


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


def get_feature_dataset( target_column: str, image_dataset=None, label_csv_path=None, extraction_params="mr_default.yml",
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


def wiki_sarcoma_df_merger(label_df: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    merged_feature_df = feature_df.merge(label_df,
                                         left_on="ID",
                                         right_on="Patient ID",
                                         how="left")
    merged_feature_df = merged_feature_df[merged_feature_df['Grade'].notna()]

    merged_feature_df['Grade'] = merged_feature_df['Grade'].map(
        {v: k for k, v in enumerate(merged_feature_df['Grade'].unique())})

    return merged_feature_df
