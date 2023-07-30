import os
from datetime import datetime
from pathlib import Path
import logging

import hydra
import pandas as pd

from src.evaluation import bootstrap, log_ci2mlflow
from src.pipeline.utils import get_data, get_feature_dataset, split_feature_dataset, get_multimodal_feature_dataset
from src.training import Trainer
from src.inference import get_pipeline_from_last_run, get_last_run_from_experiment_name
from src.preprocessing import run_auto_preprocessing
from src.models import MLClassifier

logger = logging.getLogger(__name__)


def draft_pipeline(config):
    # setup dataset, extract features, split the data
    feature_dataset = get_multimodal_feature_dataset(**config.feature_dataset)

    output_dir = hydra.utils.HydraConfig.get().run.dir
    # save the feature_dataset
    feature_dataset.df.to_csv(os.path.join(output_dir, 'extracted_features.csv'))

    feature_dataset = split_feature_dataset(feature_dataset,
                                            save_path=os.path.join(output_dir, 'splits.yml'),
                                            **config.split)

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
    pipeline = get_pipeline_from_last_run(experiment_name)

    confidence_interval = bootstrap(pipeline, feature_dataset.X.to_numpy(), feature_dataset.y.to_numpy(),
                                    **config.bootstrap)

    logger.info(confidence_interval)
    log_ci2mlflow(confidence_interval,
                  run_id=get_last_run_from_experiment_name(experiment_name).run_id)


def wiki_sarcoma_df_merger(label_df: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    merged_feature_df = feature_df.merge(label_df,
                                         left_on="ID",
                                         right_on="Patient ID",
                                         how="left")
    merged_feature_df = merged_feature_df[merged_feature_df['Grade'].notna()]

    merged_feature_df['Grade'] = merged_feature_df['Grade'].map(
        {v: k for k, v in enumerate(merged_feature_df['Grade'].unique())})

    return merged_feature_df

# TODO allow registration of images to combine multiple modalities
# TODO calibration score?
# TODO get hydra logging to work
# TODO get pytests to work


def meningioma_df_merger(label_df: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    merged_feature_df = feature_df.merge(label_df,
                                         left_on='ID',
                                         right_on="Patient_ID",
                                         how='left')
    merged_feature_df = merged_feature_df[merged_feature_df['Grade'].notna()]

    return merged_feature_df