import os
from datetime import datetime
from pathlib import Path
import logging

import hydra
import pandas as pd
import yaml
from autorad.models import MLClassifier
from autorad.preprocessing import run_auto_preprocessing
from sklearn.pipeline import Pipeline

from src.evaluation import bootstrap

from src.pipeline.utils import get_data, get_feature_dataset
from src.training import Trainer
from src.inference import get_artifacts_from_last_run

logger = logging.getLogger(__name__)


def draft_pipeline(config):
    # setup dataset, extract features, split the data
    dataset = get_data(**config.dataset)

    feature_dataset = get_feature_dataset(image_dataset=dataset,
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

    pipeline: Pipeline = artifacts["preprocessor"].pipeline
    pipeline.steps.append(['estimator', artifacts['model']])

    logger.info(bootstrap(pipeline, feature_dataset.X.to_numpy(), feature_dataset.y.to_numpy(), iters=20, num_cpu=4,
                    labels=[0, 1, 2], method='.632+'))


def wiki_sarcoma_df_merger(label_df: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    merged_feature_df = feature_df.merge(label_df,
                                         left_on="ID",
                                         right_on="Patient ID",
                                         how="left")
    merged_feature_df = merged_feature_df[merged_feature_df['Grade'].notna()]

    merged_feature_df['Grade'] = merged_feature_df['Grade'].map(
        {v: k for k, v in enumerate(merged_feature_df['Grade'].unique())})

    return merged_feature_df
