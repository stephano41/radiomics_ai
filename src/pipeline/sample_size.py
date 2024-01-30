from datetime import datetime
from pathlib import Path

from autorad.models import MLClassifier
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
import hydra
import os
import logging

from src.evaluation import Bootstrap
from src.pipeline.pipeline_components import get_multimodal_feature_dataset, split_feature_dataset
from src.preprocessing import run_auto_preprocessing
from src.training import Trainer
from src.utils.infer_utils import get_pipeline_from_last_run


logger = logging.getLogger(__name__)


def get_sample_size(config):
    feature_dataset = get_multimodal_feature_dataset(**OmegaConf.to_container(config.feature_dataset, resolve=True))
    # feature_dataset = split_feature_dataset(feature_dataset,
    #                                         save_path=os.path.join(output_dir, 'splits.yml'),
    #                                         **config.split)

    dataset_size = len(feature_dataset.df)
    output_dir = hydra.utils.HydraConfig.get().run.dir
    experiment_name = f"get_sample_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    confidence_intervals, tpr_fprs = [], []
    sample_sizes = _get_sample_sizes(dataset_size)
    for sample_size in sample_sizes:
        sample_x, _ = train_test_split(feature_dataset.df, test_size=sample_size,
                                     stratify=feature_dataset.y)
        config.feature_dataset.existing_feature_df = sample_x
        sample_feature_ds = get_multimodal_feature_dataset(**OmegaConf.to_container(config.feature_dataset, resolve=True))
        sample_feature_ds = split_feature_dataset(sample_feature_ds,
                                            save_path=os.path.join(output_dir, f'n={sample_size}_splits.yml'),
                                            **config.split)

        if config.models is None:
            models = MLClassifier.initialize_default_sklearn_models()
        else:
            models = [MLClassifier.from_sklearn(model_name) for model_name in config.models]

        run_auto_preprocessing(data=sample_feature_ds.data,
                               result_dir=Path(output_dir),
                               **OmegaConf.to_container(config.preprocessing, resolve=True))

        trainer = Trainer(
            dataset=sample_feature_ds,
            models=models,
            result_dir=output_dir,
            multi_class=config.multi_class,
            labels=config.labels
        )

        trainer.set_optimizer('optuna', n_trials=config.optimizer.n_trials)
        trainer.run(auto_preprocess=True, experiment_name=experiment_name)

        pipeline = get_pipeline_from_last_run(config.name)

        evaluator = Bootstrap(feature_dataset.X, feature_dataset.y, **config.bootstrap)

        confidence_interval, tpr_fpr = evaluator.run(pipeline)

        confidence_intervals.append(confidence_interval)
        tpr_fprs.append(tpr_fpr)

        logger.info(confidence_interval)

    logger.info('sample size calculation complete!')
    plot_confidence_intervals(sample_sizes, [interval['roc_auc'] for interval in confidence_intervals],
                              y_label='roc_auc',
                              save_dir=os.path.join(output_dir,'sample_size_calculation.png'))


def _get_sample_sizes(dataset_size, min=16):
    exponentials = []
    while min < dataset_size:
        exponentials.append(min)
        min *= 2
    exponentials.append(dataset_size)
    return exponentials


def plot_confidence_intervals(sample_sizes, confidence_intervals, y_label='roc_auc', save_dir=None):
    """
    :param sample_sizes: list of numbers
    :param confidence_intervals: expects in list of tuples [(a,b),(c,d)...]
    :return:
    """
    starts, ends = zip(*confidence_intervals)
    plt.plot(sample_sizes, starts)
    plt.plot(sample_sizes, ends)
    plt.fill_between(sample_sizes, starts, ends, alpha=0.2)
    plt.xlabel('Sample Size')
    plt.ylabel(y_label)
    plt.grid(True)
    if save_dir is not None:
        plt.savefig(save_dir, dpi=300)
    plt.show()