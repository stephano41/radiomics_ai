from collections import defaultdict
import mlflow
from autorad.data import FeatureDataset
from autorad.models import MLClassifier
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
import hydra
import os
import logging

from src.metrics import roc_auc
from src.pipeline.pipeline_components import get_multimodal_feature_dataset, split_feature_dataset


logger = logging.getLogger(__name__)


def get_sample_size(config):
    run = mlflow.get_run(config.run_id)
    artifact_uri = run.info.artifact_uri

    model = MLClassifier.load_from_mlflow(f"{artifact_uri}/model")
    pipeline = mlflow.sklearn.load_model(f'{artifact_uri}/preprocessor').pipeline
    pipeline.steps.append(['estimator', model])

    feature_dataset = get_multimodal_feature_dataset(**OmegaConf.to_container(config.feature_dataset, resolve=True))

    dataset_size = len(feature_dataset.df)
    output_dir = hydra.utils.HydraConfig.get().run.dir
    sample_sizes = _get_sample_sizes(dataset_size)
    all_aucs = defaultdict(list)
    for sample_size in sample_sizes:
        logger.info(f'evaluating sample size of {sample_size}')
        sample_x, _ = train_test_split(feature_dataset.df, train_size=sample_size,
                                     stratify=feature_dataset.y)
        sample_feature_ds = FeatureDataset(sample_x, target=config.feature_dataset.get('target_column', None), ID_colname='ID',
                              additional_features=config.feature_dataset.get('additional_features', []))
        sample_feature_ds = split_feature_dataset(sample_feature_ds,
                                            save_path=os.path.join(output_dir, f'n={sample_size}_splits.yml'),
                                            **config.split)

        for X_train, y_train, X_val, y_val in zip(sample_feature_ds.data.X.train_folds,
                                                  sample_feature_ds.data.y.train_folds,
                                                  sample_feature_ds.data.X.val_folds,
                                                  sample_feature_ds.data.y.val_folds):
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict_proba(X_val)

            auc = roc_auc(y_val, y_pred, multi_class=config.multi_class, labels=config.labels)
            all_aucs[sample_size].append(auc)

        logger.info(f'{sample_size} yielded auc of {np.nanmean(all_aucs[sample_size])}')

    logger.info('sample size calculation complete!')
    plot_cv_sample_curve(sample_sizes, list(all_aucs.values()),
                              y_label='ROC AUC',
                              save_dir=os.path.join(output_dir,'sample_size_calculation.png'))


def _get_sample_sizes(dataset_size, min=16):
    exponentials = []
    while min < dataset_size:
        exponentials.append(min)
        min *= 2
    exponentials.append(dataset_size)
    return exponentials


def inverse_power_curve(x, a, b, c):
    return (1-a) - b * np.power(x, c)

def plot_inverse_power_curve(x, y, derivative_threshold=0.00001):
    popt, _ = curve_fit(inverse_power_curve, x, y, bounds=([-np.inf, -np.inf, -1],[np.inf, np.inf, 0]), maxfev=10000)
    last_x = x[-2]

    derivative = 1
    while derivative_threshold <= derivative:
        if inverse_power_curve(last_x, *popt) > 1:
            last_x /= 2
            break
        # Double the last sample size
        last_x *= 2
        # Compute the derivative of the curve at the last point
        derivative = (inverse_power_curve(last_x, *popt) - inverse_power_curve(last_x / 2,*popt)) / (
                                 last_x - last_x / 2)

    sample_sizes_extended = np.linspace(x[0], last_x, 100)
    curve_extended = inverse_power_curve(sample_sizes_extended, *popt)
    plt.plot(sample_sizes_extended, curve_extended, '--', color='gray')


def plot_cv_sample_curve(sample_sizes, cv_scores, y_label='ROC AUC', save_dir=None):
    """
    :param sample_sizes: list of numbers
    :param cv_scores: expects in list of tuples [(a,b,c,d),(e,f,g,h)...]
    :return:
    """
    assert len(cv_scores) == len(sample_sizes)
    mean = np.nanmean(cv_scores)
    plt.scatter(sample_sizes, mean, color='black', s=6)

    for i, _ in enumerate(sample_sizes):
        plt.scatter([sample_sizes[i]]*len(cv_scores[i]), cv_scores[i], color='red', s=1)

    plot_inverse_power_curve(sample_sizes, mean)

    plt.xlabel('Sample Size')
    plt.ylabel(y_label)
    plt.grid(True)
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(save_dir, dpi=300)
    plt.show()

