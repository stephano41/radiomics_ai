from datetime import datetime
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
import hydra
import os
import seaborn as sns
import logging
import pickle
import pandas as pd
from src.evaluation import Bootstrap
from src.utils.pipeline import get_multimodal_feature_dataset
from sklearn.metrics import auc as auc_metric
from autorad.inference.infer_utils import get_pipeline_from_run, get_run_info_as_series


logger = logging.getLogger(__name__)


def get_sample_size(config):
    feature_dataset = get_multimodal_feature_dataset(**OmegaConf.to_container(config.feature_dataset, resolve=True))
    # feature_dataset = split_feature_dataset(feature_dataset,
    #                                         save_path=os.path.join(output_dir, 'splits.yml'),
    #                                         **config.split)

    dataset_size = len(feature_dataset.df)
    output_dir = hydra.utils.HydraConfig.get().run.dir
    experiment_name = f"sample_size_calculation"

    pipeline = get_pipeline_from_run(get_run_info_as_series(config.run_id))

    confidence_intervals, tpr_fprs, preds = [], [], []
    aucs=[]
    sample_sizes = config.get('sample_sizes', _get_sample_sizes(dataset_size, 25))
    for sample_size in sample_sizes:
        try:
            logger.info(f'evaluating at sample size {sample_size}')

            evaluator = Bootstrap(feature_dataset.X, feature_dataset.y, n_samples_per_iter=sample_size,**config.bootstrap)

            confidence_interval, tpr_fpr, pred = evaluator.run(pipeline)

            confidence_intervals.append(confidence_interval)
            tpr_fprs.append(tpr_fpr)
            preds.append(pred)
            aucs.append(np.array([auc_metric(d['fpr'], d['tpr']) for d in tpr_fpr]))

            with open(os.path.join(output_dir, f'sample_size_{sample_size}.pkl'), 'wb') as f:
                pickle.dump((confidence_interval, tpr_fpr, pred), f)

            logger.info(f"sample size of {sample_size} yielded {confidence_interval}")
        except Exception as e:
            logger.error(e)
            break

    logger.info('sample size calculation complete!')
    plot_confidence_intervals(sample_sizes, aucs,
                              y_label='ROC AUC',
                              save_dir=os.path.join(output_dir,'sample_size_calculation.png'))


def _get_sample_sizes(dataset_size, min=16):
    exponentials = []
    while min < dataset_size:
        exponentials.append(min)
        min *= 2
    exponentials.append(dataset_size)
    exponentials.reverse()
    return exponentials


def split_sample_w_minority(df, final_size, minority_size, stratify_array):
    if final_size==len(df):
        return df
    counts = stratify_array.value_counts()
    minority_class = counts[counts == counts.min()].index[0]

    minority_rows = df[stratify_array==minority_class].sample(minority_size)
    _, random_rows = train_test_split(df.drop(minority_rows.index), test_size=final_size-minority_size, 
                        stratify=stratify_array.drop(minority_rows.index))

    return pd.concat([random_rows.reset_index(), minority_rows.reset_index()])


def inverse_power_curve(x, a, b, c):
    return (1-a) - b * np.power(x, c)

def plot_inverse_power_curve(x, y, derivative_threshold=0.00001):
    popt, _ = curve_fit(inverse_power_curve, x, y, bounds=([-np.inf, -np.inf, -1],[np.inf, np.inf, 0]), maxfev=5000)
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


def plot_confidence_intervals(sample_sizes, aucs, y_label='roc_auc', save_dir=None):
    """
    :param sample_sizes: list of numbers
    :param confidence_intervals: expects in list of tuples [(a,b),(c,d)...]
    :return:
    """
    df = pd.DataFrame({sample_sizes[i]: confidence_interval for i, confidence_interval in enumerate(aucs)})
    df = df.iloc[::-1]

    ax = sns.boxplot(data=df, order=df.columns)

    ax.set_xlabel("Sample Size")
    ax.set_ylabel(y_label)
    ax.grid()
 
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(save_dir, dpi=1200)
        return ax
    plt.show()