import multiprocessing
from functools import partial
from typing import Dict

import mlflow
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.metrics import Scorer
from sklearn.metrics import make_scorer, roc_curve

from .roc_curve import plot_roc_curve_with_ci
from .stratified_bootstrap import BootstrapGenerator


# metrics to include: positive predictive value, negative predictive, sensitivity, specificity, AUC


def bootstrap(model, X, Y, iters: int = 500, alpha: float = 0.95, num_cpu: int = 2, method: str = '.632',
              num_gpu=0, labels=None, stratify=False):
    if method not in [".632", ".632+", "oob"]:
        raise ValueError(f"invalid bootstrap method {method}")

    oob = BootstrapGenerator(n_splits=iters, stratify=stratify)

    is_multiclass=len(np.unique(Y)) > 2

    score_func = Scorer(multiclass=is_multiclass,
                        labels=labels)
    scores = []

    partial_bootstrap = partial(_one_bootstrap, model=model, scoring_func=score_func,
                                X=X, Y=Y, method=method, multiclass=is_multiclass)

    if num_cpu > 1:

        with multiprocessing.get_context('spawn').Pool(num_cpu) as pool:
            for score in tqdm(pool.imap_unordered(partial_bootstrap, oob.split(X, Y)), total=oob.n_splits):
                scores.append(score)
    else:
        scores = [partial_bootstrap(idx) for idx in tqdm(oob.split(X, Y))]

    pd_scores = pd.concat([t[0] for t in scores], ignore_index=True)
    ci = get_ci_each_col(pd_scores, alpha)

    fpr_tpr = [t[1] for t in scores]
    final_fpr_tpr = None
    if not is_multiclass:
        final_fpr_tpr = {'fpr':[], 'tpr':[]}
        for d in fpr_tpr:
            final_fpr_tpr['fpr'].append(d['fpr'])
            final_fpr_tpr['tpr'].append(d['tpr'])
            # final_fpr_tpr['thresholds'].append(d['thresholds'])
        final_fpr_tpr['auc'] = pd_scores['roc_auc'].tolist()

    return ci, final_fpr_tpr


def _one_bootstrap(idx, model, scoring_func: Scorer, X, Y, method='.632', multiclass=False):
    train_idx = idx[0]
    test_idx = idx[1]
    # print('hey')
    model.fit(index_array(X, train_idx), index_array(Y, train_idx))

    test_acc = scoring_func(model, index_array(X, test_idx), index_array(Y, test_idx))
    test_err = 1 - test_acc
    # training error on the whole training set as mentioned in the
    # previous comment above
    train_acc = scoring_func(model, X, Y)
    train_err = 1 - train_acc

    if method == "oob":
        acc = test_acc
    else:
        if method == ".632+":
            gamma = 1 - scoring_func.no_information_rate(model, X, Y)
            R = (test_err - train_err) / (gamma - train_err)
            weight = 0.632 / (1 - 0.368 * R)

        else:
            weight = 0.632

        acc = 1 - (weight * test_err + (1.0 - weight) * train_err)

    if not multiclass:
        roc_curve_metric = make_scorer(roc_curve)
        roc_curve_results = roc_curve_metric(model, index_array(X, test_idx), index_array(Y, test_idx))
        return acc, dict(zip(['fpr','tpr','thresholds'], roc_curve_results))
    return acc, None




def log_ci2mlflow(ci_dict: Dict, tpr_fpr: dict = None, run_id=None):
    metrics_dict = {}

    for name, (lower, upper) in ci_dict.items():
        metrics_dict[name + '_ll'] = lower
        metrics_dict[name + '_ul'] = upper

    with mlflow.start_run(run_id=run_id):
        # log original confidence interval dict for future presentation
        mlflow.log_dict(ci_dict, "confidence_intervals.json")
        if tpr_fpr is not None:
            mlflow.log_figure(plot_roc_curve_with_ci(tpr_fpr), 'roc_curve.png')
        # log to metrics to display in mlflow
        mlflow.log_metrics(metrics_dict)


def index_array(a, idx):
    if isinstance(a, (pd.DataFrame, pd.Series)):
        return a.reset_index(drop=True).loc[idx]
    else:

        return a[idx]


def get_ci_each_col(df, alpha=0.95):
    result = {}
    for column in df:
        series = df[column]
        result[series.name] = get_ci(series.array, alpha)

    return result


def get_ci(data, alpha=0.95):
    p = ((1.0 - alpha) / 2.0) * 100
    lower = max(0.0, np.percentile(data, p))
    p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    upper = min(1.0, np.percentile(data, p))
    return lower, upper
