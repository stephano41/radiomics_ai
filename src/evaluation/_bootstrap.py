import json
import multiprocessing
import os
import time
from functools import partial
from pathlib import Path
from typing import Dict

import mlflow
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.metrics import Scorer
from sklearn.metrics import make_scorer, roc_curve
import pickle

from .roc_curve import plot_roc_curve_with_ci
from .stratified_bootstrap import BootstrapGenerator


# metrics to include: positive predictive value, negative predictive, sensitivity, specificity, AUC
class Bootstrap:
    def __init__(self, X, Y, iters: int = 500, alpha: float = 0.95, num_cpu: int = 2, method: str = '.632',
                 stratify=False, labels=None, log_dir=None):
        if method not in [".632", ".632+", "oob"]:
            raise ValueError(f"invalid bootstrap method {method}")

        self.X = X
        self.Y = Y
        self.is_multiclass = len(np.unique(Y)) > 2

        oob = BootstrapGenerator(n_splits=iters, stratify=stratify)
        self.oob_splits = list(oob.split(X, Y))

        self.alpha = alpha
        self.num_cpu = num_cpu
        self.method = method
        self.log_dir = log_dir
        self.labels = labels

        self._meta_file_path = os.path.join(log_dir, 'oob_splits.pkl')
        self._scores_path = os.path.join(log_dir, 'bootstrap_scores.pkl')
        self.scores = []

        if log_dir is not None:
            if os.path.exists(self._meta_file_path):
                #load the generator
                with open(self._meta_file_path, 'rb') as f:
                    self.oob_splits = pickle.load(f)
                print(f"loaded existing bootstrap splits at {self._meta_file_path}")
                # start at the right index
                with open(self._scores_path, 'rb') as f:
                    self.scores=pickle.load(f)
                    start_index = len(self.scores)
                self.oob_splits = self.oob_splits[start_index:]

            else:
                # save the generator for next time
                with open(self._meta_file_path, 'wb') as f:
                    pickle.dump(self.oob_splits, f)

    def run(self, model):
        score_func = Scorer(multiclass=self.is_multiclass, labels=self.labels)

        partial_bootstrap = partial(self._one_bootstrap, model=model, scoring_func=score_func)

        if self.num_cpu > 1:
            with multiprocessing.get_context('spawn').Pool(self.num_cpu) as pool:
                for score in tqdm(pool.imap_unordered(partial_bootstrap, self.oob_splits),
                                  total=len(self.oob_splits)):
                    self.scores.append(score)
                    self.log_scores(self.scores)
        else:
            for idx in tqdm(self.oob_splits):
                self.scores.append(partial_bootstrap(idx))
                self.log_scores(self.scores)

        return self.post_scores_processing(self.scores)

    def post_scores_processing(self, scores):
        pd_scores = pd.concat([t[0] for t in scores], ignore_index=True)
        ci = self.get_ci_each_col(pd_scores)

        fpr_tpr = [t[1] for t in scores]
        final_fpr_tpr = None
        if not self.is_multiclass:
            # unpack the fpr and tpr into a dictionary
            final_fpr_tpr = {'fpr': [], 'tpr': []}
            for d in fpr_tpr:
                final_fpr_tpr['fpr'].append(d['fpr'])
                final_fpr_tpr['tpr'].append(d['tpr'])
                # final_fpr_tpr['thresholds'].append(d['thresholds'])
            final_fpr_tpr['auc'] = pd_scores['roc_auc'].tolist()

        return ci, final_fpr_tpr

    def get_ci_each_col(self, df):
        result = {}
        for column in df:
            series = df[column]
            result[series.name] = get_ci(series.array, self.alpha)

        return result

    def log_scores(self, scores):
        if self.log_dir is not None:
            with open(self._scores_path, 'wb') as f:
                pickle.dump(scores, f)

    # def _one_bootstrap(self, idx, model, scoring_func):
    #     print(idx)
    #     time.sleep(0.5)
    #     if not self.is_multiclass:
    #         return np.random.random(), {'fpr': np.random.random(10), 'tpr': np.random.random(10), 'thresholds': np.random.random(10)}
    #     return np.random.random(), None
    def _one_bootstrap(self, idx, model, scoring_func):
        train_idx = idx[0]
        test_idx = idx[1]
        # print('hey')
        model.fit(index_array(self.X, train_idx), index_array(self.Y, train_idx))

        test_acc = scoring_func(model, index_array(self.X, test_idx), index_array(self.Y, test_idx))
        test_err = 1 - test_acc
        # training error on the whole training set as mentioned in the
        # previous comment above
        train_acc = scoring_func(model, self.X, self.Y)
        train_err = 1 - train_acc

        if self.method == "oob":
            acc = test_acc
        else:
            if self.method == ".632+":
                gamma = 1 - scoring_func.no_information_rate(model, self.X, self.Y)
                R = (test_err - train_err) / (gamma - train_err)
                weight = 0.632 / (1 - 0.368 * R)

            else:
                weight = 0.632

            acc = 1 - (weight * test_err + (1.0 - weight) * train_err)

        if not self.is_multiclass:
            roc_curve_metric = make_scorer(roc_curve)
            roc_curve_results = roc_curve_metric(model, index_array(self.X, test_idx), index_array(self.Y, test_idx))
            return acc, dict(zip(['fpr', 'tpr', 'thresholds'], roc_curve_results))
        return acc, None


def bootstrap(model, X, Y, iters: int = 500, alpha: float = 0.95, num_cpu: int = 2, method: str = '.632',
              num_gpu=0, labels=None, stratify=False):
    if method not in [".632", ".632+", "oob"]:
        raise ValueError(f"invalid bootstrap method {method}")

    oob = BootstrapGenerator(n_splits=iters, stratify=stratify)

    is_multiclass = len(np.unique(Y)) > 2

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
        final_fpr_tpr = {'fpr': [], 'tpr': []}
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
        return acc, dict(zip(['fpr', 'tpr', 'thresholds'], roc_curve_results))
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
