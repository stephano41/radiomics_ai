from functools import partial
from multiprocessing.pool import Pool
from typing import Dict

import mlflow
import pandas as pd
import ray
import numpy as np
from .stratified_bootstrap import BootstrapGenerator
from tqdm import tqdm

from src.metrics import Scorer


# metrics to include: positive predictive value, negative predictive, sensitivity, specificity, AUC


def bootstrap(model, X, Y, iters: int = 500, alpha: float = 0.95, num_cpu: int = 2, method: str = '.632',
              num_gpu=0, labels=None):
    if method not in [".632", ".632+", "oob"]:
        raise ValueError(f"invalid bootstrap method {method}")

    oob = BootstrapGenerator(n_splits=iters, stratify=True)

    score_func = Scorer(multiclass=len(np.unique(Y)>2),
                        labels=labels)
    scores = []

    if num_gpu <= 0:
        # use default python
        partial_bootstrap = partial(_one_bootstrap, model=model, scoring_func=score_func,
                                    X=X, Y=Y, method=method)
        with Pool(num_cpu) as pool:
            for score in tqdm(pool.imap_unordered(partial_bootstrap, oob.split(X, Y)), total=oob.n_splits):
                scores.append(score)
    else:
        remote_bootstrap = ray.remote(num_gpus=num_gpu, max_calls=1)(_one_bootstrap)
        model_id, X_id, Y_id, score_func_id, method_id = ray.put(model), ray.put(X), ray.put(Y), ray.put(
            score_func), ray.put(method)
        scores = ray.get([remote_bootstrap.remote(idx, model=model_id, scoring_func=score_func_id,
                                                  X=X_id, Y=Y_id, method=method_id) for idx in
                          oob.split(X, Y)])
        ray.shutdown()

    return get_ci_each_col(pd.concat(scores, ignore_index=True), alpha)


def log_ci2mlflow(ci_dict: Dict, run_id=None):
    metrics_dict={}

    for name, (lower, upper) in ci_dict.items():
        metrics_dict[name + '_ll'] = lower
        metrics_dict[name + '_ul'] = upper

    with mlflow.start_run(run_id=run_id):
        # log original confidence interval dict for future presentation
        mlflow.log_dict(ci_dict, "confidence_intervals.yaml")
        # log to metrics to display in mlflow
        mlflow.log_metrics(metrics_dict)


def _one_bootstrap(idx, model, scoring_func: Scorer, X, Y, method='.632'):
    train_idx = idx[0]
    test_idx = idx[1]
    # print('hey')
    model.fit(X[train_idx], Y[train_idx])

    test_acc = scoring_func(model, X[test_idx], Y[test_idx])
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
    return acc


def get_ci_each_col(df, alpha=0.95):
    result={}
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

