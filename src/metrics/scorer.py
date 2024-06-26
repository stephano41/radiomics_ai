from itertools import product
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import get_scorer, make_scorer, roc_curve

from src.metrics import specificity, negative_predictive_value
from autorad.metrics import roc_auc

class Scorer:
    def __init__(self, multiclass=False, labels: List = None):
        """
        Class for evaluating the performance of classification models using various scoring metrics.

        Parameters:
            multiclass (bool, optional): If True, the class handles multiclass classification problems.
                Default is False (binary classification).
            labels (List, optional): List of unique class labels. Required only for multiclass problems.
                Default is None.

        Note:
            This class works with classifiers that provide either 'decision_function' or 'predict_proba' methods
            for obtaining predicted probabilities.

        Scoring Metrics:
            - Binary Classification:
                * recall: Recall (sensitivity)
                * specificity: Specificity
                * precision: Precision
                * negative_predictive_value: Negative Predictive Value
                * roc_auc: Receiver Operating Characteristic (ROC) Area Under the Curve

            - Multiclass Classification:
                * recall_micro: Micro-averaged recall
                * recall_macro: Macro-averaged recall
                * specificity_micro: Micro-averaged specificity
                * specificity_macro: Macro-averaged specificity
                * precision_micro: Micro-averaged precision
                * precision_macro: Macro-averaged precision
                * negative_predictive_value_micro: Micro-averaged Negative Predictive Value
                * negative_predictive_value_macro: Macro-averaged Negative Predictive Value
                * roc_auc_ovr: One-vs-Rest (OvR) ROC Area Under the Curve
                * roc_auc_ovo: One-vs-One (OvO) ROC Area Under the Curve
        """
        if multiclass:
            self._scores = {'recall_micro': get_scorer('recall_micro'),  # equivalent to sensitivity
                            'recall_macro': get_scorer('recall_macro'),
                            'specificity_micro': make_scorer(specificity, average='micro', labels=labels),
                            'specificity_macro': make_scorer(specificity, average='macro', labels=labels),
                            'precision_micro': get_scorer('precision_micro'),  # equivalent to positive predictive value
                            'precision_macro': get_scorer('precision_macro'),
                            'negative_predictive_value_micro': make_scorer(negative_predictive_value, average='micro',
                                                                           labels=labels),
                            'negative_predictive_value_macro': make_scorer(negative_predictive_value, average='macro',
                                                                           labels=labels),
                            'roc_auc_ovr': make_scorer(roc_auc, average='macro', multi_class='ovr', labels=labels,
                                                       needs_proba=True),
                            'roc_auc_ovo': make_scorer(roc_auc, average='macro', multi_class='ovo', labels=labels,
                                                       needs_proba=True)
                            }
        else:
            self._scores = {'recall': get_scorer('recall'),
                            'specificity': make_scorer(specificity),
                            'precision': get_scorer('precision'),
                            'negative_predictive_value': make_scorer(negative_predictive_value),
                            "roc_auc": get_scorer('roc_auc')}

    def score(self, estimator, X, y) -> pd.DataFrame:
        """
        Calculate and return the performance scores for the given estimator.
        """

        result = {}
        for name, func in self._scores.items():
            result[name] = func(estimator, X, y)

        return pd.DataFrame(result, index=[0])

    def __call__(self, *args, **kwargs):
        return self.score(*args, **kwargs)

    def no_information_rate(self, estimator, X, y) -> pd.DataFrame:
        """
        Calculate and return the no information rate for each scoring metric.
        """
        result = {}
        for name, s in self._scores.items():
            y_pred = self.get_output(estimator, X, s)
            combinations = pd.DataFrame(list(product(y, y_pred)))
            result[name] = s._score_func(np.asarray(list(combinations.iloc[:, 0])),
                                         np.asarray(list(combinations.iloc[:, 1])), **s._kwargs)

        return pd.DataFrame(result, index=[0])

    def get_output(self, estimator, X, scorer) -> pd.DataFrame:
        """
        Get the predicted output of the estimator based on the given scorer.
        """
        try:
            y_pred = estimator.predict_proba(X)
        except (NotImplementedError, AttributeError):
            y_pred = estimator.predict(X)
        
        if y_pred.shape[1] <=2:
            y_pred = y_pred[:, 1]
        return y_pred
