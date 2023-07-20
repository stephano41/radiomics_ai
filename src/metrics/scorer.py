from itertools import product
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import get_scorer, make_scorer, roc_auc_score
from sklearn.metrics._scorer import _ThresholdScorer

from src.metrics import specificity, negative_predictive_value


class Scorer:
    def __init__(self, labels: List):
        self._running_scores=[]
        if len(labels) >2:
            self._scores = {'recall_micro': get_scorer('recall_micro'), # equivalent to sensitivity
                            'recall_macro': get_scorer('recall_macro'),
                            'specificity_micro': make_scorer(specificity, average='micro',labels=labels),
                            'specificity_macro': make_scorer(specificity, average='macro', labels=labels),
                            'precision_micro': get_scorer('precision_micro'), # equivalent to positive predictive value
                            'precision_macro': get_scorer('precision_macro'),
                            'negative_predictive_value_micro': make_scorer(negative_predictive_value, average='micro', labels=labels),
                            'negative_predictive_value_macro': make_scorer(negative_predictive_value, average='macro', labels=labels),
                            'roc_auc_ovr': make_scorer(roc_auc_score, average='macro', multi_class='ovr', labels=labels, needs_proba=True),
                            'roc_auc_ovo': make_scorer(roc_auc_score, average='macro', multi_class='ovo', labels=labels, needs_proba=True)
                            }
        else:
            self._scores = {'recall': get_scorer('recall'),
                            'specificity': make_scorer(specificity),
                            'precision': get_scorer('precision'),
                            'negative_predictive_value': make_scorer(negative_predictive_value),
                            "roc_auc": get_scorer('roc_auc')}

    def score(self, estimator, X, y) -> pd.DataFrame:

        result = {}
        for name, func in self._scores.items():
            result[name] = func(estimator, X, y)

        return pd.DataFrame(result, index=[0])

    def __call__(self, *args, **kwargs):
        return self.score(*args, **kwargs)

    def no_information_rate(self, estimator, X, y) -> pd.DataFrame:
        result={}
        for name, scorer in self._scores.items():
            y_pred = self.get_output(estimator, X, scorer)
            combinations = np.array(list(product(y, y_pred)))
            result[name] = scorer.score_func(combinations[:, 0], combinations[:, 1])

        return pd.DataFrame(result, index=[0])

    def get_output(self, estimator, X, scorer) -> pd.DataFrame:
        if isinstance(scorer, _ThresholdScorer):
            try:
                y_pred = estimator.decision_function(X)
            except (NotImplementedError, AttributeError):
                y_pred = estimator.predict_proba(X)
        else:
            y_pred = estimator.predict(X)
        return y_pred