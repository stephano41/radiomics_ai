import pandas as pd
from sklearn.metrics import get_scorer, make_scorer

from src.metrics import specificity, negative_predictive_value


class Scorer:
    def __init__(self, multiclass=False):
        self._running_scores=[]
        if multiclass:
            self._scores = {'recall_micro': get_scorer('recall_micro'), # equivalent to sensitivity
                            'recall_macro': get_scorer('recall_macro'),
                            'specificity_micro': make_scorer(specificity, average='micro'),
                            'specificity_macro': make_scorer(specificity, average='macro'),
                            'precision_micro': get_scorer('precision_micro'), # equivalent to positive predictive value
                            'precision_macro': get_scorer('precision_macro'),
                            'negative_predictive_value_micro': make_scorer(negative_predictive_value, average='micro'),
                            'negative_predictive_value_macro': make_scorer(negative_predictive_value, average='macro'),
                            'roc_auc_ovr': get_scorer('roc_auc_ovr'),
                            'roc_auc_ovo': get_scorer('roc_auc_ovo')
                            }
        else:
            self._scores = {'recall': get_scorer('recall'),
                            'specificity': make_scorer(specificity),
                            'precision': get_scorer('precision'),
                            'negative_predictive_value': make_scorer(negative_predictive_value),
                            "roc_auc": get_scorer('roc_auc')}

    def score(self, estimator, X, y):
        result = {}
        for name, func in self._scores.items():
            result[name] = func(estimator, X, y)
        return pd.DataFrame(result, index=[0])

    def __call__(self, *args, **kwargs):
        return self.score(*args, **kwargs)
