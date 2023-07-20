from src.metrics.scorer import Scorer
from sklearn.dummy import DummyClassifier
import numpy as np


def test_multiclass_scorer():
    return _testnclass_scorer(3)


def test_binaryclass_scorer():
    return _testnclass_scorer(2)


def _testnclass_scorer(n):
    scorer = Scorer(list(range(n)))

    X = np.random.random([100, 10])
    y = np.random.randint(0, 3, size=100)
    clf = DummyClassifier(strategy="most_frequent")
    clf = clf.fit(X, y)
    return scorer(clf, X, y)
