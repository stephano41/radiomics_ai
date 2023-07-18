from src.metrics.scorer import Scorer
from sklearn.dummy import DummyClassifier
import numpy as np


def test_multiclass_scorer():
    scorer = Scorer(True)

    X = np.random.random([100, 10])
    y = np.random.randint(0, 3, size=100)
    clf = DummyClassifier(strategy="most_frequent")
    clf = clf.fit(X, y)
    return scorer(clf, X, y)