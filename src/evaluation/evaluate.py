import logging

import pandas as pd

from autorad.data import FeatureDataset
from autorad.models import MLClassifier
from autorad.preprocessing import Preprocessor

log = logging.getLogger(__name__)


def evaluate_feature_dataset(
    dataset: FeatureDataset,
    model: MLClassifier,
    preprocessor: Preprocessor,
    split: str = "test",
) -> pd.DataFrame:
    """
    Evaluate a feature dataset using a model and a preprocessor.
    """
    X_preprocessed = preprocessor.transform_df(getattr(dataset.data.X, split))
    y_pred_proba = model.predict_proba(X_preprocessed)
    y_true = getattr(dataset.data.y, split)

    result = pd.DataFrame(
        {
            "ID": y_true.index,
            "y_true": y_true,
            "y_pred_proba": y_pred_proba,
        }
    ).reset_index(drop=True)

    return result