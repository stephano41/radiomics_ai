from autorad.data import FeatureDataset as OrigFeatureDataset
import logging
from pathlib import Path
from typing import Optional

from autorad.config import config
from autorad.config.type_definitions import PathLike
from autorad.utils import io, splitting
import pandas as pd

log = logging.getLogger(__name__)


class FeatureDataset(OrigFeatureDataset):
    def __init__(self,
                 dataframe: pd.DataFrame,
                 target: str,
                 ID_colname: str,
                 features: Optional[list[str]] = None,
                 additional_features=[],
                 meta_columns: list[str] = [],
                 random_state: int = config.SEED,
                 ):
        self.additional_features=additional_features

        super().__init__(dataframe=dataframe,
                         target=target,
                         ID_colname=ID_colname,
                         features=features,
                         meta_columns=meta_columns,
                         random_state=random_state
                         )

    def _init_features(
        self, features: Optional[list[str]] = None
    ) -> list[str]:
        return self.additional_features + super()._init_features(features)

    def split(
            self,
            save_path: Optional[PathLike] = None,
            method="train_with_cross_validation_test",
            split_on: Optional[str] = None,
            test_size: float = 0.2,
            *args,
            **kwargs,
    ) -> dict:
        if split_on is None:
            split_on = self.ID_colname
        if method == "train_with_cross_validation_test":
            splits = self.full_split(split_on, test_size, *args, **kwargs)
        elif method == "train_val_test":
            splits = self.split_train_val_test(
                split_on, test_size, *args, **kwargs
            )
        elif method == "train_with_cross_validation":
            splits = self.full_split_no_test(split_on, test_size, *args, **kwargs)
        else:
            raise ValueError(f"Method {method} is not supported.")

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            io.save_yaml(splits, save_path)
        self.load_splits(splits)
        return splits

    def full_split_no_test(self, split_on, test_size, n_splits: int = 5) -> dict:
        patient_df = self.df[[split_on, self.target]].drop_duplicates()
        if not patient_df[split_on].is_unique:
            raise ValueError(
                f"Selected column {split_on} has varying labels for the same ID!"
            )
        ids = patient_df[split_on].tolist()
        labels = patient_df[self.target].tolist()
        splits = splitting.split_cross_validation(
            ids_train=ids,
            y_train=labels,
            n_splits=n_splits,
            cv_type="stratified_kfold"
        )
        results={
            "split_on": split_on,
            "split_type": f"{n_splits} fold stratified_kfold cross validation on training set only",
            "test": [], # add something to test to stop code from breaking, not intended to be used
            "train": splits
        }

        return results
