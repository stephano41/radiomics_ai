from autorad.data import FeatureDataset as OrigFeatureDataset, TrainingData, TrainingInput, TrainingLabels, TrainingMeta


class FeatureDataset(OrigFeatureDataset):

    def load_splits(self, splits: dict):
        """
        Load training and test splits from a dictionary.

        `splits` dictionary should contain the following keys:
            - 'test': list of test IDs
            - 'train': dict with n keys (default n = 5)):
                - 'fold_{0..n-1}': list of training and
                                   list of validation IDs
            - 'split_on'; column name to split on,
                if None, split is performed on ID_colname
        It can be created using `full_split()` defined below.
        """
        self._splits = splits
        split_on = splits["split_on"]

        test_ids = splits["test"]
        test_rows = self.df[split_on].isin(test_ids)

        # Split dataframe rows
        X, y, meta = {}, {}, {}

        # Split the test data
        X["test"] = self.X.loc[test_rows]
        y["test"] = self.y.loc[test_rows]
        meta["test"] = self.meta_df.loc[test_rows]

        train_rows = ~self.df[split_on].isin(test_ids)

        # Split the validation data, if specified
        if "val" in splits:
            val_ids = splits["val"]
            val_rows = self.df[split_on].isin(val_ids)
            train_rows = train_rows & ~val_rows

            X["val"] = self.X.loc[val_rows]
            y["val"] = self.y.loc[val_rows]
            meta["val"] = self.meta_df.loc[val_rows]

        # Split the training data
        X["train"] = self.X.loc[train_rows]
        y["train"] = self.y.loc[train_rows]
        meta["train"] = self.meta_df.loc[train_rows]

        if isinstance(splits["train"], dict):
            train_splits = splits["train"]
            n_splits = len(train_splits)
            self.cv_splits = [
                train_splits[f"fold_{i}"] for i in range(n_splits)
            ]
            X["train_folds"], X["val_folds"] = [], []
            y["train_folds"], y["val_folds"] = [], []
            meta["train_folds"], meta["val_folds"] = [], []
            for fold in self.cv_splits:
                train_fold_ids = fold['train']
                val_fold_ids = fold['val']

                train_fold_rows = self.df[split_on].isin(train_fold_ids)
                val_fold_rows = self.df[split_on].isin(val_fold_ids)

                X["train_folds"].append(self.X[train_fold_rows])
                X["val_folds"].append(self.X[val_fold_rows])
                y["train_folds"].append(self.y[train_fold_rows])
                y["val_folds"].append(self.y[val_fold_rows])
                meta["train_folds"].append(self.meta_df[train_fold_rows])
                meta["val_folds"].append(self.meta_df[val_fold_rows])
        self._data = TrainingData(
            TrainingInput(**X), TrainingLabels(**y), TrainingMeta(**meta)
        )
        return self
