from collections import defaultdict

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
from autorad.data import FeatureDataset
import pingouin as pg
from autorad.feature_selection.selector import AnovaSelector, CoreSelector
from sklearn.feature_selection import RFECV, SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression


class SFSelector(AnovaSelector):
    def __init__(self, direction='backward', scoring='roc_auc', n_jobs=None, ):
        self.direction=direction
        self.scoring=scoring
        self.model = SequentialFeatureSelector(LogisticRegression(), direction=direction, scoring=scoring, n_jobs=5, n_features_to_select='auto', tol=0.05)
        super().__init__()

    def fit(self, X, y):
        indices = self.run_anova(X, y, True)
        _X = X.iloc[:, indices]

        self.model.fit(_X,y)
        support = self.model.get_support(indices=True)
        if support is None:
            raise ValueError("SFSelector failed to select features")
        selected_columns = support.tolist()
        self._selected_features=_X.columns[selected_columns].tolist()


class RFESelector(AnovaSelector):
    def __init__(self, min_features=2, scoring='roc_auc'):
        self.min_features=min_features
        self.scoring=scoring
        self.model = RFECV(LogisticRegression(), min_features_to_select=min_features, scoring=scoring, verbose=1, n_jobs=5)
        super().__init__()

    def fit(self, X: pd.DataFrame, y: pd.Series):
        indices = self.run_anova(X, y, True)
        _X = X.iloc[:, indices]

        self.model.fit(_X,y)
        support = self.model.get_support(indices=True)
        if support is None:
            raise ValueError("RFESelector failed to select features")
        selected_columns = support.tolist()
        self._selected_features=_X.columns[selected_columns].tolist()


class ICCSelector(CoreSelector):
    def __init__(self, icc_type='ICC2', icc_threshold=0.6):
        self.icc_type = icc_type
        self.icc_threshold = icc_threshold
        super().__init__()

    def fit(self, X, y):
        iccs = get_feature_icc(X, y, self.icc_type)
        selected_columns = np.where(iccs['ICC'].to_numpy()>self.icc_threshold)[0]
        self._selected_features = X.columns[selected_columns].to_list()

def get_feature_icc(X: pd.DataFrame, y: pd.Series, icc_type='ICC2'):
    #judges are y
    #wine is feature names
    #score is the value of the feature name

    icc_results = []
    for feature_name in X.columns:
        feature_name_col = generate_feature_name_col(y)
        combined_df = pd.concat([feature_name_col, y, X[feature_name]], axis=1)
        icc_result = pg.intraclass_corr(data=combined_df, targets='Feature Name',raters=y.name, ratings=feature_name, nan_policy='omit')
        icc_result = icc_result.set_index('Type')
        icc_results.append({'Feature Name': feature_name,'ICC':icc_result.loc[icc_type, 'ICC']})
    return pd.DataFrame(icc_results)

def generate_feature_name_col(y):
    seen_counts = defaultdict(int)
    feature_names=[]
    for value in y.to_list():
        feature_names.append(seen_counts[value])
        seen_counts[value] +=1

    return pd.Series(feature_names, name='Feature Name')






dataset = FeatureDataset(pd.read_csv('tests/meningioma_feature_dataset.csv'), target='Grade', ID_colname='ID')

# iccs = get_feature_icc(dataset.X, dataset.y)
import time
start = time.time()
sfselector = SFSelector()
sfselector.fit(dataset.X, dataset.y)
print(len(sfselector.selected_features))
print(time.time() - start)

# start = time.time()

# sfselector = RFESelector()
# sfselector.fit(dataset.X, dataset.y)
# print(sfselector.selected_features)
# print(time.time() - start)


# icc_results = []
# for feature_name in dataset.X.columns:
#     feature_name_col = pd.Series([f"feature_name_{i}" for i in range(len(dataset.y))], name='Feature Name')
#     combined_df = pd.concat([feature_name_col, dataset.y, dataset.X[feature_name]], axis=1)
#     icc_results.append(
#         pg.intraclass_corr(data=combined_df, targets='Feature Name', raters='Grade', ratings=feature_name, nan_policy='omit'))
#     print(icc_results)
#     break
# f_stat, p_value = f_classif(dataset.X, dataset.y)
#
#
#
# indices = np.where(p_value < 0.01)[0]
# selected_features = dataset.X.columns[indices].to_list()

