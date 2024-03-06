import pandas as pd
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
from autorad.data import FeatureDataset

dataset = FeatureDataset(pd.read_csv('tests/meningioma_feature_dataset.csv'), target='Grade', ID_colname='ID')

f_stat, p_value = f_classif(dataset.X, dataset.y)

indices = np.where(p_value < 0.05)[0]
selected_features = dataset.X.columns[indices].to_list()

