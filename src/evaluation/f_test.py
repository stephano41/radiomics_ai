from functools import partial

import yaml
from autorad.data import TrainingData, FeatureDataset
import scipy.stats
import tempfile
import os
from autorad.metrics import roc_auc
from tqdm import tqdm


def combined_ftest_5x2cv(estimator1, estimator2, feature_dataset1: FeatureDataset, feature_dataset2: FeatureDataset,
                         multi_class='raise', labels=None, average='macro', save_path=None):
    if save_path is None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            _save_path = os.path.join(tmp_dir + 'split.yml')
            feature_dataset1.split(method="repeated_stratified_kfold_no_test", n_splits=2, n_repeats=5,
                                   save_path=_save_path)
            with open(_save_path, 'r') as f:
                feature_dataset2.load_splits(yaml.safe_load(f))
    else:
        feature_dataset1.split(method="repeated_stratified_kfold_no_test", n_splits=2, n_repeats=5,
                               save_path=save_path)
        with open(save_path, 'r') as f:
            feature_dataset2.load_splits(yaml.safe_load(f))

    data1: TrainingData = feature_dataset1.data
    data2: TrainingData = feature_dataset2.data

    scorer = partial(roc_auc, average=average, multi_class=multi_class, labels=labels)

    assert len(data1.X.train) == len(data2.X.train)
    assert len(data1.X.train_folds) == len(data2.X.train_folds)
    assert len(data1.X.train_folds) % 2 == 0, "uneven number of folds"

    variances = []
    differences = []

    score_diffs = []
    for i in tqdm(range(len(data1.X.train_folds))):
        estimator1.fit(data1.X.train_folds[i], data1.y.train_folds[i])
        est1_score = scorer(data1.y.val_folds[i], estimator1.predict_proba(data1.X.val_folds[i]))

        estimator2.fit(data2.X.train_folds[i], data2.y.train_folds[i])
        est2_score = scorer(data2.y.val_folds[i], estimator2.predict_proba(data2.X.val_folds[i]))

        score_diffs.append(est1_score - est2_score)
        if i % 2 == 1 and i > 0:
            score_mean = (score_diffs[i - 1] + score_diffs[i]) / 2.0
            score_var = (score_diffs[i - 1] - score_mean) ** 2 + (score_diffs[i] - score_mean) ** 2
            differences.extend([score_diffs[i - 1] ** 2, score_diffs[i] ** 2])
            variances.append(score_var)

    numerator = sum(differences)
    denominator = 2 * (sum(variances))
    f_stat = numerator / denominator

    p_value = scipy.stats.f.sf(f_stat, 10, 5)

    return float(f_stat), float(p_value)
