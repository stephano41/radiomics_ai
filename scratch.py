from functools import partial

import numpy as np
from sklearn.metrics import roc_auc_score

# import six
# from matplotlib import pyplot as plt
#
# from src.dataset import WikiSarcoma
#
# import os
# import pydicom
# from dcmrtstruct2nii import dcmrtstruct2nii, list_rt_structs
# import pytest
# import SimpleITK as sitk
# from radiomics import featureextractor

y_val = np.asarray([0, 1, 0, 1, 0, 1, 0])
y_pred = np.asarray([[0.33333334, 0.33333334, 0.33333334,], [0.33333334, 0.33333334, 0.33333334,], [0.33333334, 0.33333334, 0.33333334,], [0.33333334, 0.33333334, 0.33333334,], [0.33333334, 0.33333334, 0.33333334,], [0.33333334, 0.33333334, 0.33333334,], [0.33333334, 0.33333334, 0.33333334,]])

auc_scorer = partial(roc_auc_score, multi_class='ovo', labels=[0,1,2])

print(auc_scorer(y_val, y_pred))