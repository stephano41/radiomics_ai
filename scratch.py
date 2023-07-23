import os

from src.evaluation.stratified_bootstrap import BootstrapGenerator

stratifiedbootstrap = BootstrapGenerator()
import numpy as np
X = np.random.random([50,10])
y=np.random.randint(0,3, 50)

for train_idx, test_idx in stratifiedbootstrap.split(X, y):
    print(y[train_idx])
    print(y[test_idx])
    break