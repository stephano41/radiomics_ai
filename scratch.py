from sklearn.neighbors import KNeighborsClassifier

from src.evaluation.stratified_bootstrap import BootstrapGenerator
from src.evaluation._bootstrap import Bootstrap
import numpy as np

X = np.random.random([100,500])
Y = np.random.randint(0,2,100)
oob = BootstrapGenerator(n_splits=10)
splits = oob.split(X, Y)

model = KNeighborsClassifier(3)

evaluator = Bootstrap(X, Y, iters=20, num_cpu=1, log_dir='./outputs')
print(evaluator.run(model))