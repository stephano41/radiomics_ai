import numpy as np
from skorch.net import NeuralNet

def get_total_params(skorch_net:NeuralNet):
    if not skorch_net.initialized_:
        skorch_net.initialize()
    parameter_count = []
    for parameter_set in skorch_net.get_all_learnable_params():
        parameter_count.append(np.prod(parameter_set[1].shape))
    
    return np.sum(parameter_count)