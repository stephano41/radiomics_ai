import numpy as np
from skorch.net import NeuralNet

def get_total_params(skorch_net:NeuralNet):
    if not skorch_net.initialized_:
        skorch_net.initialize()
    parameter_count = []
    for parameter_set in skorch_net.get_all_learnable_params():
        parameter_count.append(np.prod(parameter_set[1].shape))
    
    return np.sum(parameter_count)


def expand_weights(original, new):
    if len(original.shape)<=1 or len(new.shape)<=1:
        return original
    original_channels = original.shape[1]
    target_channels = new.shape[1]

    if target_channels%original_channels==0 and original_channels<target_channels:
        repeat_shape = [1] * len(new.shape)
        repeat_shape[1] = int(target_channels/original_channels)
        return original.repeat(*repeat_shape)
    return original