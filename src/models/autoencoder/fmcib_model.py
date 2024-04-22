from monai.networks.nets import resnet50
import requests
from pathlib import Path
from torch import nn
import torch
import logging
from collections import OrderedDict
from ..utils import expand_weights
from torch.optim import AdamW

logger = logging.getLogger(__name__)


class FMCIBModel(nn.Module):
    def __init__(self, trunk=None, weights_path=None, input_channels=1, output_class=1, latent_var_dim=2048):
        super().__init__()
        if trunk is None:
            trunk = resnet50(pretrained=False,
                     n_input_channels=input_channels,
                     widen_factor=2,
                    #  conv1_t_stride=2,
                     feed_forward=False,
                    #  bias_downsample=True,
                     )
            
        self.trunk = trunk
        self.latent_var_head = nn.Linear(4096, latent_var_dim, bias=True)
        self.heads=nn.Sequential(OrderedDict([('heads_relu', nn.ReLU(inplace=True)),
                                              ('heads1', nn.Linear(latent_var_dim, output_class, bias=True))
                                              ]))

        if weights_path is not None:
            self.load(weights_path)
        
    def forward(self, x:torch.Tensor):
        out = self.trunk(x)
        out = self.latent_var_head(out)
        out = self.heads(out)
        return out
    
    def load(self, weights):
        trained_trunk = torch.load(weights)['trunk_state_dict']

        loaded_weights = {}
        for k,v in self.state_dict().items():
            key = k.removeprefix('trunk.')
            if key in trained_trunk.keys():
                loaded_weights[key] = expand_weights(trained_trunk[key], v)
        
        msg = self.trunk.load_state_dict(loaded_weights, strict=False)
        logger.warning(f"Missing keys: {msg[0]} and unexpected keys: {msg[1]}")

    def generate_latent_vars(self, x:torch.tensor):
        self.eval()
        out = self.trunk(x)
        out = self.latent_var_head(out)
        return out

    


def load_fmcib_model(eval_mode=True, save_path='outputs', **kwargs):
    weights_url = "https://zenodo.org/records/10528450/files/model_weights.torch?download=1"
    weights_path = Path(save_path) / "model_weights.torch"
    if not weights_path.exists():
        r=requests.get(weights_url)
        open(weights_path, 'wb').write(r.content)
    model = FMCIBModel(weights_path=weights_path, **kwargs)

    if eval_mode:
        model.eval()

    return model

