import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from src.models.autoencoder.med3d_resnet import med3d_resnet10
import torch
from src.models.autoencoder import NeuralNetEncoder
from pytest import mark
import numpy as np

def test_nn_encoder():
    # generate random images batch 10, single channel, 16x16x16
    images = torch.randn([10,1,16,16,16]).type(torch.float32)

    encoder = NeuralNetEncoder(module='src.models.autoencoder.VanillaVAE',
                               module__in_channels=1,
                               module__latent_dim=64,
                               module__hidden_dims=[8,16,32],
                               module__finish_size=2,
                               criterion='src.models.autoencoder.VAELoss',
                               max_epochs=2
                               )

    encoder.fit(images)

    encoder.transform(images)

@mark.slow
@mark.parametrize('cfg_tune', [f'experiments={e}' for e in ['meningioma','meningioma_autoencoder','wiki_sarcoma']], indirect=True)
def test_encoder_instantiation(cfg_tune):
    if not cfg_tune.get('preprocessing', {}).get('autoencoder', False):
        return

    HydraConfig().set_config(cfg_tune)
    encoder = instantiate(cfg_tune.preprocessing.autoencoder, _convert_='object')

    encoder.fit([f'ID_{i+1}' for i in range(10)], np.random.randint(2, size=10))


@mark.parametrize('model', [med3d_resnet10(input_image_size=[96,96,96], shortcut_type='B', in_channels=5)], indirect=True)
def test_encoder_model(model):
    device = torch.device('cuda:0')
    model.to(device)
    test_input = torch.randn(2,5,96,96,96).to(device)
    print("model and data loaded")
    try:
        out = model(test_input)
        print(out[0].shape)

        # print(model(test_input)[0].shape)
    finally:
        model.to(torch.device('cpu'))
        model=None
        torch.cuda.empty_cache()
    
    assert list(out[0].shape) == [2,5,96,96,96]