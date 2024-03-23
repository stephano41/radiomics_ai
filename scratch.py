import torch
from src.models.autoencoder.fmcib_model import load_fmcib_model


model = load_fmcib_model(save_path='outputs/pretrained_models')

ip = torch.rand(1,1,96,96,96)