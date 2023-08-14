import pickle

import pandas as pd
import torch
import yaml
from radiomics import imageoperations
from radiomics.imageoperations import _checkROI
from skorch import NeuralNetClassifier
from skorch.callbacks import PassthroughScoring, PrintLog, EarlyStopping
from skorch.dataset import ValidSplit
from tqdm import tqdm

from src.dataset import ImageDataset
from src.pipeline.pipeline_components import get_data
import SimpleITK as sitk
import matplotlib.pyplot as plt
from src.dataset import DLDataset
import numpy as np


def plot_debug(stk_image):
    plt.figure()
    plt.imshow(sitk.GetArrayFromImage(stk_image)[5, :, :], cmap='gray')
    plt.show()


dataset = get_data('./data/meningioma_data', 't1ce', 'mask')


class DeepFeatureExtractor:
    def __init__(self, dataset: ImageDataset, autoencoder=None, extraction_params="CT_Baessler.yaml",
                 classifier_kwargs=None):
        if classifier_kwargs is None:
            classifier_kwargs = {
                "max_epochs": 10,
                "callbacks":[
                    ('train_loss', PassthroughScoring(
                        name='train_loss',
                        on_train=True,
                    )),
                    ('valid_loss', PassthroughScoring(
                        name='valid_loss',
                    )),
                    ('print_log', PrintLog()),
                    ('early_stop', EarlyStopping(
                        monitor='valid_loss',
                        patience=5
                    ))
                ],
                "train_split": ValidSplit(5)
            }

        self.model = NeuralNetClassifier(autoencoder, **classifier_kwargs)

        with open(extraction_params, 'r') as yaml_file:
            settings = yaml.safe_load(yaml_file)['settings']

        self.data_preprocessor = DLDataset(dataset, **settings)

    def run(self):
        data_x, _ = self.data_preprocessor.preprocess()
        self.model.fit(data_x, data_x)

        features = self.model.predict(data_x)

        return features




data_preprocessor = DLDataset(dataset)
images= data_preprocessor.preprocess()
#
#
with open('./outputs/processed_data.pkl', 'wb') as f:
    pickle.dump((images), f)
# with open('./outputs/processed_data.pkl', 'rb') as f:
#     images, _ = pickle.load(f)

# images = np.expand_dims(images, axis=1)
#
#
# from skorch.callbacks import EarlyStopping
# from sklearn.pipeline import Pipeline
# from src.models.autoencoder import VanillaVAE, VAELoss, Encoder
#
# encoder = Encoder(VanillaVAE,
#                   module__in_channels=1,
#                   module__latent_dim=100,
#                   module__hidden_dims= [16, 32],
#                   module__finish_size=3,
#                   criterion=VAELoss,
#                   std_dim=(0,2,3,4),
#                   max_epochs=10,
#                   # callbacks=[
#                   #   ('early_stop', EarlyStopping(
#                   #       monitor='valid_loss',
#                   #       patience=5
#                   #   ))]
#                   )
#
# encoder.fit(images)