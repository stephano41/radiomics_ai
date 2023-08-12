import pickle

import pandas as pd
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
from pqdm.processes import pqdm
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

        self.data_preprocessor = MRIDataPreprocessor(dataset, **settings)

    def run(self):
        data_x, _ = self.data_preprocessor.preprocess()
        self.model.fit(data_x, data_x)

        features = self.model.predict(data_x)

        return features



class MRIDataPreprocessor:
    def __init__(self, dataset: ImageDataset, **settings):
        self.dataset = dataset

        self.label = settings.get('label', 1),
        self.label_channel = settings.get('label_channel', 0),
        self.minimumROIDimensions = settings.get('minimumROIDimensions', 1),
        self.minimumROISize = settings.get('minimumROISize', 0),
        self.interpolator = settings.get('interpolator', sitk.sitkBSpline),
        self.padDistance = settings.get('padDistance', 5)

    def load_nifti(self, image_filename, mask_filename):
        image = sitk.ReadImage(image_filename)
        mask = imageoperations.getMask(sitk.ReadImage(mask_filename), label=self.label,
                                       label_channel=self.label_channel)
        return image, mask

    def crop_to_mask(self, image, mask):
        boundingBox, correctedMask = imageoperations.checkMask(image, mask,
                                                               minimumROIDimensions=self.minimumROIDimensions,
                                                               minimumROISize=self.minimumROISize)
        if correctedMask is not None:
            mask = correctedMask

        if boundingBox is None:
            raise ValueError('Mask checks failed during pre-crop')
        cropped_image, cropped_mask = imageoperations.cropToTumorMask(image, mask, boundingBox)
        return cropped_image, cropped_mask

    def resample_image_mask(self, image, mask, new_size):
        new_spacing = [curr * orig / new for curr, orig, new in
                       zip(image.GetSpacing(), image.GetSize(), new_size)]

        bb = _checkROI(image, mask, label=self.label)

        direction = np.array(mask.GetDirection())

        maskSpacing = np.array(mask.GetSpacing())
        Nd_mask = len(maskSpacing)
        maskSize = np.array(mask.GetSize())

        spacingRatio = maskSpacing / new_spacing

        # Determine bounds of cropped volume in terms of new Index coordinate space,
        # round down for lowerbound and up for upperbound to ensure entire segmentation is captured (prevent data loss)
        # Pad with an extra .5 to prevent data loss in case of upsampling. For Ubound this is (-1 + 0.5 = -0.5)
        bbNewLBound = np.floor((bb[:Nd_mask] - 0.5) * spacingRatio - self.padDistance)
        bbNewUBound = np.ceil((bb[:Nd_mask] + bb[Nd_mask:] - 0.5) * spacingRatio + self.padDistance)

        # Ensure resampling is not performed outside bounds of original image
        maxUbound = np.ceil(maskSize * spacingRatio) - 1
        bbNewLBound = np.where(bbNewLBound < 0, 0, bbNewLBound)
        # bbNewUBound = np.where(bbNewUBound > maxUbound, maxUbound, bbNewUBound)

        bbOriginalLBound = bbNewLBound / spacingRatio
        newOriginIndex = np.array(.5 * (new_spacing - maskSpacing) / maskSpacing)
        newCroppedOriginIndex = newOriginIndex + bbOriginalLBound
        newOrigin = mask.TransformContinuousIndexToPhysicalPoint(newCroppedOriginIndex)

        imagePixelType = image.GetPixelID()
        maskPixelType = mask.GetPixelID()

        rif = sitk.ResampleImageFilter()

        rif.SetOutputSpacing(new_spacing)
        rif.SetOutputDirection(direction)
        rif.SetSize(new_size)
        rif.SetOutputOrigin(newOrigin)

        rif.SetOutputPixelType(imagePixelType)
        rif.SetInterpolator(self.interpolator)
        resampledImageNode = rif.Execute(image)

        rif.SetOutputPixelType(maskPixelType)
        rif.SetInterpolator(sitk.sitkNearestNeighbor)
        resampledMaskNode = rif.Execute(mask)

        return resampledImageNode, resampledMaskNode

    def preprocess(self) -> (np.ndarray, np.ndarray):
        images = []
        masks = []

        # Load images and masks, crop, resample, and convert to numpy arrays
        smallest_size = None
        for image_path, mask_path in tqdm(zip(self.dataset.image_paths, self.dataset.mask_paths),
                                          total=len(self.dataset.image_paths),
                                          desc="Processing MRI Data"):
            image, mask = self.load_nifti(image_path, mask_path)

            cropped_image, cropped_mask = self.crop_to_mask(image, mask)

            if smallest_size is None:
                smallest_size = cropped_image.GetSize()
            else:
                smallest_size = [min(s, c) for s, c in zip(smallest_size, cropped_image.GetSize())]

            images.append(cropped_image)
            masks.append(cropped_mask)

        for i in tqdm(range(len(images)), desc="Resampling and Converting to Numpy"):
            # new_spacing = [curr * orig / new for curr, orig, new in
            #                zip(images[i].GetSpacing(), images[i].GetSize(), smallest_size)]
            #
            # resampled_image, resampled_mask = imageoperations.resampleImage(images[i], masks[i],
            #                                                                 resampledPixelSpacing=new_spacing,
            #                                                                 interpolator=self.interpolator,
            #                                                                 padDistance=self.padDistance,
            #                                                                 label=self.label)
            resampled_image, resampled_mask = self.resample_image_mask(images[i], masks[i], smallest_size)

            assert np.equal(resampled_image.GetSize(),
                            smallest_size).all(), f"resampled image {resampled_image.GetSize()}, whereas smallest_size: {smallest_size}"

            images[i] = sitk.GetArrayFromImage(resampled_image)
            masks[i] = sitk.GetArrayFromImage(resampled_mask)

        return np.array(images), np.array(masks)


# data_preprocessor = MRIDataPreprocessor(dataset, interpolator=sitk.sitkNearestNeighbor)
# processed_data = data_preprocessor.preprocess()

#
# with open('./outputs/processed_data.pkl', 'wb') as f:
#     pickle.dump((images, masks), f)
# with open('./outputs/processed_data.pkl', 'rb') as f:
#     images, _ = pickle.load(f)

from sklearn.datasets import load_digits
import torch

data = load_digits(n_class=2)

images, target = data['images'], data['target']

images = torch.unsqueeze(torch.tensor(images).type(torch.float32), dim=1)

from skorch.callbacks import PassthroughScoring, PrintLog, EarlyStopping
from src.models.autoencoder import VanillaVAE
from src.models.encoder import Encoder, VAELoss

encoder = Encoder(VanillaVAE,
                  module__in_channels=1,
                  module__latent_dim=100,
                  module__hidden_dims= [32, 64],
                  criterion=VAELoss,
                  callbacks=[
                    ('early_stop', EarlyStopping(
                        monitor='valid_loss',
                        patience=5
                    ))]
                  )

encoder.fit(images)