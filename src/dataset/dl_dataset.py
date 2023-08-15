from ._image_dataset import ImageDataset
import SimpleITK as sitk
from radiomics import imageoperations
from radiomics.imageoperations import _checkROI
import numpy as np
from tqdm import tqdm
from pqdm.processes import pqdm
import matplotlib.pyplot as plt

from src.preprocessing import sitk_transform3D, flip_image3D


class DLDataset:
    def __init__(self, dataset: ImageDataset, n_jobs=2, transform_kwargs=None, **settings):
        """
        Initialize the DLDataset.

        Parameters:
            dataset (ImageDataset): The input dataset containing image and mask paths.
            n_jobs (int): Number of parallel jobs for processing.
            transform_kwargs (dict): Keyword arguments for data augmentation transformations.
            settings (dict): Other settings including label, label_channel, interpolator, and padDistance.
        """
        self.dataset = dataset
        self.n_jobs = n_jobs

        self._settings = settings.copy()

        self.label = settings.get('label', 1)
        self.label_channel = settings.get('label_channel', 0)
        self.interpolator = settings.get('interpolator', sitk.sitkBSpline)
        self.padDistance = settings.get('padDistance', 5)

        self.transform_kwargs = transform_kwargs

        if transform_kwargs is None:
            self.transform_kwargs = dict(aug_transform=sitk.Similarity3DTransform(),
                                         thetaX=(0, 0),
                                         thetaY=(0, 0),
                                         thetaZ=(-np.pi / 2, np.pi / 2),
                                         tx=(0, 0),
                                         ty=(0, 0),
                                         tz=(0, 0),
                                         scale=(1, 1),
                                         n=2)

    def load_nifti(self, image_filename, mask_filename):
        image = sitk.ReadImage(image_filename)
        mask = imageoperations.getMask(sitk.ReadImage(mask_filename), label=self.label,
                                       label_channel=self.label_channel)
        return image, mask

    def crop_to_mask(self, image, mask):
        boundingBox, correctedMask = imageoperations.checkMask(image, mask,
                                                               **self._settings)
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

        spacingRatio = maskSpacing / new_spacing

        # Determine bounds of cropped volume in terms of new Index coordinate space,
        # round down for lowerbound and up for upperbound to ensure entire segmentation is captured (prevent data loss)
        # Pad with an extra .5 to prevent data loss in case of upsampling. For Ubound this is (-1 + 0.5 = -0.5)
        bbNewLBound = np.floor((bb[:Nd_mask] - 0.5) * spacingRatio - self.padDistance)

        # Ensure resampling is not performed outside bounds of original image
        bbNewLBound = np.where(bbNewLBound < 0, 0, bbNewLBound)

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

    def preprocess(self) -> np.ndarray:
        results = pqdm(
            ({"image_path": val[0],
              "mask_path": val[1]}
             for val in zip(self.dataset.image_paths, self.dataset.mask_paths)),
            self.read_and_crop,
            n_jobs=self.n_jobs,
            argument_type='kwargs'
        )

        images = [r[0] for r in results]
        masks = [r[1] for r in results]

        sizes = np.array([img.GetSize() for img in images])

        # adjust smallest_size to be a square otherwise conv layers won't like it
        smallest_size = [int(np.min(sizes)) for _ in range(sizes.shape[1])]

        generated_images = sum(pqdm(
            ({"image": val[0],
              "mask": val[1],
              "target_size": smallest_size}
             for val in zip(images, masks)),
            self.resample_and_augment,
            n_jobs=self.n_jobs,
            argument_type='kwargs'
        ), [])

        return np.array(generated_images)

    def read_and_crop(self, image_path, mask_path):
        image, mask = self.load_nifti(image_path, mask_path)

        cropped_image, cropped_mask = self.crop_to_mask(image, mask)
        return cropped_image, cropped_mask

    def resample_and_augment(self, image, mask, target_size):
        resampled_image, _ = self.resample_image_mask(image, mask, target_size)

        generated_images = []

        assert np.equal(resampled_image.GetSize(),
                        target_size).all(), f"resampled image {resampled_image.GetSize()}, whereas smallest_size: {target_size}"

        generated_images.append(resampled_image)

        generated_images.append(flip_image3D(resampled_image))
        generated_images.extend(sitk_transform3D(resampled_image, **self.transform_kwargs))

        return [sitk.GetArrayFromImage(img) for img in generated_images]