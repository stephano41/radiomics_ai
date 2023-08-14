from ._image_dataset import ImageDataset
import SimpleITK as sitk
from radiomics import imageoperations
from radiomics.imageoperations import _checkROI
import numpy as np
from tqdm import tqdm

from src.preprocessing import sitk_transform3D, flip_image3D


class DLDataset:
    def __init__(self, dataset: ImageDataset, transform_kwargs=None, **settings):
        self.dataset = dataset

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

    def augment_image(self, image, mask):
        # num_rotations = random.randint(0,3)
        # TODO augment images
        pass

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

        # adjust smallest_size to be a square otherwise conv layers won't like it
        smallest_size = [min(smallest_size) for _ in range(len(smallest_size))]

        generated_images = []
        for i in tqdm(range(len(images)), desc="Resampling and Converting to Numpy"):
            resampled_image, _ = self.resample_image_mask(images[i], masks[i], smallest_size)

            assert np.equal(resampled_image.GetSize(),
                            smallest_size).all(), f"resampled image {resampled_image.GetSize()}, whereas smallest_size: {smallest_size}"

            generated_images.append(resampled_image)

            generated_images.append(flip_image3D(resampled_image))
            generated_images.extend(sitk_transform3D(resampled_image, **self.transform_kwargs))

        return np.array([sitk.GetArrayFromImage(img) for img in generated_images])
