
import numpy as np
import SimpleITK as sitk


def flip_image3D(image):
    dimension = image.GetDimension()

    reference_origin = np.zeros(dimension)
    reference_size = image.GetSize()
    reference_spacing = image.GetSpacing()
    reference_direction = np.identity(dimension).flatten()

    reference_image = sitk.Image(reference_size, image.GetPixelIDValue())
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)

    reference_center = np.array(reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize())/2.0))

    transform = sitk.AffineTransform(dimension)
    transform.SetMatrix(image.GetDirection())
    transform.SetTranslation(np.array(image.GetOrigin()) - reference_origin)
    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(
        image.TransformContinuousIndexToPhysicalPoint(np.array(image.GetSize()) / 2.0)
    )
    centering_transform.SetOffset(
        np.array(transform.GetInverse().TransformPoint(img_center) - reference_center)
    )
    centered_transform = sitk.CompositeTransform([transform, centering_transform])

    flipped_transform = sitk.AffineTransform(dimension)
    flipped_transform.SetCenter(
        reference_image.TransformContinuousIndexToPhysicalPoint(
            np.array(reference_image.GetSize()) / 2.0
        )
    )

    flipped_transform.SetMatrix([1, 0, 0, 0, -1, 0, 0, 0, 1])
    centered_transform.AddTransform(flipped_transform)

    return sitk.Resample(image, reference_image, centered_transform, sitk.sitkLinear, 0.0)