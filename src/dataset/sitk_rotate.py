import numpy as np
import SimpleITK as sitk
from typing import Union,Tuple

def eul2quat(ax, ay, az, atol=1e-8):
    """
    Translate between Euler angle (ZYX) order and quaternion representation of a rotation.
    Args:
        ax: X rotation angle in radians.
        ay: Y rotation angle in radians.
        az: Z rotation angle in radians.
        atol: tolerance used for stable quaternion computation (qs==0 within this tolerance).
    Return:
        Numpy array with three entries representing the vectoral component of the quaternion.

    """
    # Create rotation matrix using ZYX Euler angles and then compute quaternion using entries.
    cx = np.cos(ax)
    cy = np.cos(ay)
    cz = np.cos(az)
    sx = np.sin(ax)
    sy = np.sin(ay)
    sz = np.sin(az)
    r = np.zeros((3, 3))
    r[0, 0] = cz * cy
    r[0, 1] = cz * sy * sx - sz * cx
    r[0, 2] = cz * sy * cx + sz * sx

    r[1, 0] = sz * cy
    r[1, 1] = sz * sy * sx + cz * cx
    r[1, 2] = sz * sy * cx - cz * sx

    r[2, 0] = -sy
    r[2, 1] = cy * sx
    r[2, 2] = cy * cx

    # Compute quaternion:
    qs = 0.5 * np.sqrt(r[0, 0] + r[1, 1] + r[2, 2] + 1)
    qv = np.zeros(3)
    # If the scalar component of the quaternion is close to zero, we
    # compute the vector part using a numerically stable approach
    if np.isclose(qs, 0.0, atol):
        i = np.argmax([r[0, 0], r[1, 1], r[2, 2]])
        j = (i + 1) % 3
        k = (j + 1) % 3
        w = np.sqrt(r[i, i] - r[j, j] - r[k, k] + 1)
        qv[i] = 0.5 * w
        qv[j] = (r[i, j] + r[j, i]) / (2 * w)
        qv[k] = (r[i, k] + r[k, i]) / (2 * w)
    else:
        denom = 4 * qs
        qv[0] = (r[2, 1] - r[1, 2]) / denom
        qv[1] = (r[0, 2] - r[2, 0]) / denom
        qv[2] = (r[1, 0] - r[0, 1]) / denom
    return qv


def similarity3D_parameter_space_random_sampling(
        thetaX: Union[float, Tuple[float, float]],
        thetaY: Union[float, Tuple[float, float]],
        thetaZ: Union[float, Tuple[float, float]],
        tx: Union[float, Tuple[float, float]],
        ty: Union[float, Tuple[float, float]],
        tz: Union[float, Tuple[float, float]],
        scale: Union[float, Tuple[float, float]],
        n: int
):
    """
    Create a list representing a random (uniform) sampling of the 3D similarity transformation parameter space. As the
    SimpleITK rotation parametrization uses the vector portion of a versor we don't have an
    intuitive way of specifying rotations. We therefor use the ZYX Euler angle parametrization and convert to
    versor.
    Args:
        thetaX, thetaY, thetaZ: Ranges of Euler angle values to use, in degrees.
        tx, ty, tz: Ranges of translation values to use in mm.
        scale: Range of scale values to use.
        n: Number of samples.
    Return:
        List of lists representing the parameter space sampling (vx,vy,vz,tx,ty,tz,s).
    """
    thetaX = tuple(degrees_to_radians(i) for i in thetaX)
    thetaY = tuple(degrees_to_radians(i) for i in thetaY)
    thetaZ = tuple(degrees_to_radians(i) for i in thetaZ)

    theta_x_vals = (thetaX[1] - thetaX[0]) * np.random.random(n) + thetaX[0]
    theta_y_vals = (thetaY[1] - thetaY[0]) * np.random.random(n) + thetaY[0]
    theta_z_vals = (thetaZ[1] - thetaZ[0]) * np.random.random(n) + thetaZ[0]
    tx_vals = (tx[1] - tx[0]) * np.random.random(n) + tx[0]
    ty_vals = (ty[1] - ty[0]) * np.random.random(n) + ty[0]
    tz_vals = (tz[1] - tz[0]) * np.random.random(n) + tz[0]
    s_vals = (scale[1] - scale[0]) * np.random.random(n) + scale[0]
    res = list(
        zip(theta_x_vals, theta_y_vals, theta_z_vals, tx_vals, ty_vals, tz_vals, s_vals)
    )
    return [list(eul2quat(*(p[0:3]))) + list(p[3:7]) for p in res]


def degrees_to_radians(degrees):
    return degrees * np.pi / 180.0


def augment_images_spatial(
        original_image,
        reference_image,
        T0,
        T_aug,
        transformation_parameters,
        interpolator=sitk.sitkLinear,
        default_intensity_value=0.0,
):
    """
    Generate the resampled images based on the given transformations.
    Note: Images are written to disk with the useCompression flag
          set to true. This uses the default compression level for the user selected file
          format (via output_suffix).
    Args:
        original_image (SimpleITK image): The image which we will resample and transform.
        reference_image (SimpleITK image): The image onto which we will resample.
        T0 (SimpleITK transform): Transformation which maps points from the reference image coordinate system
            to the original_image coordinate system.
        T_aug (SimpleITK transform): Map points from the reference_image coordinate system back onto itself using the
               given transformation_parameters. The reason we use this transformation as a parameter
               is to allow the user to set its center of rotation to something other than zero.
        transformation_parameters (List of lists): parameter values which we use T_aug.SetParameters().
        output_prefix (string): output file name prefix (file name: output_prefix_p1_p2_..pn_.output_suffix).
        output_suffix (string): output file name suffix (file name: output_prefix_p1_p2_..pn_.output_suffix), also determines
                                the file formt usef for writing to disk.
        interpolator: One of the SimpleITK interpolators.
        default_intensity_value: The value to return if a point is mapped outside the original_image domain.
        additional_image_information: (Iterable([SimpleITK.Image, Interpolator, default_intensity_value])): Apply the same
                                     transformations to this set of images using the given interpolators and intensity values.
                                     The iterable cannot be a zip because that will not allow for repeated iterations.
    """
    all_images = []  # Used only for display purposes in this notebook.
    for current_parameters in transformation_parameters:
        T_aug.SetParameters(current_parameters)
        # Augmentation is done in the reference image space, so we first map the points from the reference image space
        # back onto itself T_aug (e.g. rotate the reference image) and then we map to the original image space T0.
        T_all = sitk.CompositeTransform(T0)
        T_all.AddTransform(T_aug)
        aug_image = sitk.Resample(
            original_image,
            reference_image,
            T_all,
            interpolator,
            default_intensity_value,
        )

        all_images.append(aug_image)  # Used only for display purposes in this notebook.
    return all_images  # Used only for display purposes in this notebook.


def sitk_transform3D(image, aug_transform=None, thetaX=(0, 0), thetaY=(0, 0), thetaZ=(0, 0), tx=(0, 0), ty=(0, 0),
                     tz=(0, 0), scale=(1, 1), n=2):
    if aug_transform is None:
        aug_transform = sitk.Similarity3DTransform()

    dimension = image.GetDimension()
    reference_origin = np.zeros(dimension)
    reference_size = image.GetSize()
    reference_spacing = image.GetSpacing()
    reference_direction = np.identity(dimension).flatten()

    reference_image = sitk.Image(reference_size, image.GetPixelIDValue())
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)

    reference_center = np.array(
        reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize()) / 2.0))

    transform = sitk.AffineTransform(dimension)
    transform.SetMatrix(image.GetDirection())
    transform.SetTranslation(np.array(image.GetOrigin()) - reference_origin)
    # Modify the transformation to align the centers of the original and reference image instead of their origins.
    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(
        image.TransformContinuousIndexToPhysicalPoint(np.array(image.GetSize()) / 2.0)
    )
    centering_transform.SetOffset(
        np.array(transform.GetInverse().TransformPoint(img_center) - reference_center)
    )
    centered_transform = sitk.CompositeTransform(transform)
    centered_transform.AddTransform(centering_transform)

    # Set the augmenting transform's center so that rotation is around the image center.
    aug_transform.SetCenter(reference_center)

    transformation_parameters_list = similarity3D_parameter_space_random_sampling(
        thetaX=thetaX,
        thetaY=thetaY,
        thetaZ=thetaZ,
        tx=tx,
        ty=ty,
        tz=tz,
        scale=scale,
        n=n)

    generated_images = augment_images_spatial(
        image,
        reference_image,
        centered_transform,
        aug_transform,
        transformation_parameters_list,
    )

    return generated_images
