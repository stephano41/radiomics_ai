import os
import subprocess
from pathlib import Path

import SimpleITK as sitk
from dipy.viz import regtools
from matplotlib import pyplot as plt


class Register:
    def __init__(self, output_path: [str, Path], transform_method="com", out_affine=None):
        """

        :param output_path: (str or Path): The output path for the registered image or transformation.
        :param transform_method: (str, optional): The registration transform method. Default is
        "com". Options include "com" for center of mass-based affine registration or "sdr" for symmetric
        diffeomorphic. com: center of mass, trans: translation, rigid: rigid body, rigid_isoscaling: rigid body +
        isotropic scaling, rigid_scaling: rigid body + scaling, affine: full affine including translation, rotation,
        shearing and scaling
        :param out_affine (str, optional): The output filename for the estimated affine transformation parameters.
                If not provided, it will be set to the input filename followed by "_affine.txt".
        """
        if not isinstance(output_path, Path):
            output_path = Path(output_path)

        self.output_dir = output_path.parent
        self.output_name = output_path.name
        self.transform_method = transform_method

        if out_affine is None:
            self.out_affine = f"{output_path.name}_affine.txt"

    def transform(self, moving, static):
        if self.transform_method == 'sdr':
            subprocess.run(["dipy_align_syn", static, moving,
                            '--out_dir', self.output_dir,
                            '--out_warped', self.output_name])
        else:
            subprocess.run(["dipy_align_affine", static, moving,
                            '--transform', self.transform_method,
                            '--out_dir', self.output_dir,
                            '--out_moved', self.output_name,
                            '--out_affine', self.out_affine])


def move_plot_nii(static, moving, out_path, transform_method='com'):
    """
    Perform image registration using the specified transform method, and plot the results.

        This function takes a static (target) image and a moving (source) image, performs image registration
        using the specified transform method, and then plots the static, original moving, registered moving,
        and an overlay of the static and registered moving images.

        Parameters:
            static (str): The path to the static (target) image for registration.
            moving (str): The path to the moving (source) image for registration.
            out_path (str): The output path for the registered image or transformation.
            transform_method (str, optional): The registration transform method. Default is "com".
                Options include "com" for center of mass-based affine registration or "sdr" for symmetric diffeomorphic
                registration (SDR). Note that "sdr" requires dipy_align_syn to be installed separately.

        Returns:
            None

        Example:
            with tempfile.TemporaryDirectory() as tmpdirname:
                move_plot_nii("./data/meningioma_data/ID_1/t2.nii.gz",
                              "./data/meningioma_data/ID_1/ADC.nii",
                              os.path.join(tmpdirname, "affine_adc.nii.gz"),
                              transform_method='affine'
                              )
            plt.show()

        Note:
            The function uses the `Register` class to perform image registration, which internally uses either
            center of mass-based affine registration or symmetric diffeomorphic registration (SDR) depending on the
            specified `transform_method`.

            The static, original moving, and registered moving images are displayed as separate subplots in the first
            figure using matplotlib. Additionally, a second figure with an overlay of the static and registered moving
            images is also displayed. The overlay is created using SimpleITK's `overlay_images` function from `regtools`.

            To display the figures, call `plt.show()` after calling this function.
    """
    reg = Register(out_path, transform_method)
    reg.transform(moving, static)

    fig, axes = plt.subplots(1, 3, figsize=(20, 10))
    for i, img_path in enumerate([static, moving, out_path]):
        img = sitk.ReadImage(img_path)
        axes[i].imshow(sitk.GetArrayFromImage(img)[12, :, :], cmap='gray')
        axes[i].set_title(os.path.basename(img_path))
    fig.tight_layout()

    static_img = sitk.ReadImage(static)
    registered_moving_img = sitk.ReadImage(out_path)

    fig2 = regtools.overlay_images(sitk.GetArrayFromImage(static_img)[12, :, :],
                            sitk.GetArrayFromImage(registered_moving_img)[12, :, :],
                            title0='static', title1='moving')
    fig2.suptitle(f"{transform_method} static: {os.path.basename(static)}, moving: {os.path.basename(moving)}")
    fig2.tight_layout()
