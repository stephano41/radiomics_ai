import os
import subprocess
from pathlib import Path


def register_patients(data_dir, static_stem, moving_stem, output_stem='warped.nii', transform_method='rigid', id_regex="ID_*"):
    """
    Register multiple patients' MRI images using the specified transform method.

    Parameters:
        data_dir (str or Path): Path to the directory containing patient folders, each containing the MRI images.
        static_stem (str): Stem of the static (reference) MRI image filename within each patient folder.
        moving_stem (str): Stem of the moving (image to be registered) MRI image filename within each patient folder.
        output_stem (str, optional): Stem of the output filename for the registered image. Default is 'warped.nii'.
        transform_method (str, optional): The registration transform method. Default is 'rigid'.
            Options include:
                'com': Center of mass-based affine registration.
                'sdr': Symmetric diffeomorphic registration.
                'trans': Translation.
                'rigid': Rigid body.
                'rigid_isoscaling': Rigid body + isotropic scaling.
                'rigid_scaling': Rigid body + scaling.
                'affine': Full affine including translation, rotation, shearing, and scaling.
        n_cpu (int, optional): Number of CPU cores to use for parallel processing. Default is 2.

    Returns:
        None (Prints registration completion status for each patient folder).
    """
    data_dir = Path(data_dir)

    # Get a list of all patient folders
    patient_folders = list(data_dir.glob(id_regex))

    # multiprocessing doesn't work
    for folder in patient_folders:
        static_path = folder / static_stem
        moving_path = folder / moving_stem
        out_path = folder / output_stem

        # Perform rigid transform registration
        reg = Register(out_path, transform_method=transform_method)
        reg.transform(moving_path, static_path)

        if os.path.exists(out_path):
            print(f"Registration completed for patient: {folder.name}")
        else:
            print(f"{out_path} not found")


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
