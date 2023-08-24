import logging
import os
from typing import List, Tuple

import pandas as pd

from autorad.config.type_definitions import PathLike
from autorad.utils.preprocessing import make_relative

log = logging.getLogger(__name__)


def get_multi_paths_with_separate_folder_per_case(
    data_dir: PathLike,
    image_stems: Tuple[str, ...] = ("image"),
    mask_stem: str = "segmentation",
    relative: bool = False,
) -> pd.DataFrame:
    """
    Create a DataFrame containing paths to multiple images with separate folders per case.

    Parameters:
        data_dir (PathLike): Path to the main data directory containing individual patient folders.
        image_stems (List[str], optional): List of image stems for image files. Default is ["image"].
        mask_stem (str, optional): Stem for the segmentation mask filename. Default is "segmentation".
        relative (bool, optional): If True, store relative paths in the DataFrame instead of absolute paths.
            Default is False.

    Returns:
        pd.DataFrame: DataFrame containing paths to images and segmentation masks for each case.
            The DataFrame has columns: ["ID"] + image_columns + ["segmentation_path"], where "ID" corresponds to
            the case identifier, image_columns contains paths to the image files, and "segmentation_path" contains
            paths to the segmentation mask files.

    Note:
        This function iterates through the individual patient folders in the data directory and looks for image files
        and a segmentation mask file within each folder. The image file(s) are expected to be named using the specified
        image_stems, and the segmentation mask is expected to have the filename specified by the mask_stem. The function
        creates a DataFrame containing the paths to these files for each case and returns it.

        If relative is set to True, the function converts the paths to relative paths with respect to the data_dir.
        Otherwise, the paths are stored as absolute paths in the DataFrame.

        The function logs warnings if an image or mask file is missing for any case.
    """

    image_columns = [f"image_{image_stem}" for image_stem in image_stems]
    columns = ["ID"] + image_columns + ["segmentation_path"]
    data_dict = {column: [] for column in columns}

    for i, id_ in enumerate(os.listdir(data_dir)):
        id_dir = os.path.join(data_dir, id_)
        if not os.path.isdir(id_dir):
            continue

        for image_stem in image_stems:
            image_path = os.path.join(id_dir, f"{image_stem}.nii.gz")
            if not os.path.exists(image_path):
                log.warning(f"Image for ID={id_} and stem={image_stem} does not exist ({image_path})")
                continue
            data_dict[f"image_{image_stem}"].insert(i, image_path)

        mask_path = os.path.join(id_dir, f"{mask_stem}.nii.gz")

        if not os.path.exists(mask_path):
            log.warning(f"Mask for ID={id_} does not exist ({mask_path})")
            continue

        data_dict['ID'].insert(i, id_)
        data_dict['segmentation_path'].insert(i, mask_path)

    if relative:
        for image_stem in image_stems:
            data_dict[f"image_{image_stem}"] = make_relative(data_dict[f"image_{image_stem}"], data_dir)
        data_dict["segmentation_path"] = make_relative(data_dict["segmentation_path"], data_dir)

    path_df = pd.DataFrame(data_dict)

    return path_df


# def create_autoencoder(image_processor_kwargs, )