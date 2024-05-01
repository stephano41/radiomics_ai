import os
from typing import Sequence

import SimpleITK as sitk
import pandas as pd
from autorad.config.type_definitions import PathLike
from autorad.utils.preprocessing import make_relative
from matplotlib import pyplot as plt
import logging

log = logging.getLogger(__name__)


def plot_debug(stk_image):
    plt.figure()
    plt.imshow(sitk.GetArrayFromImage(stk_image)[5, :, :], cmap='gray')
    plt.show()


def plot_slices(output_tensor, slice_index, num_samples=5, original_tensor=None,
                title=None, save_dir=None):
    """
    Plot a slice from each image modality of the output tensor for a specified number of samples,
    optionally including the corresponding slices from an original tensor for comparison.

    This function is primarily used to visualize the outputs of a model, such as an autoencoder, by displaying
    the reconstructed slices alongside the original data (if provided).

    Parameters:
        output_tensor (torch.Tensor): The tensor containing image data outputs from a model.
                                     The tensor should have dimensions (batch_size, num_modalities, depth, height, width).
        slice_index (int): The index of the slice to be plotted within the depth dimension.
        num_samples (int, optional): The number of samples to display from the batch. Defaults to 5.
        original_tensor (torch.Tensor, optional): The tensor containing the original image data for comparison.
                                                  Must have the same dimensions as output_tensor.
        title (str, optional): The title for the plots. If specified, this title will be displayed above each figure.
        save_dir (str, optional): Directory path where the plots will be saved. If not specified, plots will not be saved.

    Returns:
        None

    Example:
        >>> # Assuming output_tensor and original_tensor are 5D torch.Tensors from an autoencoder
        >>> plot_slices(output_tensor, 10, num_samples=3, original_tensor=original_tensor,
                        title="Comparison of Autoencoder Outputs", save_dir="/path/to/save")

    Note:
        - This function uses matplotlib for plotting, and figures will be shown using plt.show().
          Ensure that this behavior is suitable for your use case (e.g., running in a Jupyter notebook).
        - If save_dir is specified, each plot will be saved with a filename corresponding to the sample index,
          e.g., '/path/to/save_0.png', '/path/to/save_1.png', etc.
    """
    batch_size, num_modalities, length, width, height = output_tensor.shape
    plt.close('all')

    for sample_idx in range(min(num_samples, batch_size)):
        plt.figure(figsize=(15, 5))  # Adjust the figure size as needed

        for modality_idx in range(num_modalities):
            plt.subplot(2, num_modalities, modality_idx + 1)
            plt.imshow(output_tensor[sample_idx, modality_idx, slice_index, :, :], cmap='gray')
            plt.title(f'generated Sample {sample_idx + 1}, {modality_idx}')
            plt.axis('off')

        if original_tensor is not None:
            for modality_idx in range(num_modalities):
                plt.subplot(2, num_modalities, num_modalities + modality_idx + 1)
                plt.imshow(original_tensor[sample_idx, modality_idx, slice_index, :, :], cmap='gray')
                plt.title(f'original Sample {sample_idx + 1}, {modality_idx}')
                plt.axis('off')

        if title is not None:
            plt.suptitle(title)
        if save_dir is not None:
            plt.savefig(save_dir + f'_{sample_idx}.png')

        plt.show()


def get_multi_paths_with_separate_folder_per_case(
        data_dir: PathLike,
        image_stems: Sequence[str, ...] = ("image"),
        mask_stem: str = "segmentation",
        relative: bool = False,
) -> pd.DataFrame:
    """
       Generates a DataFrame containing the paths to multiple image files and a single mask file per case, based
       on specified filename stems, from a directory structure where each case is contained in its own subfolder.

       The function iterates through each subfolder in the provided `data_dir`, identifying image files that match
       the specified `image_stems` and a mask file that matches the `mask_stem`. It then compiles these paths into
       a DataFrame, which can be adjusted to contain relative or absolute paths.

       Parameters:
       ----------
       data_dir : PathLike
           The directory containing subfolders for each case. Each subfolder should contain the image files and a
           single mask file.
       image_stems : Sequence[str], optional
           A sequence of strings representing the filename stems to identify multiple image files in each case
           subfolder (default is a tuple containing just "image").
       mask_stem : str, optional
           The filename stem used to identify the mask file in each case subfolder (default is "segmentation").
       relative : bool, optional
           A boolean flag that determines whether the paths in the resulting DataFrame should be relative to
           `data_dir` (if True) or absolute (if False, default).

       Returns:
       -------
       pd.DataFrame
           A pandas DataFrame with columns corresponding to 'ID', one column for each image stem prefixed by 'image_',
           and 'segmentation_path' for the mask path. Each row represents a single case.

       Notes:
       -----
       - Ensure that every subfolder in the `data_dir` ideally contains all the specified image files and one mask file.
       - Missing files will be logged as warnings and the corresponding paths will not be included in the DataFrame.
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
