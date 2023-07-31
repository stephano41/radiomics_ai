import logging
import os
from typing import List

import pandas as pd

from autorad.config.type_definitions import PathLike
from autorad.utils.preprocessing import make_relative

log = logging.getLogger(__name__)


def get_multi_paths_with_separate_folder_per_case(
    data_dir: PathLike,
    image_stems: List[str] = ["image"],
    mask_stem: str = "segmentation",
    relative: bool = False,
) -> pd.DataFrame:

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
