# import numpy as np
# import six
# from matplotlib import pyplot as plt
#
# from src.dataset import WikiSarcoma
#
# import os
# import pydicom
# from dcmrtstruct2nii import dcmrtstruct2nii, list_rt_structs
# import pytest
# import SimpleITK as sitk
# from radiomics import featureextractor
#
# import os
from matplotlib import pyplot as plt

# params = os.path.join(os.getcwd(), 'example_settings', 'Params.yaml')
if __name__ == '__main__':
    # wiki_sarcoma = WikiSarcoma('./data', 'example_data/INFOclinical_STS.csv', params, 255, 4)
    from autorad.utils.preprocessing import get_paths_with_separate_folder_per_case
    root_dir='./data'

    paths_df = get_paths_with_separate_folder_per_case(root_dir, relative=True, image_stem='image',
                                                       mask_stem='mask_GTV_Mass')

    from src.dataset import ImageDataset
    import logging

    logging.getLogger().setLevel(logging.ERROR)

    image_dataset = ImageDataset(
        paths_df,
        ID_colname='ID',
        root_dir=root_dir
    )

    image_dataset.plot_examples(n=10, window=None, label=255)

    plt.show()