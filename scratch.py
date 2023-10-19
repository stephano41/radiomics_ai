import pandas as pd
from torchio import SubjectsDataset

from src.preprocessing import SitkImageProcessor
import os
import torchio as tio
import torch
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from torchio.visualization import plot_histogram
from tqdm import tqdm


def get_adc_tensor(ID_list, data_path, tio_transforms=None):
    sitk_processor = SitkImageProcessor(data_path, mask_stem='mask',
                                        image_stems=('registered_adc',))

    subject_list = sitk_processor.transform(ID_list)['subjects'].to_list()

    subject_dataset = SubjectsDataset(subject_list, transform=tio_transforms)

    # adc_tensor = torch.concatenate(
    #     [torch.concatenate([torch.flatten(i.data).nonzero() for i in subject.get_images()]) for subject in
    #      subject_dataset])

    adc_tensor = torch.tensor([[torch.mean(torch.flatten(i.data).nonzero().float()) for i in subject.get_images()] for subject in
         tqdm(subject_dataset)])

    return adc_tensor


def get_id_by_grade(data_meta, grade):
    data_meta = pd.read_csv(data_meta)
    # ID_list = data_meta['ID'].to_list()
    id = data_meta['Patient_ID'].loc[(data_meta['Grade'] == grade)].to_list()

    return id


def plot_histogram_by_grade(data_meta, data_path, grade, transforms=None, **kwargs):
    id_list = get_id_by_grade(data_meta, grade)
    adc_tensor = get_adc_tensor(id_list, data_path, tio_transforms=transforms)

    # hist = ndi.histogram(adc_tensor, min=0, max=max_histo, bins=bins)
    # plt.plot(hist)
    plot_histogram(adc_tensor.numpy(), show=False, **kwargs)

    return plt.gcf()

def get_mean_by_grade(data_meta, data_path, grade, transforms=None):
    id_list = get_id_by_grade(data_meta, grade)
    adc_tensor = get_adc_tensor(id_list, data_path, tio_transforms=transforms)

    return torch.mean(adc_tensor), torch.std(adc_tensor)


data_path = './data/meningioma_data'
data_meta = './data/meningioma_meta.csv'
# ID_list = data_meta['ID'].to_list()

transforms = tio.Compose([tio.Resample((1, 1, 1)),
                          tio.CropOrPad(mask_name='mask'),
                          tio.ToCanonical(),
                          tio.Mask(masking_method='mask', outside_value=0),
                          tio.ZNormalization(masking_method='mask')
                          ])

print(get_mean_by_grade(data_meta,
                        data_path=data_path,
                        grade=1,
                        transforms=transforms))
print(get_mean_by_grade(data_meta,
                        data_path=data_path,
                        grade=0,
                        transforms=transforms))

# plot_histogram_by_grade(data_meta,
#                         data_path=data_path,
#                         grade=1,
#                         transforms=transforms,
#                         alpha=0.5,
#                         color='red',
#                         label='Grade 1',
#                         density=False
#                         )
#
# plot_histogram_by_grade(data_meta,
#                         data_path=data_path,
#                         grade=0,
#                         transforms=transforms,
#                         alpha=0.5,
#                         color='lightblue',
#                         label='Grade 0',
#                         density=False
#                         )
# plt.legend()
# plt.show()