import numpy as np
import six
from matplotlib import pyplot as plt

from src.dataset import WikiSarcoma

import os
import pydicom
from dcmrtstruct2nii import dcmrtstruct2nii, list_rt_structs

import SimpleITK as sitk
from radiomics import featureextractor

import os

params = os.path.join(os.getcwd(), 'example_settings', 'Params.yaml')

wiki_sarcoma = WikiSarcoma('./data', 'example_data/INFOclinical_STS.csv', params, 255)

print(wiki_sarcoma.x)
print(wiki_sarcoma.y)
print(wiki_sarcoma.x.shape)
print(wiki_sarcoma.y.shape)