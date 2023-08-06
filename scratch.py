from radiomics import imageoperations
from src.pipeline.pipeline_components import get_data
import SimpleITK as sitk

dataset = get_data('./data/meningioma_data', 't1ce','mask')

class DeepFeatureExtractor:
    def __init__(self):
        pass

    def execute(self, imageFilepath, maskFilepath, label=1, label_channel=0, minimumROIDimensions=1, minimumROISize=None):
        image = sitk.ReadImage(imageFilepath)
        mask = imageoperations.getMask(sitk.ReadImage(maskFilepath), label=label, label_channel=label_channel)

        boundingBox, correctedMask = imageoperations.checkMask(image, mask, minimumROIDimensions=minimumROIDimensions, minimumROISize=minimumROISize)

        if correctedMask is not None:
            mask = correctedMask

        cropped_image, cropped_mask = imageoperations.cropToTumorMask(image, mask, boundingBox)

        return cropped_image, mask


dfe = DeepFeatureExtractor()
image, mask = dfe.execute(dataset.image_paths[0], dataset.mask_paths[0])

print(image)