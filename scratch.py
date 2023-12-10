from src.dataset.skorch_subject_ds import SkorchSubjectsDataset
import os
from src.pipeline.pipeline_components import get_multimodal_feature_dataset, split_feature_dataset
import torchio as tio


output_dir = './outputs/meningioma+autoencoder/2023-11-01-02-38-00'

feature_dataset = get_multimodal_feature_dataset(data_dir='./data/meningioma_data',
                                                 image_stems=('registered_adc', 't2', 'flair', 't1', 't1ce'),
                                                 mask_stem='mask',
                                                 target_column='Grade',
                                                 label_csv_path='./data/meningioma_meta.csv',
                                                 extraction_params='./conf/radiomic_params/meningioma_mr.yaml',
                                                 feature_df_merger={
                                                     '_target_': 'src.pipeline.df_mergers.meningioma_df_merger'},
                                                 n_jobs=6,
                                                 existing_feature_df=os.path.join(output_dir, 'extracted_features.csv'),
                                                 additional_features=['ID']
                                                 )
feature_dataset = split_feature_dataset(feature_dataset,
                                        existing_split=os.path.join(output_dir, 'splits.yml'))

ds = SkorchSubjectsDataset(feature_dataset.X.ID, None, data_dir='./data/meningioma_data',
                           image_stems=('registered_adc', 't2', 'flair', 't1', 't1ce'),
                           mask_stem='mask',
                           transform=tio.Compose([tio.Resample(target=1),tio.ToCanonical(),tio.CropOrPad(target_shape=None, mask_name='mask')]))

shapes = [max(ds[i][0].shape) for i in range(len(ds))]
print(max(shapes))