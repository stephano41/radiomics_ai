from src.dataset.custom_dicom2nifti import dicomdir2nii, get_dicom_meta, get_dicom_meta, get_dicomdir_meta

dataset_meta = dicomdir2nii('./data/sarcoma_dicom', './data/sarcoma_nifti', series_description_filter=['T2.*AX','DIFF.*ADC'], codependent=True, codependent_tolerance=0.5)
dataset_meta.to_csv('data/sarcoma_meta.csv')
# print(dataset_meta)


# print(get_dicomdir_meta('data/sarcoma_dicom/SARC_00004/DICOMDIR'))
# inspect_dicom('data/sarcoma_dicom/SARC_00002/DICOMDIR')
# sitk_inspect_dicom('data/sarcoma_dicom/SARC_00001/DICOM/PA000003/ST000001/SE000001')
# dicommetas = []
# for dicomdir_file in Path('data/sarcoma_dicom').rglob("DICOMDIR"):
#     dicommetas.extend(get_dicomdir_meta(str(dicomdir_file)))
#
# meta_csv = pd.DataFrame(dicommetas)
# meta_csv.to_csv('outputs/test_sarcoma_csv.csv')
