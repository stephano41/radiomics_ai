from src.dataset.custom_dicom2nifti import dicomdir2nii

dicomdir2nii('./data/sarcoma_dicom', output_folder='./data/sarcoma_nifti',
             codependent=True,
             codependent_tolerance=0.5,
             series_description_filter=['T2.*AX','DIFF.*ADC'])