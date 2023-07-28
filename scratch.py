import multiprocessing
import shutil
import tempfile
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from multiprocessing import Pool

import os


def process_patient(patient_info):
    patient_src, dest_patient_folder = patient_info
    expected_files = {'ADC.nii.gz', 'flair.nii.gz', 't1.nii.gz', 't1ce.nii.gz', 't2.nii.gz', 'mask.nii.gz'}
    found_files = set()
    for file in os.listdir(patient_src):
        if file.endswith('.nii') or file.endswith('.gz'):
            if 'ADC' in file:
                new_filename = 'ADC.nii'
            elif 'flair' in file:
                new_filename = 'flair.nii.gz'
            elif 't1ce' in file:
                new_filename = 't1ce.nii.gz'
            elif 't1' in file:
                new_filename = 't1.nii.gz'
            elif 't2' in file:
                new_filename = 't2.nii.gz'
            elif '_1.nii' in file or '_2.nii' in file or '_3.nii' in file:
                new_filename = 'mask.nii.gz'
            else:
                warnings.warn(f"Unexpected file {file} in patient folder {patient_src}")
                continue

            src_file_path = os.path.join(patient_src, file)
            dest_file_path = os.path.join(dest_patient_folder, new_filename)
            shutil.copy(src_file_path, dest_file_path)
            print(f"Copied file {src_file_path} to {dest_file_path}")

            found_files.add(new_filename)

    if expected_files != found_files:
        missing_files = expected_files - found_files
        warnings.warn(f"Missing files {missing_files} in patient folder {patient_src}")

    return dest_patient_folder


def copy_patient_data(src_dir, dest_dir, n_cpu=2):
    """
    Copies patient data from the source directory to the destination directory
    while maintaining an ascending pattern of integer-based patient IDs. The function
    creates a mapping of patient IDs, grades, and their original data paths,
    and saves the mapping to a CSV file. It also renames the files as per the required format.

    Parameters:
        src_dir (str): The path to the source directory containing patient data organized by grades.
        dest_dir (str): The path to the destination directory where the copied patient data will be stored.

    Returns:
        None: The function performs the copy operation, file renaming, and saves the mapping to a CSV file.

    Raises:
        FileNotFoundError: If any of the expected files are missing.
        ValueError: If there are additional files that don't match the pattern.
    """
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    patient_data = []
    patient_info_list = []
    for grade_folder in os.listdir(src_dir):
        grade_path = os.path.join(src_dir, grade_folder)
        if os.path.isdir(grade_path):
            for patient_folder in os.listdir(grade_path):
                patient_src = os.path.join(grade_path, patient_folder)
                if os.path.isdir(patient_src):
                    dest_patient_id = len(os.listdir(dest_dir)) + 1
                    dest_patient_folder = os.path.join(dest_dir, f"ID_{dest_patient_id}")
                    os.makedirs(dest_patient_folder, exist_ok=True)
                    patient_info_list.append((patient_src, dest_patient_folder))
                    grade = 1 if grade_folder == 'Grade1Data' else 2
                    patient_data.append((f"ID_{dest_patient_id}", grade, patient_src))

    # Process patient data in parallel using multiprocessing
    with Pool(n_cpu) as pool:
        dest_patient_folders = pool.map(process_patient, patient_info_list)

    # Create a pandas DataFrame from the patient_data list
    df = pd.DataFrame(patient_data, columns=["Patient_ID", "Grade", "Original_Data_Path"])

    # Save the DataFrame to a CSV file
    df.to_csv("patient_grade_mapping.csv", index=False)


def generate_patient_paths_csv(src_dir, output_csv=None):
    """
    Generates a CSV file that records the paths of each modality and mask for each patient
    under different columns.

    Parameters:
        src_dir (str): The path to the source directory containing patient data organized by grades.
        output_csv (str): The path to the output CSV file where the paths will be saved.

    Returns:
        None: The function creates the CSV file with the patient paths.

    Note:
        This function assumes the original folder structure and naming conventions as provided.

    Example:
        Suppose we have the following directory structure in 'src_dir':

        src_dir/
            |- meningioma/
                |- Grade1Data/
                    |- patient_1/
                        |- patient_1_ADC.nii
                        |- patient_1_flair.nii
                        |- patient_1_t1.nii
                        |- patient_1_t1ce.nii
                        |- patient_1_t2.nii
                        |- patient_1_1.nii
                    |- patient_2/
                        |- patient_2_ADC.nii
                        |- patient_2_flair.nii
                        |- patient_2_t1.nii
                        |- patient_2_t1ce.nii
                        |- patient_2_t2.nii
                        |- patient_2_1.nii
                |- Grade2Data/
                    |- patient_1/
                        |- patient_1_ADC.nii
                        |- patient_1_flair.nii
                        |- patient_1_t1.nii
                        |- patient_1_t1ce.nii
                        |- patient_1_t2.nii
                        |- patient_1_1.nii
                    |- patient_2/
                        |- patient_2_ADC.nii
                        |- patient_2_flair.nii
                        |- patient_2_t1.nii
                        |- patient_2_t1ce.nii
                        |- patient_2_t2.nii
                        |- patient_2_1.nii

        After calling 'generate_patient_paths_csv(src_dir, output_csv)', the 'output_csv' will have the following structure:

        Patient_ID, Grade, ADC_Path, Flair_Path, T1_Path, T1CE_Path, T2_Path, Mask_Path
        patient_1, Grade1Data, path_to_ADC, path_to_flair, path_to_t1, path_to_t1ce, path_to_t2, path_to_mask
        patient_2, Grade1Data, path_to_ADC, path_to_flair, path_to_t1, path_to_t1ce, path_to_t2, path_to_mask
        patient_1, Grade2Data, path_to_ADC, path_to_flair, path_to_t1, path_to_t1ce, path_to_t2, path_to_mask
        patient_2, Grade2Data, path_to_ADC, path_to_flair, path_to_t1, path_to_t1ce, path_to_t2, path_to_mask
    """
    patient_data = []
    unique_id = 0
    for grade_folder in os.listdir(src_dir):
        grade_path = os.path.join(src_dir, grade_folder)
        if os.path.isdir(grade_path):
            for patient_folder in os.listdir(grade_path):
                patient_src = os.path.join(grade_path, patient_folder)
                if os.path.isdir(patient_src):
                    patient_id = patient_folder
                    adc_path = os.path.join(patient_src, f"{patient_id}_ADC.nii")
                    flair_path = os.path.join(patient_src, f"{patient_id}_flair.nii.gz")
                    t1_path = os.path.join(patient_src, f"{patient_id}_t1.nii.gz")
                    t1ce_path = os.path.join(patient_src, f"{patient_id}_t1ce.nii.gz")
                    t2_path = os.path.join(patient_src, f"{patient_id}_t2.nii.gz")
                    mask_path = os.path.join(patient_src, f"{patient_id}_1.nii.gz")

                    if not os.path.exists(mask_path):
                        mask_path = os.path.join(patient_src, f"{patient_id}_2.nii.gz")

                    # Check if the paths exist and raise a warning if they don't
                    for path in [adc_path, flair_path, t1_path, t1ce_path, t2_path, mask_path]:
                        if not os.path.exists(path):
                            warnings.warn(f"Path {path} does not exist.")

                    # Make the paths relative to src_dir
                    adc_path = os.path.relpath(adc_path, src_dir)
                    flair_path = os.path.relpath(flair_path, src_dir)
                    t1_path = os.path.relpath(t1_path, src_dir)
                    t1ce_path = os.path.relpath(t1ce_path, src_dir)
                    t2_path = os.path.relpath(t2_path, src_dir)
                    mask_path = os.path.relpath(mask_path, src_dir)

                    patient_data.append(
                        {
                            "ID": f"ID_{unique_id}",
                            "Patient_ID": patient_id,
                            "Grade": 1 if grade_folder == "Grade1Data" else 2,
                            "ADC_Path": adc_path,
                            "Flair_Path": flair_path,
                            "T1_Path": t1_path,
                            "T1CE_Path": t1ce_path,
                            "T2_Path": t2_path,
                            "Mask_Path": mask_path,
                        }
                    )
                    unique_id += 1

    # Create a pandas DataFrame from the patient_data list
    df = pd.DataFrame(patient_data,
                      columns=["ID", "Patient_ID", "Grade", "ADC_Path", "Flair_Path", "T1_Path", "T1CE_Path", "T2_Path",
                               "Mask_Path"])

    # Save the DataFrame to a CSV file
    if output_csv is not None:
        df.to_csv(output_csv, index=False)

    return df


import SimpleITK as sitk
from src.registeration.reg_funcs import move_plot_nii, Register


# if __name__ == '__main__':
#     copy_patient_data('./data/meningioma',
#                       './data/meningioma_data',
#                       5)
# generate_patient_paths_csv('./data/meningioma', './data/meningioma_paths.csv')
# dataset = get_data('./data/meningioma_data', image_stem='t2', mask_stem='mask')
# dataset.plot_examples(10, label=1, window=None)
# plt.show()

def register_patients(data_dir, static_stem, moving_stem, output_stem='warped.nii', transform_method='rigid', n_cpu=2):
    data_dir = Path(data_dir)

    # Get a list of all patient folders
    patient_folders = list(data_dir.glob("ID_*"))

    # Use multiprocessing.Pool to parallelize the registration process
    with multiprocessing.Pool(n_cpu) as pool:
        pool.starmap(_register_patient, [(folder, static_stem, moving_stem, transform_method, output_stem) for folder in patient_folders])


def _register_patient(patient_folder, static_stem, moving_stem, transform_method, output_stem='warped.nii'):
    static_path = patient_folder / static_stem
    moving_path = patient_folder / moving_stem
    out_path = patient_folder / output_stem

    # Perform rigid transform registration
    reg = Register(out_path, transform_method=transform_method)
    reg.transform(moving_path, static_path)

    if os.path.exists(out_path):
        print(f"Registration completed for patient: {patient_folder.name}")
    else:
        print(f"{out_path} not found")


if __name__ == '__main__':
    register_patients('./data/meningioma_data', static_stem='t1ce.nii.gz',
                      moving_stem='ADC.nii',
                      output_stem='registered_adc.nii',
                      transform_method='rigid',
                      n_cpu=1)