import os
import json
import cv2
import numpy as np
import nibabel as nib

def load_nifti(file_path):
    """
    Load a NIfTI file from the given file path.

    Parameters
    ----------
    file_path : str
        Path to the NIfTI file.

    Returns
    -------
    np.ndarray
        NIfTI image data.
    """
    nifti = nib.load(file_path)
    return nifti.get_fdata(), nifti.affine

def save_npy(array, file_path):
    """
    Save a NumPy array to the given file path.

    Parameters
    ----------
    array : np.ndarray
        Array to save.
    file_path : str
        Path to save the array.
    """
    np.save(file_path, array)

def save_png(image, file_path):
    """
    Save an image to the given file path.

    Parameters
    ----------
    image : np.ndarray
        Image to save.
    file_path : str
        Path to save the image.
    """
    cv2.imwrite(file_path, image)

def get_patients_in_processed_folder(data_dir):
    """
    Get the list of patient IDs from the metadata file in the processed folder.
    Parameters

    ----------
    data_dir : str
        Path to the processed data directory.

    Returns
    -------
    list
        List of patient IDs.
    """
    metadata_path = os.path.join(data_dir, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    patient_ids = set([])
    for k, v in metadata.items():
        patient_ids.add(v["patient_id"])

    return list(patient_ids)

def save_segmentation_mask(mask, affine, file_path):
    """
    Save a segmentation mask to a NIfTI file.

    Parameters
    ----------
    mask : np.ndarray
        Segmentation mask to save.
    affine : np.ndarray
        Affine transformation matrix.
    file_path : str
        Path to save the segmentation mask. It should end with .nii or .nii.gz.
        If it doesn't, the function will raise a ValueError.
    """
    if not file_path.endswith(('.nii', '.nii.gz')):
        raise ValueError("File path must end with .nii or .nii.gz")
    # Ensure type
    mask = mask.astype(np.uint8)
    # Save the mask as a NIfTI file
    nifti_mask = nib.Nifti1Image(mask, affine=affine)
    nib.save(nifti_mask, file_path)

def get_original_info(data_dir, patient_id):
    """
    Get the original information of a patient from the metadata file.

    Parameters
    ----------
    data_dir : str
        Path to the processed data directory.
    patient_id : str
        Patient ID.

    Returns
    -------
    np.ndarray
        Affine transformation matrix.
    tuple
        Original spacing of the image.
    tuple
        Original orientation of the image, e.g. (L, P, S).   
    """
    volume_dir = os.path.join(data_dir, patient_id, 'SEQ', f'CTport-{patient_id}.nii')
    image_nii = nib.load(volume_dir)

    # Get the affine transformation matrix
    affine = image_nii.affine.copy()
    # Get the original spacing
    original_spacing = image_nii.header.get_zooms()[:3]
    # Get the original orientation
    orientation = nib.aff2axcodes(image_nii.affine)

    return affine, original_spacing, orientation