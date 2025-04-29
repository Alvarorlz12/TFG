import os
import json
import cv2
import numpy as np
import nibabel as nib

from scipy.ndimage import zoom
from nibabel.processing import resample_from_to
from nibabel.orientations import apply_orientation

import src.data.transforms as transforms

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

def pad_volume(volume, target_size):
    """Pad a 3D volume along the Z-axis to match the target size."""
    current_size = volume.shape[2]
    if current_size >= target_size:
        return volume  # No padding needed

    pad_width = target_size - current_size
    pad_before = pad_width // 2  # Padding before
    pad_after = pad_width - pad_before  # Padding after

    return np.pad(volume, ((0, 0), (0, 0), (pad_before, pad_after)), mode="constant")

def resample_volume_spacing(image, target_spacing, current_spacing=None):
    """
    Resample the image to the target spacing.

    Parameters
    ----------
    image_nii : nibabel.Nifti1Image or np.ndarray
        NIfTI image or image data to resample.
    target_spacing : tuple of float
        Target spacing for resampling.
    current_spacing : tuple of float, optional
        Current spacing of the image. If not specified, it is obtained from the
        image header. If None, the current spacing is used.

    Returns
    -------
    np.ndarray
        Resampled image data.
    """
    if isinstance(image, nib.Nifti1Image):
        current_spacing = image.header.get_zooms()[:3]
        image = image.get_fdata().astype(np.float64)
    elif isinstance(image, np.ndarray):
        if current_spacing is None:
            raise ValueError("Current spacing must be provided when image is a NumPy array.")
        
    zoom_factors = np.array(current_spacing) / np.array(target_spacing)
    
    return zoom(image, zoom_factors, order=1)  # Linear interpolation
    
def resample_mask_spacing(mask, target_spacing, current_spacing=None):
    """
    Resample the mask to the target spacing.

    Parameters
    ----------
    mask_nii : nibabel.Nifti1Image or np.ndarray
        NIfTI image or image data to resample.
    target_spacing : tuple of float
        Target spacing for resampling.
    current_spacing : tuple of float, optional
        Current spacing of the image. If not specified, it is obtained from the
        image header. If None, the current spacing is used.

    Returns
    -------
    np.ndarray
        Resampled mask data.
    """
    if isinstance(mask, nib.Nifti1Image):
        current_spacing = mask.header.get_zooms()[:3]
        mask = mask.get_fdata().astype(np.uint8)
    elif isinstance(mask, np.ndarray):
        if current_spacing is None:
            raise ValueError("Current spacing must be provided when mask is a NumPy array.")
        
    zoom_factors = np.array(current_spacing) / np.array(target_spacing)
    
    return zoom(mask, zoom_factors, order=0)  # Nearest-neighbor interpolation

def update_affine(affine, target_spacing):
    """
    Update the affine matrix to reflect the new spacing.

    Parameters
    ----------
    affine : np.ndarray
        Affine matrix to update.
    target_spacing : tuple of float
        Target spacing for resampling.

    Returns
    -------
    np.ndarray
        Updated affine matrix.
    """
    new_affine = affine.copy()
    for i in range(3):
        new_affine[i, i] = -target_spacing[i] if affine[i, i] < 0 else target_spacing[i]
    
    return new_affine

def crop_or_pad_to_size(image, target_size):
    """
    Crop or pad the image to the target size.

    Parameters
    ----------
    image : np.ndarray
        Image to crop or pad.
    target_size : tuple of int
        Target size for cropping or padding.

    Returns
    -------
    np.ndarray
        Cropped or padded image.
    """
    Hi, Wi, _ = image.shape
    Ht, Wt = target_size

    if Hi == Ht and Wi == Wt:
        return image  # No cropping or padding needed

    # Crop the image if it's larger than the target size
    if Hi > Ht or Wi > Wt:
        h_start = (Hi - Ht) // 2
        w_start = (Wi - Wt) // 2
        return image[h_start:h_start + Ht, w_start:w_start + Wt, :]
    # Pad the image if it's smaller than the target size
    else:
        pad_h_after = (Ht - Hi) // 2
        pad_h_before = (Ht - Hi) - pad_h_after
        pad_w_after = (Wt - Wi) // 2
        pad_w_before = (Wt - Wi) - pad_w_after
        return np.pad(
            image,
            ((pad_h_before, pad_h_after), (pad_w_before, pad_w_after), (0, 0)),
            mode="constant",
            constant_values=0
        )
    
def process_volume(
    patient_dir,
    target_spacing=None,
    target_orientation=("R", "P", "S"),
    rotation_axes=(0, 1),
    h_min=0, h_max=512,
    w_min=0, w_max=512
):
    """
    Process a patient directory containing NIfTI files. It loads the NIfTI files
    and returns the image and masks as NumPy arrays. The image is reoriented,
    rotated, and cropped to the specified size. The masks are also reoriented,
    rotated, and cropped to the specified size. The masks are combined into a 
    single mask.

    Parameters
    ----------
    patient_dir : str
        Path to the patient directory.
    target_spacing : tuple of float, optional
        Target spacing for resampling, by default None. If None, no resampling
        is done.
    target_orientation : Tuple[str], optional
        Target orientation of the image, by default ("R", "P", "S").
    rotation_axes : Tuple[int], optional
        Axis to rotate the image, by default (0, 1).
    h_min : int, optional
        Minimum height of the image, by default 0.
    h_max : int, optional
        Maximum height of the image, by default 512.
    w_min : int, optional
        Minimum width of the image, by default 0.
    w_max : int, optional
        Maximum width of the image, by default 512.

    Returns
    -------
    np.ndarray
        Image data.
    np.ndarray
        Combined mask data.
    str
        Patient ID.
    """
    patient_id = os.path.basename(patient_dir)
    reorient = transforms.Orientation(target_orientation)

    # Image and mask paths
    image_path = os.path.join(patient_dir, "SEQ", f"CTport-{patient_id}.nii")
    mask_paths = {
        "pancreas": os.path.join(patient_dir, "SEG", f"Pancreas-{patient_id}.nii"),
        "tumor": os.path.join(patient_dir, "SEG", f"Tumor-{patient_id}.nii"),
        "arteries": os.path.join(patient_dir, "SEG", f"Arterias-{patient_id}.nii"),
        "veins": os.path.join(patient_dir, "SEG", f"Venas-{patient_id}.nii"),
    }

    # Load the image and masks
    image_nii = nib.load(image_path)
    affine = image_nii.affine

    if target_spacing is not None:
        image = resample_volume_spacing(image_nii, target_spacing)
        image = crop_or_pad_to_size(image, (512, 512))  # Crop or pad to size
        affine = update_affine(affine, target_spacing)  # Update the affine matrix
        image, transform = reorient(image, affine)  # Reorient the image
    else:
        image, transform = reorient(image_nii) # Reorient the image

    image = np.rot90(image, k=-1, axes=rotation_axes)   # Rotate the image
    image = image[h_min:h_max, w_min:w_max, :]  # Apply cropping

    masks = np.zeros_like(image)

    # Combine the segmentation masks
    for i, (_, mask_path) in enumerate(mask_paths.items(), start=1):
        mask_nii = nib.load(mask_path)

        # Ensure the number of slices match the image
        if mask_nii.shape[2] != image_nii.shape[2]:
            mask_nii = resample_from_to(mask_nii, image_nii, order=0)

        # Resample the mask to the target spacing if specified
        if target_spacing is not None:
            mask_data = resample_mask_spacing(mask_nii, target_spacing)
            mask_data = crop_or_pad_to_size(mask_data, (512, 512))
        else:
            mask_data = mask_nii.get_fdata()

        # Binarize the mask
        mask_data = (mask_data > 0).astype(np.uint8)

        # Apply orientation transformation
        mask_data = apply_orientation(mask_data, transform)

        # Rotate the mask
        mask_data = np.rot90(mask_data, k=-1, axes=rotation_axes)

        # Apply cropping
        mask_data = mask_data[h_min:h_max, w_min:w_max, :]

        # Class label assignment: 1 for pancreas, 2 for tumor...
        masks[mask_data > 0] = i

    return image, masks, patient_id

def process_patient_3d(
    patient_dir,
    output_dir,
    target_spacing=None,
    subvolume_size=64,
    subvolume_stride=32,
    target_orientation=("R", "P", "S"),
    rotation_axes=(0, 1),
    h_min=0, h_max=512,
    w_min=0, w_max=512
):
    """
    Process a patient directory containing NIfTI files. It loads the NIfTI files
    and divides the 3D volume into sub-volumes of a specified size. The sub-volumes
    are saved as NumPy files.

    Parameters
    ----------
    patient_dir : str
        Path to the patient directory.
    output_dir : str
        Directory to save the processed sub-volumes.
    target_spacing : tuple of float, optional
        Target spacing for resampling, by default None. If None, no resampling 
        is done.
    subvolume_size : int, optional
        Size of the sub-volumes, by default 64.
    subvolume_stride : int, optional
        Stride of the sub-volumes, by default 32.
    target_orientation : Tuple[str], optional
        Target orientation of the image, by default ("R", "P", "S").
    rotation_axes : Tuple[int], optional
        Axis to rotate the image, by default (0, 1).
    h_min : int, optional
        Minimum height of the image, by default 0.
    h_max : int, optional
        Maximum height of the image, by default 512.
    w_min : int, optional
        Minimum width of the image, by default 0.
    w_max : int, optional
        Maximum width of the image, by default 512.

    Returns
    -------
    int
        Number of sub-volumes saved.
    dict
        Metadata for the sub-volumes.
    """
    image, masks, patient_id = process_volume(
        patient_dir,
        target_spacing=target_spacing,
        target_orientation=target_orientation,
        rotation_axes=rotation_axes,
        h_min=h_min, h_max=h_max,
        w_min=w_min, w_max=w_max
    )

    num_slices = image.shape[2]

    padded = 0
    if num_slices < subvolume_size:
        print(f"⚠️ {patient_id} has less slices than the subvolume size. Adding padding...")
        image = pad_volume(image, subvolume_size)
        masks = pad_volume(masks, subvolume_size)
        padded = subvolume_size - num_slices
        num_slices = subvolume_size

    # Transpose the image and masks to (D, H, W)
    image = np.transpose(image, (2, 0, 1))
    masks = np.transpose(masks, (2, 0, 1))

    # Saving directories
    output_img_dir = os.path.join(output_dir, "images")
    output_mask_dir = os.path.join(output_dir, "masks")
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    metadata = {}   # Metadata for the sub-volumes

    # Generate sub-volumes
    subvolume_idx = 0
    for start in range(0, num_slices - subvolume_size + 1, subvolume_stride):
        image_subvol = image[start:start + subvolume_size, ...]
        mask_subvol = masks[start:start + subvolume_size, ...]
        # image_subvol = image[:, :, start:start + subvolume_size]
        # mask_subvol = masks[:, :, start:start + subvolume_size]

        img_filename = f"image_{patient_id}_{subvolume_idx}.npy"
        mask_filename = f"mask_{patient_id}_{subvolume_idx}.npy"

        # Save the sub-volumes
        image_save_path = os.path.join(output_img_dir, img_filename)
        mask_save_path = os.path.join(output_mask_dir, mask_filename)

        save_npy(image_subvol, image_save_path)
        save_npy(mask_subvol, mask_save_path)

        metadata[img_filename] = {
            "patient_id": patient_id,
            "slices": [start, start + subvolume_size - 1],
            "mask_filename": mask_filename
        }

        subvolume_idx += 1

    if padded > 0:
        metadata[img_filename]["padded"] = padded

    # Last sub-volume
    if num_slices % subvolume_stride != 0:
        image_subvol = image[-subvolume_size:, ...]
        mask_subvol = masks[-subvolume_size:, ...]
        # image_subvol = image[:, :, -subvolume_size:]
        # mask_subvol = masks[:, :, -subvolume_size:]

        img_filename = f"image_{patient_id}_{subvolume_idx}.npy"
        mask_filename = f"mask_{patient_id}_{subvolume_idx}.npy"

        # Save the sub-volumes
        save_npy(image_subvol, os.path.join(output_img_dir, img_filename))
        save_npy(mask_subvol, os.path.join(output_mask_dir, mask_filename))

        metadata[img_filename] = {
            "patient_id": patient_id,
            "slices": [num_slices - subvolume_size, num_slices - 1],
            "mask_filename": mask_filename
        }

        subvolume_idx += 1

    return subvolume_idx, metadata

def process_patient_2d(
    patient_dir,
    output_dir,
    target_spacing=None,
    target_orientation=("R", "P", "S"),
    rotation_axes=(0, 1),
    h_min=0, h_max=512,
    w_min=0, w_max=512
):
    """
    Process a patient directory containing NIfTI files. It loads the NIfTI files
    and saves the 2D slices as NumPy files. The masks are also saved as PNG
    files. The slices are saved in the specified orientation and with the 
    specified orientation and with the specified rotation. The masks are also 
    saved in the same orientation and rotation.

    Parameters
    ----------
    patient_dir : str
        Path to the patient directory.
    output_dir : str
        Directory to save the processed sub-volumes.
    target_spacing : tuple of float, optional
        Target spacing for resampling, by default None. If None, no resampling
        is done.
    target_orientation : Tuple[str], optional
        Target orientation of the image, by default ("R", "P", "S").
    rotation_axes : Tuple[int], optional
        Axis to rotate the image, by default (0, 1).
    h_min : int, optional
        Minimum height of the image, by default 0.
    h_max : int, optional
        Maximum height of the image, by default 512.
    w_min : int, optional
        Minimum width of the image, by default 0.
    w_max : int, optional
        Maximum width of the image, by default 512.

    Returns
    -------
    int
        Number of slices saved.
    dict
        Metadata for the slices.
    """
    image, masks, patient_id = process_volume(
        patient_dir,
        target_spacing=target_spacing,
        target_orientation=target_orientation,
        rotation_axes=rotation_axes,
        h_min=h_min, h_max=h_max,
        w_min=w_min, w_max=w_max
    )

    num_slices = image.shape[2]

    # Transpose the image and masks to (D, H, W)
    image = np.transpose(image, (2, 0, 1))
    masks = np.transpose(masks, (2, 0, 1))

    # Saving directories
    output_img_dir = os.path.join(output_dir, "images")
    output_mask_dir = os.path.join(output_dir, "masks")
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    metadata = {}   # Metadata for the slices

    # Generate slices
    for slice in range(num_slices):
        image_slice = image[slice, ...]
        mask_slice = masks[slice, ...]

        img_filename = f"image_{patient_id}_{slice:03d}.npy"
        mask_filename = f"mask_{patient_id}_{slice:03d}.png"

        # Save the slice
        image_save_path = os.path.join(output_img_dir, img_filename)
        mask_save_path = os.path.join(output_mask_dir, mask_filename)

        save_npy(image_slice, image_save_path)
        save_png(mask_slice.astype(np.uint8), mask_save_path)

        metadata[img_filename] = {
            "patient_id": patient_id,
            "slice_index": slice,
            "mask_filename": mask_filename
        }

    return num_slices, metadata

def apply_window(image, window_level, window_width):
    """
    Apply a window to the image, which is commonly used in medical imaging to
    visualize the image with a specific intensity range. It changes the contrast
    (level) and brightness (window) of the image.

    Parameters
    ----------
    image : np.ndarray
        Image to apply the window to.
    window_level : int
        Window level.
    window_width : int
        Window width.

    Returns
    -------
    np.ndarray
        Image with window applied.
    """
    window_min = window_level - (window_width / 2)
    window_max = window_level + (window_width / 2)
    windowed_image = np.clip(image, window_min, window_max)
    return (windowed_image - window_min) / (window_max - window_min)

def normalize(image):
    """
    Normalize the image to the range [0, 1].

    Parameters
    ----------
    image : Torch.Tensor
        Image to normalize.

    Returns
    -------
    Torch.Tensor
        Normalized image.
    """
    if image.max() == image.min():
        return image
    return (image - image.min()) / (image.max() - image.min())