import os
import json
import numpy as np
import nibabel as nib

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

def pad_volume(volume, target_size):
    """Pad a 3D volume along the Z-axis to match the target size."""
    current_size = volume.shape[2]
    if current_size >= target_size:
        return volume  # No padding needed

    pad_width = target_size - current_size
    pad_before = pad_width // 2  # Padding before
    pad_after = pad_width - pad_before  # Padding after

    return np.pad(volume, ((0, 0), (0, 0), (pad_before, pad_after)), mode="constant")

def process_patient(
    patient_dir,
    output_dir,
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
    image, transform = reorient(image_nii) # Reorient the image

    image = np.rot90(image, k=-1, axes=rotation_axes)   # Rotate the image
    image = image[h_min:h_max, w_min:w_max, :]  # Apply cropping

    masks = np.zeros_like(image)

    # Combine the segmentation masks
    for i, (_, mask_path) in enumerate(mask_paths.items(), start=1):
        mask_nii = nib.load(mask_path)

        # Ensure the number of slices match the image
        if mask_nii.shape[2] != image.shape[2]:
            mask_nii = resample_from_to(mask_nii, image_nii, order=0)
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

    num_slices = image.shape[2]

    padded = 0
    if num_slices < subvolume_size:
        print(f"⚠️ {patient_id} has less slices than the subvolume size. Adding padding...")
        image = pad_volume(image, subvolume_size)
        masks = pad_volume(masks, subvolume_size)
        padded = subvolume_size - num_slices
        num_slices = subvolume_size

    # Saving directories
    output_img_dir = os.path.join(output_dir, "images")
    output_mask_dir = os.path.join(output_dir, "masks")
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    metadata = {}   # Metadata for the sub-volumes

    # Generate sub-volumes
    subvolume_idx = 0
    for start in range(0, num_slices - subvolume_size + 1, subvolume_stride):
        image_subvol = image[:, :, start:start + subvolume_size]
        mask_subvol = masks[:, :, start:start + subvolume_size]

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
        image_subvol = image[:, :, -subvolume_size:]
        mask_subvol = masks[:, :, -subvolume_size:]

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

    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    return subvolume_idx

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