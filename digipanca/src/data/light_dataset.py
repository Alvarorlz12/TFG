import os
import torch
import numpy as np
import nibabel as nib

from torch.utils.data import Dataset
from nibabel.orientations import apply_orientation

from src.data.split_data import load_train_test_split
from src.data.transforms import Orientation

class LightPancreasDataset(Dataset):
    """
    PyTorch dataset for the Pancreas dataset. This is an optimized version
    of the `full_dataset.FullPancreasDataset` class that loads the NIfTI files
    and creates the segmentation masks on-the-fly. This reduces the memory
    footprint and speeds up the data loading process.
    """
    def __init__(
        self,
        data_dir,
        split_file,
        split_type,
        transform=None,
        augment=None
    ):
        """
        Dataset class for the Pancreas dataset. The dataset loads the NIfTI
        files from the data directory and saves the patient IDs and number of
        slices for each patient. Then, when getting an item, it loads the
        corresponding NIfTI files and applies the transformations and 
        augmentations.

        Parameters
        ----------
        data_dir : str
            The directory containing the NIfTI files.
        split_file : str
            The path to the split file.
        split_type : str
            The type of split to load (train, val, test, or all).
        transform : callable
            A function/transform to apply to the image and mask.
        augment : callable
            A function/transform to apply data augmentation.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.augment = augment
        self.reorient = Orientation(target_orientation=('R', 'P', 'S'))
        self.slices = []

        # Load the train-test split
        if split_type != "all":
            split_dict = load_train_test_split(split_file)
            if split_type not in split_dict:
                raise ValueError(f"Invalid split type: {split_type}")
            self.patient_ids = split_dict[split_type]
        else:
            self.patient_ids = [os.path.basename(p) for p in os.listdir(data_dir)]

        print(f"📊 Loading dataset ({split_type})... {len(self.patient_ids)} patients found.")

        for patient_id in self.patient_ids:
            image_path = os.path.join(
                self.data_dir, patient_id, "SEQ", f"CTport-{patient_id}.nii"
            )
            num_slices = nib.load(image_path).shape[2]
            for slice_idx in range(num_slices):
                self.slices.append((patient_id, slice_idx))

        print(f"📊 Dataset loaded with {len(self.slices)} slices.")

    def __len__(self):
        return len(self.slices)
    
    def __getitem__(self, idx):
        patient_id, slice_idx = self.slices[idx]

        # Load the NIfTI files (slice and mask)
        image, mask = self._load_nifti_slices(patient_id, slice_idx)

        # Prevent negative stride
        image = image.copy()
        mask = mask.copy()

        # Apply transformations
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        # Apply augmentations
        if self.augment is not None:
            # Augmentations require NumPy arrays
            if isinstance(image, torch.Tensor):
                image = image.numpy().squeeze(0)
            if isinstance(mask, torch.Tensor):
                mask = mask.numpy()

            image, mask = self.augment(image, mask)

        # Ensure image and mask are tensors
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32).cpu()
        else:
            image = image.clone().detach().to(dtype=torch.float32).cpu()

        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.long).cpu()
        else:
            mask = mask.clone().detach().to(dtype=torch.long).cpu()

        # Add channel dimension if missing: (H, W) -> (C, H, W)
        if len(image.shape) == 2:
            image = image.unsqueeze(0)

        return image, mask, patient_id

    def _load_nifti_slices(self, patient_id, slice_idx=0):
        """
        Load the NIfTI files for a given patient and create a segmentation mask
        for the specified slice index. The segmentation mask is created by
        combining the masks for pancreas, tumor, arteries, and veins.

        Parameters:
        -----------
        patient_id : str
            The patient ID.
        slice_idx : int
            The slice index to load. Default is 0.

        Returns:
        --------
        image : np.ndarray
            The image data (slice of the CT scan).
        masks : np.ndarray
            Combined segmentation mask.
        """
        patient_dir = os.path.join(self.data_dir, patient_id)
        image_path = os.path.join(patient_dir, "SEQ", f"CTport-{patient_id}.nii")
        mask_paths = {
            "pancreas": os.path.join(patient_dir, "SEG", 
                                     f"Pancreas-{patient_id}.nii"),
            "tumor": os.path.join(patient_dir, "SEG", f"Tumor-{patient_id}.nii"),
            "arteries": os.path.join(patient_dir, "SEG", 
                                     f"Arterias-{patient_id}.nii"),
            "veins": os.path.join(patient_dir, "SEG", f"Venas-{patient_id}.nii"),
        }

        image, transform = self.reorient(nib.load(image_path)) # Reorient the image
        image = image[:, :, slice_idx]
        masks = np.zeros_like(image)

        # Combine the segmentation masks
        for i, (_, path) in enumerate(mask_paths.items(), start=1):
            mask_data = nib.load(path).get_fdata()
            # Apply orientation transformation
            mask_data = apply_orientation(mask_data, transform)
            # Get the slice
            mask_slice = mask_data[:, :, slice_idx]
            # Binary mask
            mask_slice = (mask_slice > 0).astype(np.uint8)
            # Combine the masks
            masks[mask_slice > 0] = i

        # Rotate the image and mask for visualization
        image = np.rot90(image, k=-1)
        masks = np.rot90(masks, k=-1)

        return image, masks