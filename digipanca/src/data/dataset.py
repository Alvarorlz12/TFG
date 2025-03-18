import os
import torch
import numpy as np
import nibabel as nib

from torch.utils.data import Dataset
from collections import defaultdict
from nibabel.orientations import apply_orientation

from src.data.split_data import load_train_test_split
from src.data.transforms import Orientation

# Dataset class
class PancreasDataset(Dataset):
    def __init__(self, 
                 sample_dirs,
                 split_path="data/splits/train_test_split.json",
                 split_type="train",
                 transform=None,
                 augment=None
    ):
        """
        Initialize the Pancreas dataset. The dataset is loaded from the
        NIfTI files in the sample directories. The segmentation masks are
        created by combining the masks for pancreas, tumor, arteries, and veins.

        Parameters
        ----------
        sample_dirs : list
            List of directories containing the samples.
        split_path : str
            Path to the train-test split JSON file.
        split_type : str
            Type of split to load (train, val, or test). Default is train.
            Use "all" to load all samples.
        transform : callable
            A function/transform to apply to the image and mask.
        augment : callable
            A function/transform to apply data augmentation.
        """
        self.sample_dirs = sample_dirs
        self.transform = transform
        self.augment = augment
        self.reorient = Orientation(target_orientation=('R', 'P', 'S'))
        self.slices = defaultdict(list)

        # Load the train-test split if specified
        if split_type != "all":
            split_dict = load_train_test_split(split_path=split_path)
            if split_type not in split_dict:
                raise ValueError(f"Invalid split type: {split_type}")
            self.patient_ids = split_dict[split_type]
        else:
            self.patient_ids = [os.path.basename(p) for p in sample_dirs]

        print(f"📊 Loading dataset ({split_type})... {len(self.patient_ids)} patients found.")

        for sample_dir in sample_dirs:
            patient_id = os.path.basename(sample_dir)
            if patient_id not in self.patient_ids:
                continue

            # Load NIfTI files and create segmentation mask
            image, masks = self._load_nifti_slices(sample_dir)

            for i in range(image.shape[2]): # i is the slice index
                # Rotate for correct visualization
                img_slice = np.rot90(image[:, :, i], k=-1)
                masks_slice = np.rot90(masks[:, :, i], k=-1)

                self.slices[patient_id].append((img_slice, masks_slice))

        # Flatten slices for indexing
        self.flat_slices = [
            (img, mask, pid) for pid, slices in self.slices.items() 
            for img, mask in slices
        ]

        print(f"📊 Dataset loaded with {len(self.flat_slices)} slices.")

    def __len__(self):
        return len(self.flat_slices)
    
    def __getitem__(self, idx):
        img, mask, pid = self.flat_slices[idx]

        # Prevent negative stride
        img = img.copy()
        mask = mask.copy()

        # Apply transformations
        if self.transform is not None:
            img, mask = self.transform(img, mask)

        # Apply augmentations
        if self.augment is not None:
            # Augmentations require NumPy arrays
            if isinstance(img, torch.Tensor):
                img = img.numpy().squeeze(0)
            if isinstance(mask, torch.Tensor):
                mask = mask.numpy()

            img, mask = self.augment(img, mask)

        # Ensure img and mask are tensors
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img, dtype=torch.float32)
        else:
            img = img.clone().detach().to(dtype=torch.float32)

        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.long)
        else:
            mask = mask.clone().detach().to(dtype=torch.long)

        # Add channel dimension if missing: (H, W) -> (C, H, W)
        if len(img.shape) == 2:
            img = img.unsqueeze(0)

        return img, mask, pid
    
    def _load_nifti_slices(self, sample_dir):
        """
        Load the NIfTI files for a given patient and create a segmentation mask.
        The segmentation mask is created by combining the masks for pancreas, 
        tumor, arteries, and veins.

        Parameters:
        -----------
        sample_dir : str
            Path to the patient directory.

        Returns:
        --------
        image : np.ndarray
            3D image volume.
        masks : np.ndarray
            Combined segmentation mask.
        """
        patient_id = os.path.basename(os.path.normpath(sample_dir))
        image_path = os.path.join(sample_dir, "SEQ", f"CTport-{patient_id}.nii")
        mask_paths = {
            "pancreas": os.path.join(sample_dir, "SEG", 
                                     f"Pancreas-{patient_id}.nii"),
            "tumor": os.path.join(sample_dir, "SEG", f"Tumor-{patient_id}.nii"),
            "arteries": os.path.join(sample_dir, "SEG", 
                                     f"Arterias-{patient_id}.nii"),
            "veins": os.path.join(sample_dir, "SEG", f"Venas-{patient_id}.nii"),
        }

        image, transform = self.reorient(nib.load(image_path)) # Reorient the image
        num_slices = image.shape[2]
        masks = np.zeros_like(image)

        # Combine the segmentation masks
        for i, (_, path) in enumerate(mask_paths.items(), start=1):
            mask_data = nib.load(path).get_fdata()
            # Ensure the number of slices match the image
            mask_data = mask_data[:, :, :num_slices]
            # Binarize the mask
            mask_data = (mask_data > 0).astype(np.uint8)
            # Apply orientation transformation
            mask_data = apply_orientation(mask_data, transform)
            # Class label assignment: 1 for pancreas, 2 for tumor...
            masks[mask_data > 0] = i

        return image, masks