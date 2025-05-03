import os
import torch
import json
import numpy as np

from PIL import Image
from torch.utils.data import Dataset

class PancreasDataset2D(Dataset):
    """
    Pancreas dataset for 2D slices. The dataset is loaded from the .npy 
    (volumes) and .png (masks) files in the processed directory. 
    """
    def __init__(
        self,
        data_dir,
        transform=None,
        augment=None,
        load_into_memory=False,
        patient_ids=None,
        verbose=True
    ):
        """
        Initialize the Pancreas 2D dataset.

        Parameters
        ----------
        data_dir : str
            Path to the directory containing the processed slices.
        transform : callable
            A function/transform to apply to the image and mask.
        augment : callable
            A function/transform to apply for data augmentation.
        load_into_memory : bool, optional
            Whether to load the entire dataset into memory, by default False.
        patient_ids : list, optional
            List of patient IDs to filter the dataset, by default None.
        verbose : bool, optional
            Whether to print loading messages, by default True.
        """
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, "images")
        self.mask_dir = os.path.join(data_dir, "masks")
        self.transform = transform
        self.augment = augment
        self.load_into_memory = load_into_memory
        self.verbose = verbose

        metadata_path = os.path.join(data_dir, "metadata.json")
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        # Filter metadata by patient IDs if provided
        if patient_ids is not None:
            self.metadata = {
                k: v for k, v in self.metadata.items() if v["patient_id"] in patient_ids
            }

        # Get the list of image filenames
        self.image_filenames = sorted(self.metadata.keys())

        if self.verbose:
            print(f"📊 Loading dataset... {len(self.image_filenames)} slices found.")

        # Load the data into memory if specified
        if self.load_into_memory:
            self.data = {
                filename: (
                    torch.tensor(np.load(os.path.join(self.image_dir, filename))),
                    torch.from_numpy(
                        np.array(
                            Image.open(
                                os.path.join(
                                    self.mask_dir, 
                                    self.metadata[filename]["mask_filename"]
                                )
                            ).convert("L")
                        )
                    ).long(),
                    self.metadata[filename]["patient_id"],
                    self.metadata[filename]["slice_index"]
                )
                for filename in self.image_filenames
            }
            if self.verbose:
                print(f"✅ Dataset loaded. {len(self.image_filenames)} slices loaded.")

    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        filename = self.image_filenames[idx]

        if self.load_into_memory:
            image, mask, patient_id, slice_index = self.data[filename]
        else:
            image_path = os.path.join(self.image_dir, filename)
            mask_path = os.path.join(self.mask_dir, self.metadata[filename]["mask_filename"])
            image = torch.tensor(np.load(image_path))
            mask = torch.from_numpy(np.array(
                Image.open(mask_path).convert("L")
            )).long()
            patient_id = self.metadata[filename]["patient_id"]
            slice_index = self.metadata[filename]["slice_index"]

        if self.transform:
            image, mask = self.transform(image, mask)

        # Apply augmentations if specified
        if self.augment is not None:
            # Augmentations require NumPy arrays
            if isinstance(image, torch.Tensor):
                image = image.numpy()
                # If it has been transformed, it may have a channel dimension
                if image.ndim == 3:
                    image = image.squeeze(0)
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

        # Add channel dimension to the image
        if image.dim() == 2:
            image = image.unsqueeze(0)

        return image, mask, patient_id
    
    def get_volume_slice_idx(self, idx):
        """
        Get the real slice index in the original volume.

        Parameters
        ----------
        idx : int
            Index of the sample in the dataset.

        Returns
        -------
        int
            Real slice index in the original volume.
        """
        filename = self.image_filenames[idx]
        return self.metadata[filename]["slice_index"]
    
    def get_patient_volume(self, patient_id):
        """
        Get the full volume (images and masks) for a specific patient.

        Parameters
        ----------
        patient_id : str
            Patient ID to retrieve the volume for.

        Returns
        -------
        tuple
            A tuple containing:
            - volume (torch.Tensor): Tensor of shape (B, C, D, H, W).
            - masks (torch.Tensor): Corresponding masks tensor of shape (B, D, H, W).
        """

        # Filter and sort slices for the given patient ID
        patient_slices = sorted(
            [filename for filename, meta in self.metadata.items() if meta["patient_id"] == patient_id],
            key=lambda x: self.metadata[x]["slice_index"]
        )

        # Load images and masks for the patient
        if self.load_into_memory:
            patient_volume = [self.data[filename][0].unsqueeze(0) for filename in patient_slices]
            patient_masks = [self.data[filename][1] for filename in patient_slices]
        else:
            patient_volume, patient_masks = [], []
            for filename in patient_slices:
                image_path = os.path.join(self.image_dir, filename)
                mask_path = os.path.join(self.mask_dir, self.metadata[filename]["mask_filename"])
                image = torch.tensor(np.load(image_path))
                mask = torch.tensor(np.array(Image.open(mask_path).convert("L")), dtype=torch.long)
                if self.transform is not None:
                    image, mask = self.transform(image, mask)
                if image.dim() == 2:
                    image = image.unsqueeze(0)
                patient_volume.append(image)  # Add channel dimension
                patient_masks.append(mask)

        # Stack slices along the depth dimension (D)
        patient_volume = torch.stack(patient_volume, dim=0)  # Shape: (D, C, H, W)
        patient_masks = torch.stack(patient_masks, dim=0)    # Shape: (D, H, W)

        # Add batch dimension (B)
        patient_volume = patient_volume.unsqueeze(0)  # Shape: (1, D, C, H, W)
        patient_masks = patient_masks.unsqueeze(0)    # Shape: (1, D, H, W)

        # Permute to (B, C, D, H, W) and (B, 1, D, H, W)
        patient_volume = patient_volume.permute(0, 2, 1, 3, 4)  # (B, C, D, H, W)

        return patient_volume, patient_masks
    
    def get_patient_subset(self, patient_id):
        """
        """
        return PancreasDataset2D(
            data_dir=self.data_dir,
            transform=self.transform,
            augment=self.augment,
            load_into_memory=False,
            patient_ids=[patient_id]
        )