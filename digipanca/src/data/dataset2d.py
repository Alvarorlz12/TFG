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
        load_into_memory=False
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
        """
        self.image_dir = os.path.join(data_dir, "images")
        self.mask_dir = os.path.join(data_dir, "masks")
        self.transform = transform
        self.augment = augment
        self.load_into_memory = load_into_memory

        metadata_path = os.path.join(data_dir, "metadata.json")
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        # Get the list of image filenames
        self.image_filenames = sorted(self.metadata.keys())

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
                # It it has been transformed, it may have a channel dimension
                if image.ndim == 3:
                    image = image.squeeze(0)
            if isinstance(mask, torch.Tensor):
                mask = mask.numpy()

            image, mask = self.augment(image, mask)

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