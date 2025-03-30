import os
import torch
import json
import numpy as np

from torch.utils.data import Dataset

class PancreasDataset3D(Dataset):
    """
    Pancreas dataset for 3D volumes. The dataset is loaded from the .npy files
    in processed directory. The volume may be divided into sub-volumes of a
    specified size, with a specified stride.
    """
    def __init__(self, data_dir, transform=None, load_into_memory=False):
        """
        Initialize the Pancreas 3D dataset.

        Parameters
        ----------
        data_dir : str
            Path to the directory containing the processed sub-volumes.
        transform : callable
            A function/transform to apply to the image and mask.
        load_into_memory : bool, optional
            Whether to load the entire dataset into memory, by default False.
        """
        self.image_dir = os.path.join(data_dir, "images")
        self.mask_dir = os.path.join(data_dir, "masks")
        self.transform = transform
        self.load_into_memory = load_into_memory

        metadata_path = os.path.join(data_dir, "metadata.json")
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        # Get the list of image filenames
        self.image_filenames = sorted(self.metadata.keys())

        # Load the data into memory if specified
        if self.load_into_memory:
            self.data = {
                filename: (
                    torch.tensor(np.load(os.path.join(self.image_dir, filename))),
                    torch.tensor(np.load(os.path.join(self.mask_dir, self.metadata[filename]["mask_filename"]))),
                    self.metadata[filename]["patient_id"],
                    self.metadata[filename]["slices"]
                )
                for filename in self.image_filenames
            }

    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        filename = self.image_filenames[idx]

        if self.load_into_memory:
            image, mask, patient_id, slices = self.data[filename]
        else:
            image_path = os.path.join(self.image_dir, filename)
            mask_path = os.path.join(self.mask_dir, self.metadata[filename]["mask_filename"])
            image = torch.tensor(np.load(image_path))
            mask = torch.tensor(np.load(mask_path))
            patient_id = self.metadata[filename]["patient_id"]
            slices = self.metadata[filename]["slices"]

        if self.transform:
            image, mask = self.transform(image, mask)

        # Add channel dimension to the image
        if image.dim() == 3:
            image = image.unsqueeze(0)

        return image, mask, patient_id
    
    def get_volume_slice_idx(self, idx, slice_idx):
        """
        Get the real slice index in the original volume.

        Parameters
        ----------
        idx : int
            Index of the sample in the dataset.
        slice_idx : int
            Index of the slice in the sub-volume.

        Returns
        -------
        int
            Real slice index in the original volume.
        """
        filename = self.image_filenames[idx]
        slices = self.metadata[filename]["slices"]
        padded = self.metadata[filename].get("padded", 0) // 2
        return slices[0] - padded + slice_idx
    
    def get_subvolume_slice_idx(self, idx, slice_idx):
        """
        Get the slice index in the sub-volume.

        Parameters
        ----------
        idx : int
            Index of the sample in the dataset.
        slice_idx : int
            Index of the slice in the original volume.

        Returns
        -------
        int
            Slice index in the sub-volume.
        """
        filename = self.image_filenames[idx]
        slices = self.metadata[filename]["slices"]
        padded = self.metadata[filename].get("padded", 0) // 2
        return slices[0] + padded + slice_idx
        
