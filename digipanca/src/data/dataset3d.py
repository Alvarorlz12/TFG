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
    def __init__(self,
        data_dir,
        transform=None,
        augment=None,
        load_into_memory=False,
        patient_ids=None,
        verbose=True
    ):
        """
        Initialize the Pancreas 3D dataset.

        Parameters
        ----------
        data_dir : str
            Path to the directory containing the processed sub-volumes.
        transform : callable
            A function/transform to apply to the image and mask.
        augment : callable
            A function/transform to apply to the image and mask for augmentation.
        load_into_memory : bool, optional
            Whether to load the entire dataset into memory, by default False.
        patient_ids : list, optional
            List of patient IDs to filter the dataset, by default None.
        verbose : bool, optional
            Whether to print loading messages, by default True.
        """
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
            print(f"📊 Loading dataset... {len(self.image_filenames)} sub-volumes found.")

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
            if self.verbose:
                print(f"✅ Dataset loaded. {len(self.image_filenames)} sub-volumes loaded.")

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

        if self.augment:
            image, mask = self.augment(image, mask)

        # Add channel dimension to the image
        if image.dim() == 3:
            image = image.unsqueeze(0)

        return image.float(), mask.long(), patient_id
    
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

        # Filter sub-volumes for the given patient ID
        patient_subvolumes = [
            filename for filename, meta in self.metadata.items() if meta["patient_id"] == patient_id
        ]

        if not patient_subvolumes:
            raise ValueError(f"No sub-volumes found for patient ID: {patient_id}")

        # Sort sub-volumes by their slice indices
        patient_subvolumes.sort(key=lambda x: self.metadata[x]["slices"][0])

        # Initialize lists to store the full volume and masks
        full_volume = []
        full_masks = []

        for idx, filename in enumerate(patient_subvolumes):
            img_path = os.path.join(self.image_dir, filename)
            msk_path = os.path.join(self.mask_dir, self.metadata[filename]["mask_filename"])

            image = torch.tensor(np.load(img_path))   # (D, H, W)
            mask = torch.tensor(np.load(msk_path), dtype=torch.long)    # (D, H, W)

            padded = self.metadata[filename].get("padded", 0)
            if padded > 0:
                first_pad = padded // 2
                last_pad = padded - first_pad
                image = image[first_pad:-last_pad]
                mask = mask[first_pad:-last_pad]

            if idx == 0:
                # First sub-volume, no overlap
                full_volume.append(image)
                full_masks.append(mask)
            else:
                prev_end = self.metadata[patient_subvolumes[idx - 1]]["slices"][1]
                curr_start = self.metadata[filename]["slices"][0]
                # Check for overlap
                overlap = prev_end - curr_start + 1
                non_overlap = image[overlap:]  # Non-overlapping part
                non_overlap_mask = mask[overlap:]

                full_volume.append(non_overlap)
                full_masks.append(non_overlap_mask)

        # Concatenate all sub-volumes and masks along the depth dimension
        full_volume = torch.cat(full_volume, dim=0) # (D, H, W)
        full_masks = torch.cat(full_masks, dim=0)   # (D, H, W)

        # Apply transform if provided
        if self.transform:
            full_volume, full_masks = self.transform(full_volume, full_masks)

        # Add batch dimension to the volume and masks if needed
        if full_volume.dim() == 3:  # (D, H, W) to (B, C, D, H, W)
            full_volume = full_volume.unsqueeze(0).unsqueeze(0)
        elif full_volume.dim() == 4:  # (C, D, H, W) to (B, C, D, H, W)
            full_volume = full_volume.unsqueeze(0)
        if full_masks.dim() == 3:   # (D, H, W) to (B, D, H, W)
            full_masks = full_masks.unsqueeze(0)

        return full_volume, full_masks
    
    def get_patient_subset(self, patient_id):
        """
        """
        return PancreasDataset3D(
            data_dir=self.data_dir,
            transform=self.transform,
            load_into_memory=False,
            patient_ids=[patient_id]
        )
    
    def get_patient_subvolumes_slices(self, patient_id):
        """
        Get the slices of the sub-volumes for a specific patient.

        Parameters
        ----------
        patient_id : str
            Patient ID to retrieve the slices for.

        Returns
        -------
        list of tuples
            A list of tuples containing the start and end slice indices of each sub-volume.
        """
        # Filter sub-volumes for the given patient ID
        patient_subvolumes = [
            filename for filename, meta in self.metadata.items() if meta["patient_id"] == patient_id
        ]

        if not patient_subvolumes:
            raise ValueError(f"No sub-volumes found for patient ID: {patient_id}")

        # Sort sub-volumes by their slice indices
        patient_subvolumes.sort(key=lambda x: self.metadata[x]["slices"][0])

        # Get the slices of each sub-volume
        slices = [
            (self.metadata[filename]["slices"][0], self.metadata[filename]["slices"][1])
            for filename in patient_subvolumes
        ]

        return slices