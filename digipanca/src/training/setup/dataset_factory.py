import os
import json

from src.data.dataset2d import PancreasDataset2D
from src.data.dataset3d import PancreasDataset3D

def get_dataset(
    config,
    split_data=None,
    split_type='train',
    data_folder='train',
    transform=None,
    augment=None
):
    """Initialize dataset based on configuration.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary.
    split_data : dict, optional
        Dictionary containing the split data, by default None. If None, the
        dataset will be loaded without filtering by patient IDs.
    split_type : str, optional
        Split type (train/val/test), by default 'train'.
    data_folder : str, optional
        Data folder name, by default 'train'.
    transform : callable, optional
        Transform function, by default None.
    augment : callable, optional
        Augmentation function, by default None.

    Returns
    -------
    PancreasDataset2D or PancreasDataset3D
        Pancreas dataset object.
    """
    # Check that there is not augmentation for validation/test sets
    if (split_type != 'train' or data_folder != 'train') \
        and augment is not None:
        raise ValueError("Augmentations are only allowed for the training set.")
    # Ensure split type is valid
    if split_type not in ['train', 'val', 'test']:
        raise ValueError(f"Invalid split type: {split_type}")
    # Ensure data folder is valid
    if data_folder not in ['train', 'test']:
        raise ValueError(f"Invalid data folder: {data_folder}")
    
    # Get the patient IDs for the specified split type
    patient_ids = split_data.get(split_type, None) if split_data else None

    # data_folder is the folder name in the processed directory
    # e.g. 'train' or 'test'
    data_dir = os.path.join(config['data']['processed_dir'], data_folder)

    if config['data'].get('is_3d', False):
        return PancreasDataset3D(
            data_dir=data_dir,
            transform=transform,
            augment=augment,
            load_into_memory=config['data'].get('load_into_memory', False),
            patient_ids=patient_ids
        )
    else:
        return PancreasDataset2D(
            data_dir=data_dir,
            transform=transform,
            augment=augment,
            load_into_memory=config['data'].get('load_into_memory', False),
            patient_ids=patient_ids
        )