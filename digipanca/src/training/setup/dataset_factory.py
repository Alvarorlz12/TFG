import os

from src.data.dataset2d import PancreasDataset2D
from src.data.dataset3d import PancreasDataset3D

def get_dataset(config, split_type='train', transform=None, augment=None):
    """Initialize dataset based on configuration.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary.
    split_type : str, optional
        Split type (train/val/test), by default 'train'.
    transform : callable, optional
        Transform function, by default None.
    augment : callable, optional
        Augmentation function, by default None.

    Returns
    -------
    PancreasDataset or PancreasDataset3D
        Pancreas dataset object.
    """
    # Check that there is not augmentation for validation/test sets
    if split_type != 'train' and augment is not None:
        raise ValueError("Augmentations are only allowed for the training set.")
    # Ensure split type is valid
    if split_type not in ['train', 'val', 'test']:
        raise ValueError(f"Invalid split type: {split_type}")
    
    data_dir = os.path.join(config['data']['processed_dir'], split_type)
    if config['data'].get('is_3d', False):
        return PancreasDataset3D(
            data_dir=data_dir,
            transform=transform,
            load_into_memory=config['data'].get('load_into_memory', False)
        )
    else:
        return PancreasDataset2D(
            data_dir=data_dir,
            transform=transform,
            augment=augment,
            load_into_memory=config['data'].get('load_into_memory', False),
        )