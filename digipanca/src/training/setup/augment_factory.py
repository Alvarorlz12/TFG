from src.data.augmentation import build_augmentations_from_config
from src.data.augmentation_3d import build_3d_augmentations_from_config

def get_augment(config):
    """
    Initialize augmentations based on configuration.

    Parameters
    ----------
    config : Dict
        The configuration dictionary containing augmentation settings.

    Returns
    -------
    Augment
        The initialized augmentation pipeline.
    """
    augment_config = config.get('augmentations', None)
    is_3d = config['data'].get('is_3d', False)
    if is_3d:
        return build_3d_augmentations_from_config(augment_config)
    else:
        return build_augmentations_from_config(augment_config)