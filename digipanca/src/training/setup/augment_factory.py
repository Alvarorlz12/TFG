from src.data.augmentation import build_augmentations_from_config

def get_augment(config):
    """Initialize augmentations based on configuration."""
    augment_config = config.get('augmentations', None)
    return build_augmentations_from_config(augment_config)