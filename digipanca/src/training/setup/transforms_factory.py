from src.data.transforms import build_transforms_from_config

def get_transforms(config):
    """Initialize transforms based on configuration."""
    transform_config = config.get('transforms', None)
    return build_transforms_from_config(transform_config)