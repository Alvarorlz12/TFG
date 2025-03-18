import albumentations as A

from albumentations.pytorch import ToTensorV2

_AUGMENTATIONS = {
    "RandomCrop": A.RandomCrop,
    "Affine": A.Affine,
    "RandomBrightnessContrast": A.RandomBrightnessContrast,
    "GaussianBlur": A.GaussianBlur,
    "ElasticTransform": A.ElasticTransform,
    "GridDistortion": A.GridDistortion,
    "ToTensorV2": ToTensorV2,
}

def build_augmentations_from_config(config):
    """
    Build an augmentations pipeline from a configuration list of dictionaries.

    Parameters
    ----------
    config : List[Dict]
        The configuration list of dictionaries.

    Returns
    -------
    Augment
        The augmentation pipeline.
    """
    if config is None:
        return None
    
    augmentations_list = []
    for aug in config:
        name, params = list(aug.items())[0]  # Get the name and parameters
        aug_class = _AUGMENTATIONS.get(name)

        if aug_class:
            aug_instance = aug_class(**(params or {}))
            augmentations_list.append(aug_instance)
        else:
            print(f"⚠️ Unknown augmentation: {name}")
    
    return Augment(augmentations_list)

standard_augmentations = [
    A.Affine(scale=(0.95, 1.05), translate_percent=(0.02, 0.02),
             rotate=(-10, 10), p=0.2),
    A.RandomBrightnessContrast(brightness_limit=0.2,
                               contrast_limit=0.2, p=0.3),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.ElasticTransform(alpha=1.0, sigma=50, p=0.3),
    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
    ToTensorV2(),
]

class Augment:
    """Apply data augmentation."""
    def __init__(self, augmentations):
        self.augmentations = A.Compose(augmentations)
    
    def __call__(self, image, mask):
        augmented = self.augmentations(image=image, mask=mask)
        return augmented["image"], augmented["mask"]