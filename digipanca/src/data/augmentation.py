import albumentations as A

from albumentations.pytorch import ToTensorV2

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