import cv2
import torch

from src.data.preprocessing import apply_window, normalize

class ApplyWindow:
    """Apply a window to the image."""
    def __init__(self, window_level=50, window_width=400):
        self.window_level = window_level
        self.window_width = window_width

    def __call__(self, image, mask):
        image = apply_window(image, self.window_level, self.window_width)
        return image, mask
    
class Normalize:
    """Normalize the image to the range [0, 1]."""
    def __call__(self, image, mask):
        image = normalize(image)
        return image, mask

class CropBorders:
    """Crop the borders of the image and mask by a specified size."""
    def __init__(self, crop_size=100):
        self.crop_size = crop_size

    def __call__(self, image, mask):
        h, w = image.shape
        # Ensure crop is within image bounds
        crop = min(self.crop_size, h // 2, w // 2)
        return image[crop:-crop, crop:-crop], mask[crop:-crop, crop:-crop]
    
class Resize:
    """Resize the image and mask to a specified size."""
    def __init__(self, size=(512, 512)):
        self.size = size

    def __call__(self, image, mask):
        return (
            cv2.resize(image, self.size, interpolation=cv2.INTER_CUBIC),
            cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
        )
    
class ToTensor:
    """Convert NumPy arrays to PyTorch tensors."""
    def __call__(self, image, mask):
        return (
            torch.tensor(image, dtype=torch.float32).unsqueeze(0),
            torch.tensor(mask, dtype=torch.long)
        )
    
class Compose:
    """Compose multiple transforms."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for transform in self.transforms:
            image, mask = transform(image, mask)
        return image, mask
    
# Standard transforms pipeline
standard_transforms = Compose([
    ApplyWindow(window_level=50, window_width=400),
    Normalize(),
    CropBorders(crop_size=100),
    Resize(size=(512, 512)),
    ToTensor()
])