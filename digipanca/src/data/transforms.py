import cv2
import torch
import numpy as np
import nibabel as nib

from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform, apply_orientation

from src.data.preprocessing import apply_window, normalize

#region TRANSFORMS
class ApplyWindow:
    """Apply a window to the image (2D or 3D)."""
    def __init__(self, window_level=50, window_width=400):
        self.window_level = window_level
        self.window_width = window_width

    def __call__(self, image, mask):
        image = apply_window(image, self.window_level, self.window_width)
        return image, mask
    
class Normalize:
    """Normalize the image (2D or 3D) to the range [0, 1]."""
    def __call__(self, image, mask):
        image = normalize(image)
        return image, mask

class CropBorders:
    """
    Crop the borders of the image and mask by a specified size (only in 
    H and W dimensions)
    """
    def __init__(self, crop_size=100):
        self.crop_size = crop_size

    def __call__(self, image, mask):        
        if image.ndim == 2:
            h, w = image.shape
            # Ensure crop is within image bounds
            crop = min(self.crop_size, h // 2, w // 2)
            return image[crop:-crop, crop:-crop], mask[crop:-crop, crop:-crop]
        
        elif image.ndim == 3:
            _, h, w = image.shape   # Assuming image is in (D, H, W) format
            # Ensure crop is within image bounds
            crop = min(self.crop_size, h // 2, w // 2)
            return image[:, crop:-crop, crop:-crop], mask[:, crop:-crop, crop:-crop]
    
class Resize:
    """
    Resize the image and mask to a specified size (only in H and W 
    dimensions, keeping D fixed if 3D).
    """
    def __init__(self, size=(512, 512)):
        self.size = size    # Target size (H, W)

    def __call__(self, image, mask):
        if isinstance(image, torch.Tensor):
            image = image.numpy()
            mask = mask.numpy()
        
        if image.ndim == 2: # 2D image
            return (
                cv2.resize(image, self.size, interpolation=cv2.INTER_CUBIC),
                cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
            )
        
        elif image.ndim == 3:   # 3D image
            d, h, w = image.shape
            resized_image = np.zeros((d, *self.size), dtype=image.dtype)
            resized_mask = np.zeros((d, *self.size), dtype=mask.dtype)

            for i in range(d):  # Resize each slice individually
                resized_image[i, :, :] = cv2.resize(
                    image[i, :, :], self.size, interpolation=cv2.INTER_CUBIC
                )
                resized_mask[i, :, :] = cv2.resize(
                    mask[i, :, :], self.size, interpolation=cv2.INTER_NEAREST
                )

            return resized_image, resized_mask
    
class Orientation:
    """Orient the image to a specified orientation."""
    def __init__(self, target_orientation=('R', 'A', 'S')):
        self.target_orientation = axcodes2ornt(target_orientation)

    def __call__(self, image, affine=None):
        if isinstance(image, nib.Nifti1Image):
            image = image.get_fdata().astype(np.float64)
            affine = image.affine if affine is None else affine
        else:
            if affine is None:
                raise ValueError("Affine matrix is required for non-NIfTI images.")
            
        current_orientation = io_orientation(affine)
        transform = ornt_transform(current_orientation, self.target_orientation)
        image = apply_orientation(image, transform)
        return image, transform

class CropROI:
    """
    Crop the image and mask using predefined ROI values (h_min, h_max, w_min, w_max).

    Parameters
    ----------
    h_min : int
        Minimum height index for cropping.
    h_max : int
        Maximum height index for cropping.
    w_min : int
        Minimum width index for cropping.
    w_max : int
        Maximum width index for cropping.
    """
    def __init__(self, h_min, h_max, w_min, w_max):
        self.h_min = h_min
        self.h_max = h_max
        self.w_min = w_min
        self.w_max = w_max

    def __call__(self, image, mask):
        if image.ndim == 2:
            return (
                image[self.h_min:self.h_max, self.w_min:self.w_max], 
                mask[self.h_min:self.h_max, self.w_min:self.w_max]
            )
        elif image.ndim == 3:
            return (
                image[:, self.h_min:self.h_max, self.w_min:self.w_max], 
                mask[:, self.h_min:self.h_max, self.w_min:self.w_max]
            )

class ToTensor:
    """Convert NumPy arrays to PyTorch tensors."""
    def __call__(self, image, mask):
        if image.ndim == 2:
            return (
                torch.tensor(image, dtype=torch.float32).unsqueeze(0),
                torch.tensor(mask, dtype=torch.long)
            )
        elif image.ndim == 3:
            image = torch.from_numpy(image) if isinstance(image, np.ndarray) else image
            mask = torch.from_numpy(mask) if isinstance(mask, np.ndarray) else mask

            return (
                image.clone().detach().float().unsqueeze(0),
                mask.clone().detach().long()
            )
    
class Compose:
    """Compose multiple transforms."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for transform in self.transforms:
            result = transform(image, mask)
            
            if result is None:
                raise ValueError(f"{transform.__class__.__name__} returned None")
            
            try:
                image, mask = result
            except ValueError:
                raise ValueError(f"{transform.__class__.__name__} returned an unexpected output: {result}")

        return image, mask
    
#region PIPELINES
    
# Mapping of transformations to their names
_TRANSFORMS = {
    'ApplyWindow': ApplyWindow,
    'Normalize': Normalize,
    'CropBorders': CropBorders,
    'Resize': Resize,
    'Orientation': Orientation,
    'CropROI': CropROI,
    'ToTensor': ToTensor,
    'Compose': Compose
}

def build_transforms_from_config(config):
    """
    Build a transforms pipeline from a configuration list of dictionaries.

    Parameters
    ----------
    config : List[Dict]
        The configuration list of dictionaries.

    Returns
    -------
    Compose
        The transforms pipeline.
    """
    if config is None:
        return None
    
    transforms_list = []
    for transform in config:
        name, params = list(transform.items())[0]  # Get the name and parameters
        transform_class = _TRANSFORMS.get(name)

        if transform_class:
            transform_instance = transform_class(**(params or {}))
            transforms_list.append(transform_instance)
        else:
            print(f"⚠️ Unknown transform: {name}")

    return Compose(transforms_list)

# Standard transforms pipeline
standard_transforms = Compose([
    ApplyWindow(window_level=50, window_width=400),
    Normalize(),
    CropBorders(crop_size=120),
    Resize(size=(512, 512)),
    ToTensor()
])