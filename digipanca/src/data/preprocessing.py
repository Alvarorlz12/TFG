import numpy as np

def apply_window(image, window_level, window_width):
    """
    Apply a window to the image, which is commonly used in medical imaging to
    visualize the image with a specific intensity range. It changes the contrast
    (level) and brightness (window) of the image.

    Parameters
    ----------
    image : np.ndarray
        Image to apply the window to.
    window_level : int
        Window level.
    window_width : int
        Window width.

    Returns
    -------
    np.ndarray
        Image with window applied.
    """
    window_min = window_level - (window_width / 2)
    window_max = window_level + (window_width / 2)
    windowed_image = np.clip(image, window_min, window_max)
    return (windowed_image - window_min) / (window_max - window_min)

def normalize(image):
    """
    Normalize the image to the range [0, 1].

    Parameters
    ----------
    image : Torch.Tensor
        Image to normalize.

    Returns
    -------
    Torch.Tensor
        Normalized image.
    """
    if image.max() == image.min():
        return image
    return (image - image.min()) / (image.max() - image.min())