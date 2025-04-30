import torchio as tio

_AUGMENTATIONS_3D = {
    "RandomAffine": tio.RandomAffine,
    "RandomFlip": tio.RandomFlip,
    "RandomNoise": tio.RandomNoise,
    "RandomBlur": tio.RandomBlur
}

def build_3d_augmentations_from_config(config):
    """
    Build a 3D augmentation pipeline from a configuration list of dictionaries.

    Parameters
    ----------
    config : List[Dict]
        The configuration list of dictionaries.

    Returns
    -------
    Callable
        The TorchIO transformation pipeline.
    """
    if config is None:
        return None

    augmentations_list = []
    for aug in config:
        name, params = list(aug.items())[0]
        aug_class = _AUGMENTATIONS_3D.get(name)

        if aug_class:
            aug_instance = aug_class(**(params or {}))
            augmentations_list.append(aug_instance)
        else:
            print(f"⚠️ Unknown 3D augmentation: {name}")

    return Augment3D(augmentations_list)

class Augment3D:
    """Apply 3D data augmentation."""
    def __init__(self, augmentations):
        self.augmentations = tio.Compose(augmentations)

    def __call__(self, image, mask):
        """
        Apply the augmentations to the image and mask.

        Parameters
        ----------
        image : torch.Tensor
            The input image tensor.
        mask : torch.Tensor
            The input mask tensor.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The augmented image and mask tensors.
        """
        # Ensure image and mask are 4D tensors (C, D, H, W)
        if image.dim() == 3:
            image = image.unsqueeze(0)  # Add C dimension
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)    # Add C dimension

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=image),
            mask=tio.LabelMap(tensor=mask)
        )
        augmented_subject = self.augmentations(subject)
        return (
            augmented_subject["image"].data,
            augmented_subject["mask"].data.squeeze(0)  # Remove C dimension
        )