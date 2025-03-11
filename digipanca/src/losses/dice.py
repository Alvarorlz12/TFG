import torch
import torch.nn as nn
import torch.nn.functional as F

class MulticlassDiceLoss(nn.Module):
    """Dice Loss for multiple classes."""

    def __init__(self, smooth=1e-6, ignore_background=True, class_weights=None):
        """
        Parameters
        ----------
        smooth : float, optional
            Smoothing factor to avoid division by zero.
        ignore_background : bool, optional
            Whether to ignore the background class.
        class_weights : list, optional
            Weights for each class.
        """
        super(MulticlassDiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_background = ignore_background
        self.class_weights = class_weights

    def forward(self, y_pred, y_true):
        """
        Compute the Dice loss for multiple classes.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted class probabilities.
        y_true : torch.Tensor
            True class labels.

        Returns
        -------
        torch.Tensor
            Dice loss.
        """
        num_classes = y_pred.shape[1]

        # Convert y_true to one-hot encoding
        if len(y_true.shape) == 4:
            y_true_one_hot = F.one_hot(y_true.long(), num_classes).permute(0, 4, 1, 2, 3).float()
        else:
            y_true_one_hot = F.one_hot(y_true.long(), num_classes).permute(0, 3, 1, 2).float()

        # Apply softmax to y_pred
        y_pred_softmax = F.softmax(y_pred, dim=1)

        # Initialize loss
        dice_loss = 0.0
        total_weight = 0.0

        # Initialize class range (0 if background is not ignored, 1 otherwise)
        start_class = 1 if self.ignore_background else 0

        for class_idx in range(start_class, num_classes):
            y_true_class = y_true_one_hot[:, class_idx, ...]
            y_pred_class = y_pred_softmax[:, class_idx, ...]

            # Calculate intersection and union
            intersection = torch.sum(y_true_class * y_pred_class)
            union = torch.sum(y_true_class) + torch.sum(y_pred_class)

            # Calculate Dice coefficient
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

            # Apply class weights
            weight = 1.0
            if self.class_weights is not None:
                weight_idx = class_idx - start_class  # Fix weight indexing
                weight = self.class_weights[weight_idx]

            # Update loss
            dice_loss += (1.0 - dice.mean()) * weight
            total_weight += weight

        # Normalize loss
        if total_weight > 0:
            return dice_loss / total_weight
        return dice_loss / (num_classes - start_class)