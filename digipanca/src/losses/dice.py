import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss as MONAIDiceLoss
from monai.losses import DiceFocalLoss as MONAIDiceFocalLoss

#region Multiclass Dice Loss
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
        super().__init__()
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
        y_pred_softmax = F.softmax(y_pred.float(), dim=1)

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
#endregion

#region Weighted Dice Loss
class WeightedDiceLoss(nn.Module):
    def __init__(self, num_classes, include_background=True, reduction="mean"):
        super(WeightedDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.dice_loss = MONAIDiceLoss(
            include_background=True,
            reduction="mean"
        )

    def _compute_class_weights(self, y_true):
        """
        Compute class weights based on the number of pixels in each class.
        """
        class_counts = torch.bincount(y_true.flatten(), minlength=self.num_classes)
        total_pixels = class_counts.sum().float()
        weights = total_pixels / (class_counts + 1e-6)  # Add epsilon to avoid division by zero
        weights /= weights.sum()  # Normalize weights
        return weights

    def forward(self, y_pred, y_true):

        # Convert y_true to one-hot encoding
        y_true_onehot = F.one_hot(y_true, self.num_classes).permute(0, 3, 1, 2).float()

        # Compute class weights
        weights = self._compute_class_weights(y_true).to(y_pred.device)

        # Compute Dice loss
        loss = self.dice_loss(y_pred, y_true_onehot)
        return (loss * weights).sum()
#endregion

#region Dice Focal Loss
class DiceFocalLoss(nn.Module):
    def __init__(
        self,
        alpha=None,
        gamma=2.0,
        reduction='mean',
        include_background=False,
        lambda_dice=1.0,
        lambda_focal=1.0
    ):
        """
        Dice Focal Loss implementation using MONAI's DiceFocalLoss.

        Parameters
        ----------
        alpha : float, optional
            Focal loss alpha parameter.
        gamma : float, optional
            Focal loss gamma parameter.
        reduction : str, optional
            Reduction method for the loss. Default is 'mean'.
        include_background : bool, optional
            Whether to include the background class.
        lambda_dice : float, optional
            Weight for the Dice loss.
        lambda_focal : float, optional
            Weight for the Focal loss.
        """
        super(DiceFocalLoss, self).__init__()
        self.monai_dice_focal = MONAIDiceFocalLoss(
            alpha=alpha,
            gamma=gamma,
            reduction=reduction,
            include_background=include_background,
            lambda_dice=lambda_dice,
            lambda_focal=lambda_focal,
            softmax=True  # Assume y_pred is logits
        )

    def forward(self, y_pred, y_true):
        """
        Compute the Dice Focal Loss.

        Parameters
        ----------
        y_pred : torch.Tensor
            The model's predictions.
        y_true : torch.Tensor
            The ground truth labels.

        Returns
        -------
        torch.Tensor
            The computed Dice Focal Loss.
        """
        num_classes = y_pred.shape[1]

        # Convert y_true to one-hot encoding if necessary
        if y_true.dim() == 3:   # 2D: (B, H, W) → (B, C, H, W)
            y_true = F.one_hot(y_true, num_classes=num_classes).permute(0, 3, 1, 2).float()
        elif y_true.dim() == 4: # 3D: (B, D, H, W) → (B, C, D, H, W)
            y_true = F.one_hot(y_true, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

        return self.monai_dice_focal(y_pred, y_true)