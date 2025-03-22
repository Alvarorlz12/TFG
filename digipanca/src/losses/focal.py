import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import FocalLoss as MONAIFocalLoss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        """
        Focal Loss implementation using MONAI's FocalLoss.

        Parameters
        ----------
        gamma : float
            Focal loss gamma parameter.
        reduction : str
            Reduction method for the loss. Default is 'mean'.
        """
        super(FocalLoss, self).__init__()
        self.monai_focal_loss = MONAIFocalLoss(gamma=gamma, reduction=reduction)

    def forward(self, y_pred, y_true):
        """
        Compute the Focal Loss.

        Parameters
        ----------
        y_pred : torch.Tensor
            The model's predictions.
        y_true : torch.Tensor
            The ground truth labels.

        Returns
        -------
        torch.Tensor
            The computed Focal Loss.
        """
        num_classes = y_pred.shape[1]

        # Convert y_true to one-hot encoding if necessary
        if y_true.dim() == 3:  # (B, H, W) → (B, C, H, W)
            y_true = F.one_hot(y_true, num_classes).permute(0, 3, 1, 2).float()

        return self.monai_focal_loss(y_pred, y_true)