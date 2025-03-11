import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, class_weights=None):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.class_weights = class_weights
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=class_weights)

    def dice_loss(self, y_pred, y_true, smooth=1e-6):

        num_classes = y_pred.shape[1]
        y_true_one_hot = F.one_hot(y_true, num_classes).permute(0, 3, 1, 2).float()
        y_pred_softmax = F.softmax(y_pred, dim=1)

        dice_loss = 0.0
        for class_idx in range(num_classes):
            y_true_c = y_true_one_hot[:, class_idx, :, :]
            y_pred_c = y_pred_softmax[:, class_idx, :, :]
            intersection = (y_true_c * y_pred_c).sum(dim=(1, 2))
            union = y_true_c.sum(dim=(1, 2)) + y_pred_c.sum(dim=(1, 2))
            dice_loss += 1 - (2.0 * intersection + smooth) / (union + smooth)

        return dice_loss.mean()

    def forward(self, y_pred, y_true):
        ce_loss = self.cross_entropy_loss(y_pred, y_true)
        dice = self.dice_loss(y_pred, y_true)
        combined_loss = self.alpha * ce_loss + self.beta * dice
        return combined_loss, ce_loss, dice