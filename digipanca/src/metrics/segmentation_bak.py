import torch
from typing import Dict

class SegmentationMetrics:
    """
    Class for computing segmentation metrics for pancreas segmentation.
    """
    
    @staticmethod
    def dice_coefficient(y_pred, y_true, smooth=1e-6):
        """
        Compute Dice coefficient.
        
        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted segmentation mask
        y_true : torch.Tensor
            Ground truth segmentation mask
        smooth : float, optional
            Smoothing factor to avoid division by zero
        
        Returns
        -------
        torch.Tensor
            Dice coefficient
        """
        # Flatten the tensors
        if y_pred.dim() == 4:
            # Multi-class case
            n_classes = y_pred.size(1)
            dice_scores = []
            
            # Calculate dice for each class
            for i in range(n_classes):
                pred_class = y_pred[:, i, :, :]
                true_class = y_true[:, i, :, :]
                
                intersection = torch.sum(pred_class * true_class)
                union = torch.sum(pred_class) + torch.sum(true_class)
                dice = (2.0 * intersection + smooth) / (union + smooth)
                dice_scores.append(dice)
                
            return torch.mean(torch.stack(dice_scores))
        else:
            # Binary case
            intersection = torch.sum(y_pred * y_true)
            union = torch.sum(y_pred) + torch.sum(y_true)
            dice = (2.0 * intersection + smooth) / (union + smooth)
            return dice
    
    @staticmethod
    def iou_score(y_pred, y_true, smooth=1e-6):
        """
        Compute IoU (Jaccard Index).
        
        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted segmentation mask
        y_true : torch.Tensor
            Ground truth segmentation mask
        smooth : float, optional
            Smoothing factor to avoid division by zero
            
        Returns
        -------
        torch.Tensor
            IoU score
        """
        intersection = torch.sum(y_pred * y_true)
        union = torch.sum(y_pred) + torch.sum(y_true) - intersection
        iou = (intersection + smooth) / (union + smooth)
        return iou
    
    @staticmethod
    def precision(y_pred, y_true, smooth=1e-6):
        """
        Compute precision.
        
        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted segmentation mask
        y_true : torch.Tensor
            Ground truth segmentation mask
        smooth : float, optional
            Smoothing factor to avoid division by zero
            
        Returns
        -------
        torch.Tensor
            Precision score
        """
        true_positives = torch.sum(y_pred * y_true)
        predicted_positives = torch.sum(y_pred)
        precision = (true_positives + smooth) / (predicted_positives + smooth)
        return precision
    
    @staticmethod
    def recall(y_pred, y_true, smooth = 1e-6):
        """
        Compute recall (sensitivity).
        
        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted segmentation mask
        y_true : torch.Tensor
            Ground truth segmentation mask
        smooth : float, optional
            Smoothing factor to avoid division by zero
            
        Returns
        -------
        torch.Tensor
            Recall score
        """
        true_positives = torch.sum(y_pred * y_true)
        actual_positives = torch.sum(y_true)
        recall = (true_positives + smooth) / (actual_positives + smooth)
        return recall
    
    @staticmethod
    def all_metrics(y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Args:
            y_pred: Predicted segmentation mask (torch.Tensor)
            y_true: Ground truth segmentation mask (torch.Tensor)
            
        Returns:
            Dictionary with all metrics
        """
        # Convert to binary masks if probabilities
        if y_pred.dim() == 4 and y_pred.size(1) == 1:
            y_pred = y_pred.squeeze(1)
        if y_pred.dim() == 3:
            y_pred_binary = (y_pred > 0.5).float()
        else:
            y_pred_binary = (torch.argmax(y_pred, dim=1) > 0).float()
            
        if y_true.dim() == 4 and y_true.size(1) == 1:
            y_true = y_true.squeeze(1)
        
        # Calculate basic metrics
        dice = SegmentationMetrics.dice_coefficient(y_pred_binary, y_true)
        iou = SegmentationMetrics.iou_score(y_pred_binary, y_true)
        precision = SegmentationMetrics.precision(y_pred_binary, y_true)
        recall = SegmentationMetrics.recall(y_pred_binary, y_true)
        
        # Convert to numpy for distance-based metrics (if on CPU)
        metrics = {
            'dice': dice.item(),
            'iou': iou.item(),
            'precision': precision.item(),
            'recall': recall.item()
        }
        
        return metrics