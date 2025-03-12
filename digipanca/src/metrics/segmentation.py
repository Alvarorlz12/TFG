import torch

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
            Predicted segmentation mask (class indices or one-hot)
        y_true : torch.Tensor
            Ground truth segmentation mask (class indices or one-hot)
        smooth : float, optional
            Smoothing factor to avoid division by zero
        
        Returns
        -------
        torch.Tensor
            Dice coefficient (overall and per-class)
        """
        # Convert to one-hot if inputs are class indices
        if y_pred.dim() == 3:
            # Convert predicted class indices to one-hot
            n_classes = torch.max(y_true).item() + 1
            y_pred_one_hot = torch.zeros(
                y_pred.size(0), n_classes, y_pred.size(1), y_pred.size(2), 
                device=y_pred.device
            )
            y_pred_one_hot.scatter_(1, y_pred.unsqueeze(1), 1)
            
            y_true_one_hot = torch.zeros(
                y_true.size(0), n_classes, y_true.size(1), y_true.size(2), 
                device=y_true.device
            )
            y_true_one_hot.scatter_(1, y_true.unsqueeze(1), 1)
        else:
            # If already in form [B, C, H, W] (logits or one-hot)
            if y_pred.dim() == 4 and y_true.dim() == 3:
                # y_pred is [B, C, H, W] logits and y_true is [B, H, W] indices
                n_classes = y_pred.size(1)
                y_pred_one_hot = torch.nn.functional.softmax(y_pred, dim=1)
                
                y_true_one_hot = torch.zeros(
                    y_true.size(0), n_classes, y_true.size(1), y_true.size(2), 
                    device=y_true.device
                )
                y_true_one_hot.scatter_(1, y_true.unsqueeze(1), 1)
            else:
                # Assume both are already in proper format
                y_pred_one_hot = y_pred
                y_true_one_hot = y_true
                n_classes = y_pred.size(1)
        
        # Calculate dice for each class
        dice_scores = []
        class_dice = {}
        
        for i in range(n_classes):
            pred_class = y_pred_one_hot[:, i, :, :]
            true_class = y_true_one_hot[:, i, :, :]
            
            intersection = torch.sum(pred_class * true_class)
            union = torch.sum(pred_class) + torch.sum(true_class)
            dice = (2.0 * intersection + smooth) / (union + smooth)
            dice_scores.append(dice)
            class_dice[f"dice_class_{i}"] = dice.item()
        
        mean_dice = torch.mean(torch.stack(dice_scores))
        class_dice["dice_mean"] = mean_dice.item()
        
        return mean_dice, class_dice
    
    @staticmethod
    def iou_score(y_pred, y_true, smooth=1e-6):
        """
        Compute IoU (Jaccard Index) for multiclass segmentation.
        
        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted segmentation mask (class indices)
        y_true : torch.Tensor
            Ground truth segmentation mask (class indices)
        smooth : float, optional
            Smoothing factor to avoid division by zero
            
        Returns
        -------
        tuple
            (mean_iou, per_class_iou_dict)
        """
        # Convert to one-hot if inputs are class indices
        if y_pred.dim() == 3:
            # Assume both are class indices
            n_classes = torch.max(y_true).item() + 1
            y_pred_one_hot = torch.zeros(
                y_pred.size(0), n_classes, y_pred.size(1), y_pred.size(2), 
                device=y_pred.device
            )
            y_pred_one_hot.scatter_(1, y_pred.unsqueeze(1), 1)
            
            y_true_one_hot = torch.zeros(
                y_true.size(0), n_classes, y_true.size(1), y_true.size(2), 
                device=y_true.device
            )
            y_true_one_hot.scatter_(1, y_true.unsqueeze(1), 1)
        else:
            # If already in form [B, C, H, W] (logits or one-hot)
            if y_pred.dim() == 4 and y_true.dim() == 3:
                # y_pred is [B, C, H, W] logits and y_true is [B, H, W] indices
                n_classes = y_pred.size(1)
                y_pred_one_hot = torch.nn.functional.softmax(y_pred, dim=1)
                
                y_true_one_hot = torch.zeros(
                    y_true.size(0), n_classes, y_true.size(1), y_true.size(2), 
                    device=y_true.device
                )
                y_true_one_hot.scatter_(1, y_true.unsqueeze(1), 1)
            else:
                # Assume both are already in proper format
                y_pred_one_hot = y_pred
                y_true_one_hot = y_true
                n_classes = y_pred.size(1)
                
        # Calculate IoU for each class
        iou_scores = []
        class_iou = {}
        
        for i in range(n_classes):
            pred_class = y_pred_one_hot[:, i, :, :]
            true_class = y_true_one_hot[:, i, :, :]
            
            intersection = torch.sum(pred_class * true_class)
            union = torch.sum(pred_class) + torch.sum(true_class) - intersection
            iou = (intersection + smooth) / (union + smooth)
            iou_scores.append(iou)
            class_iou[f"iou_class_{i}"] = iou.item()
        
        mean_iou = torch.mean(torch.stack(iou_scores))
        class_iou["iou_mean"] = mean_iou.item()
        
        return mean_iou, class_iou
    
    @staticmethod
    def precision_recall(y_pred, y_true, smooth=1e-6):
        """
        Compute precision and recall for multiclass segmentation.
        
        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted segmentation mask (class indices)
        y_true : torch.Tensor
            Ground truth segmentation mask (class indices)
        smooth : float, optional
            Smoothing factor to avoid division by zero
            
        Returns
        -------
        tuple
            (mean_precision, mean_recall, per_class_precision_dict, per_class_recall_dict)
        """
        # Convert to one-hot if inputs are class indices
        if y_pred.dim() == 3:
            # Assume both are class indices
            n_classes = torch.max(y_true).item() + 1
            y_pred_one_hot = torch.zeros(
                y_pred.size(0), n_classes, y_pred.size(1), y_pred.size(2), 
                device=y_pred.device
            )
            y_pred_one_hot.scatter_(1, y_pred.unsqueeze(1), 1)
            
            y_true_one_hot = torch.zeros(
                y_true.size(0), n_classes, y_true.size(1), y_true.size(2), 
                device=y_true.device
            )
            y_true_one_hot.scatter_(1, y_true.unsqueeze(1), 1)
        else:
            # If already in form [B, C, H, W] (logits or one-hot)
            if y_pred.dim() == 4 and y_true.dim() == 3:
                # y_pred is [B, C, H, W] logits and y_true is [B, H, W] indices
                n_classes = y_pred.size(1)
                y_pred_one_hot = torch.nn.functional.softmax(y_pred, dim=1)
                
                y_true_one_hot = torch.zeros(
                    y_true.size(0), n_classes, y_true.size(1), y_true.size(2), 
                    device=y_true.device
                )
                y_true_one_hot.scatter_(1, y_true.unsqueeze(1), 1)
            else:
                # Assume both are already in proper format
                y_pred_one_hot = y_pred
                y_true_one_hot = y_true
                n_classes = y_pred.size(1)
                
        # Calculate precision and recall for each class
        precision_scores = []
        recall_scores = []
        class_precision = {}
        class_recall = {}
        
        for i in range(n_classes):
            pred_class = y_pred_one_hot[:, i, :, :]
            true_class = y_true_one_hot[:, i, :, :]
            
            true_positives = torch.sum(pred_class * true_class)
            predicted_positives = torch.sum(pred_class)
            actual_positives = torch.sum(true_class)
            
            precision = (true_positives + smooth) / (predicted_positives + smooth)
            recall = (true_positives + smooth) / (actual_positives + smooth)
            
            precision_scores.append(precision)
            recall_scores.append(recall)
            
            class_precision[f"precision_class_{i}"] = precision.item()
            class_recall[f"recall_class_{i}"] = recall.item()
        
        mean_precision = torch.mean(torch.stack(precision_scores))
        mean_recall = torch.mean(torch.stack(recall_scores))
        
        class_precision["precision_mean"] = mean_precision.item()
        class_recall["recall_mean"] = mean_recall.item()
        
        return mean_precision, mean_recall, class_precision, class_recall
    
    @staticmethod
    def all_metrics(y_pred, y_true):
        """
        Compute all metrics for multiclass segmentation.
        
        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted segmentation mask (class indices)
        y_true : torch.Tensor
            Ground truth segmentation mask (class indices)
            
        Returns
        -------
        dict
            Dictionary of all metrics
        """
        metrics = {}
        
        # Convert logits to class indices if necessary
        if y_pred.dim() == 4:
            # Logits [B, C, H, W] -> Class indices [B, H, W]
            y_pred_indices = torch.argmax(y_pred, dim=1)
        else:
            # Already class indices
            y_pred_indices = y_pred
            
        # Calculate Dice coefficient
        mean_dice, class_dice = SegmentationMetrics.dice_coefficient(y_pred, y_true)
        metrics.update(class_dice)
        
        # Calculate IoU score
        mean_iou, class_iou = SegmentationMetrics.iou_score(y_pred, y_true)
        metrics.update(class_iou)
        
        # Calculate precision and recall
        mean_precision, mean_recall, class_precision, class_recall = SegmentationMetrics.precision_recall(y_pred, y_true)
        metrics.update(class_precision)
        metrics.update(class_recall)
        
        # Add overall metrics
        metrics['dice'] = mean_dice.item()
        metrics['iou'] = mean_iou.item()
        metrics['precision'] = mean_precision.item()
        metrics['recall'] = mean_recall.item()
        
        return metrics