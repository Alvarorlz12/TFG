import torch

class SegmentationMetrics:
    """
    Class for computing segmentation metrics for pancreas segmentation.
    """

    @staticmethod
    def convert_to_one_hot(y_pred, y_true):
        """
        Convert a prediction and ground truth to one-hot encoding. It works for 
        both 2D data (H, W) and 3D data (D, H, W), which tensors are (B, H, W) 
        and (B, D, H, W) respectively.

        Parameters
        ----------
        y_pred : torch.Tensor
            The predicted segmentation map.
        y_true : torch.Tensor
            The ground truth segmentation map.

        Returns
        -------
        torch.Tensor
            The one-hot encoded tensor for the predicted segmentation map.
        torch.Tensor
            The one-hot encoded tensor for the ground truth segmentation map.
        """
        def is_one_hot(tensor):
            """
            Check if the tensor is one-hot encoded.
            """
            return (tensor.sum(dim=1) == 1).all() and \
                   torch.all((tensor == 0) | (tensor == 1))
        
        # Check if the input is already one-hot encoded
        if is_one_hot(y_pred) and is_one_hot(y_true):
            return y_pred, y_true
        
        # Check if the input is 2D or 3D
        if y_pred.dim() == 4 and y_true.dim() == 3: # 2D case
            B, C, H, W = y_pred.shape
            n_classes = C

            # Convert y_pred to one-hot encoding
            y_pred_classes = torch.argmax(y_pred, dim=1, keepdim=True)
            y_pred_one_hot = torch.zeros(B, n_classes, H, W, device=y_pred.device)
            y_pred_one_hot.scatter_(1, y_pred_classes, 1)

            # Convert y_true to one-hot encoding
            y_true_one_hot = torch.zeros(B, n_classes, H, W, device=y_true.device)
            y_true_one_hot.scatter_(1, y_true.unsqueeze(1).long(), 1)

            return y_pred_one_hot, y_true_one_hot
        
        elif y_pred.dim() == 5 and y_true.dim() == 4:   # 3D case
            B, C, D, H, W = y_pred.shape
            n_classes = C

            # Convert y_pred to one-hot encoding
            y_pred_classes = torch.argmax(y_pred, dim=1, keepdim=True)
            y_pred_one_hot = torch.zeros(B, n_classes, D, H, W, device=y_pred.device)
            y_pred_one_hot.scatter_(1, y_pred_classes, 1)

            # Convert y_true to one-hot encoding
            y_true_one_hot = torch.zeros(B, n_classes, D, H, W, device=y_true.device)
            y_true_one_hot.scatter_(1, y_true.unsqueeze(1).long(), 1)

            return y_pred_one_hot, y_true_one_hot

        else:
            raise ValueError("Input tensors must be either 2D or 3D.")
    
    @staticmethod
    def dice_coefficient(y_pred, y_true, smooth=1e-12):
        """
        Compute Dice coefficient.
        
        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted segmentation mask (class indices, logits or one-hot)
        y_true : torch.Tensor
            Ground truth segmentation mask (class indices, logits or one-hot)
        smooth : float, optional
            Smoothing factor to avoid division by zero
        
        Returns
        -------
        torch.Tensor
            Dice coefficient (overall and per-class)
        """
        # Convert to one-hot if inputs are class indices or logits
        y_pred_one_hot, y_true_one_hot = SegmentationMetrics.convert_to_one_hot(y_pred, y_true)
        C = y_true_one_hot.size(1)
        sum_dims = tuple(range(2, y_true_one_hot.ndim))
        
        # Compute intersection and union
        intersection = torch.sum(y_pred_one_hot * y_true_one_hot, dim=sum_dims)
        union = torch.sum(y_pred_one_hot, dim=sum_dims) \
              + torch.sum(y_true_one_hot, dim=sum_dims)

        # Compute Dice score
        dice_scores = 2. * intersection / (union + smooth)
        dice_scores = dice_scores.mean(dim=0)

        dice_dict = {f"dice_class_{i}": dice_scores[i].item() for i in range(C)}
        dice_dict["dice_mean"] = dice_scores.mean().item()
        
        return dice_scores.mean(), dice_dict
        
    @staticmethod
    def iou_score(y_pred, y_true, smooth=1e-12):
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
        # Convert to one-hot if inputs are class indices or logits
        y_pred_one_hot, y_true_one_hot = SegmentationMetrics.convert_to_one_hot(y_pred, y_true)
        C = y_true_one_hot.size(1)
        sum_dims = tuple(range(2, y_true_one_hot.ndim))

        # Compute intersection and union
        intersection = torch.sum(y_pred_one_hot * y_true_one_hot, dim=sum_dims)
        union = torch.sum(y_pred_one_hot, dim=sum_dims) \
              + torch.sum(y_true_one_hot, dim=sum_dims) - intersection
        
        # Compute IoU score
        iou_scores = intersection / (union + smooth)
        iou_scores = iou_scores.mean(dim=0)

        iou_dict = {f"iou_class_{i}": iou_scores[i].item() for i in range(C)}
        iou_dict["iou_mean"] = iou_scores.mean().item()

        return iou_scores.mean(), iou_dict
    
    @staticmethod
    def precision_recall(y_pred, y_true, smooth=1e-12):
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
        # Convert to one-hot if inputs are class indices or logits
        y_pred_one_hot, y_true_one_hot = SegmentationMetrics.convert_to_one_hot(y_pred, y_true)
        C = y_true_one_hot.size(1)
        sum_dims = tuple(range(2, y_true_one_hot.ndim))

        # Compute true positives, false positives, and false negatives
        tp = torch.sum(y_pred_one_hot * y_true_one_hot, dim=sum_dims)
        fp = torch.sum(y_pred_one_hot, dim=sum_dims) - tp
        fn = torch.sum(y_true_one_hot, dim=sum_dims) - tp

        # Compute precision and recall
        precision = tp / (tp + fp + smooth)
        recall = tp / (tp + fn + smooth)

        precision = precision.mean(dim=0)
        recall = recall.mean(dim=0)

        precision_dict = {f"precision_class_{i}": precision[i].item() for i in range(C)}
        recall_dict = {f"recall_class_{i}": recall[i].item() for i in range(C)}
        
        return precision.mean(), recall.mean(), precision_dict, recall_dict
    
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
        
        # Convert to one-hot encoding
        y_pred_one_hot, y_true_one_hot = SegmentationMetrics.convert_to_one_hot(y_pred, y_true)
            
        # Calculate Dice coefficient
        mean_dice, class_dice = SegmentationMetrics.dice_coefficient(y_pred_one_hot, y_true_one_hot)
        metrics.update(class_dice)
        
        # Calculate IoU score
        mean_iou, class_iou = SegmentationMetrics.iou_score(y_pred_one_hot, y_true_one_hot)
        metrics.update(class_iou)
        
        # Calculate precision and recall
        mean_precision,mean_recall, class_precision, class_recall = SegmentationMetrics.precision_recall(y_pred_one_hot, y_true_one_hot)
        metrics.update(class_precision)
        metrics.update(class_recall)
        
        # Add overall metrics
        metrics['dice'] = mean_dice.item()
        metrics['iou'] = mean_iou.item()
        metrics['precision'] = mean_precision.item()
        metrics['recall'] = mean_recall.item()
        
        return metrics