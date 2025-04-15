import torch

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

class SegmentationMetricsAccumulator:
    """
    Class for computing segmentation metrics for pancreas segmentation.
    """

    def __init__(self, zero_division="nan", smooth_factor=1e-12):
        """
        Parameters
        ----------
        num_classes : int
            Number of classes in the segmentation task.
        absent_class_strategy : str
            What to do when a class is absent in both prediction and ground truth.
            Options: "nan", "one", "zero"
        """
        if zero_division not in ["nan", "one", "zero"]:
            raise ValueError("zero_division must be 'nan', 'one', or 'zero'")
        if zero_division == "nan":
            self.zero_division = float("nan")
        elif zero_division == "one":
            self.zero_division = 1.0
        elif zero_division == "zero":
            self.zero_division = 0.0
        self.zero_division_method = zero_division
        self.smooth_factor = smooth_factor
        self.reset()

    def reset(self):
        """Reset the metrics accumulator."""
        self.dice_scores = None
        self.iou_scores = None
        self.precision_scores = None
        self.recall_scores = None
        
    @staticmethod
    def compute_tp_fp_fn_tn(y_pred, y_true):
        """
        Compute true positives, false positives, false negatives, and true
        negatives.
        
        Parameters
        ----------
        y_pred : torch.Tensor
            The predicted segmentation map (one-hot encoded).
        y_true : torch.Tensor
            The ground truth segmentation map (one-hot encoded).

        Returns
        -------
        torch.Tensor
            True positives.
        torch.Tensor
            False positives.
        torch.Tensor
            False negatives.
        torch.Tensor
            True negatives.

        """
        # Ensure y_pred and y_true are one-hot encoded
        y_pred_oh, y_true_oh = convert_to_one_hot(y_pred, y_true)
        dims = tuple(range(2, y_pred_oh.ndim))

        tp = torch.sum(y_pred_oh * y_true_oh, dim=dims)
        fp = torch.sum(y_pred_oh, dim=dims) - tp
        fn = torch.sum(y_true_oh, dim=dims) - tp
        tn = torch.sum((1 - y_pred_oh) * (1 - y_true_oh), dim=dims)

        return tp, fp, fn, tn
    
    def compute_dice(self, tp, fp, fn):
        """
        Compute the Dice coefficient.

        Parameters
        ----------
        tp : torch.Tensor
            True positives.
        fp : torch.Tensor
            False positives.
        fn : torch.Tensor
            False negatives.

        Returns
        -------
        torch.Tensor
            The Dice coefficient.
        """
        dice = 2. * tp / (2. * tp + fp + fn + self.smooth_factor)
        # Handle zero division cases
        mask = (tp + fp + fn) == 0
        dice = self._handle_zero_division(dice, mask)

        return dice
    
    def compute_iou(self, tp, fp, fn):
        """
        Compute the Intersection over Union (IoU).

        Parameters
        ----------
        tp : torch.Tensor
            True positives.
        fp : torch.Tensor
            False positives.
        fn : torch.Tensor
            False negatives.

        Returns
        -------
        torch.Tensor
            The IoU.
        """
        iou = tp / (tp + fp + fn + self.smooth_factor)
        # Handle zero division cases
        mask = (tp + fp + fn) == 0
        iou = self._handle_zero_division(iou, mask)

        return iou
    
    def compute_precision(self, tp, fp):
        """
        Compute the precision.

        Parameters
        ----------
        tp : torch.Tensor
            True positives.
        fp : torch.Tensor
            False positives.

        Returns
        -------
        torch.Tensor
            The precision.
        """
        precision = tp / (tp + fp + self.smooth_factor)
        # Handle zero division cases
        mask = (tp + fp) == 0
        precision = self._handle_zero_division(precision, mask)

        return precision
    
    def compute_recall(self, tp, fn):
        """
        Compute the recall.

        Parameters
        ----------
        tp : torch.Tensor
            True positives.
        fn : torch.Tensor
            False negatives.

        Returns
        -------
        torch.Tensor
            The recall.
        """
        recall = tp / (tp + fn + self.smooth_factor)
        # Handle zero division cases
        mask = (tp + fn) == 0
        recall = self._handle_zero_division(recall, mask)

        return recall

    def update(self, y_pred, y_true):
        """
        Update the metrics accumulator with new predictions and ground truth.

        Parameters
        ----------
        y_pred : torch.Tensor
            The predicted segmentation map (one-hot encoded).
        y_true : torch.Tensor
            The ground truth segmentation map (one-hot encoded).
        """
        # Ensure y_pred and y_true are one-hot encoded
        y_pred_oh, y_true_oh = convert_to_one_hot(y_pred, y_true)
        dims = tuple(range(2, y_pred_oh.ndim))

        # Compute true positives, false positives, and false negatives
        tp, fp, fn, tn = self.compute_tp_fp_fn_tn(y_pred_oh, y_true_oh)

        # Compute dice
        dice = 2. * tp / (2. * tp + fp + fn + self.smooth_factor)
        mask = (tp + fp + fn) == 0
        dice = self._handle_zero_division(dice, mask)

        # Compute IoU
        iou = tp / (tp + fp + fn + self.smooth_factor)
        iou = self._handle_zero_division(iou, mask) # IoU mask == Dice mask

        # Compute precision
        precision = tp / (tp + fp + self.smooth_factor)
        mask = (tp + fp) == 0
        precision = self._handle_zero_division(precision, mask)

        # Compute recall
        recall = tp / (tp + fn + self.smooth_factor)
        mask = (tp + fn) == 0
        recall = self._handle_zero_division(recall, mask)

        # Store the results
        if self.dice_scores is None:
            self.dice_scores = dice
            self.iou_scores = iou
            self.precision_scores = precision
            self.recall_scores = recall
        else:
            self.dice_scores = torch.cat((self.dice_scores, dice), dim=0)
            self.iou_scores = torch.cat((self.iou_scores, iou), dim=0)
            self.precision_scores = torch.cat((self.precision_scores, precision), dim=0)
            self.recall_scores = torch.cat((self.recall_scores, recall), dim=0)

    def _handle_zero_division(self, value_tensor, mask):
        # If there are no absent classes, return the original tensor
        if mask.sum() == 0:
            return value_tensor
        # Handle zero division cases
        if self.zero_division_method == "nan":
            value_tensor[mask] = float("nan")
        elif self.zero_division_method == "one":
            value_tensor[mask] = 1.0
        elif self.zero_division_method == "zero":
            value_tensor[mask] = 0.0
        return value_tensor

    def aggregate(self):
        C = self.dice_scores.shape[1]  # Number of classes

        # Calculate means excluding NaNs
        dice = torch.nanmean(self.dice_scores, dim=0)
        iou = torch.nanmean(self.iou_scores, dim=0)
        precision = torch.nanmean(self.precision_scores, dim=0)
        recall = torch.nanmean(self.recall_scores, dim=0)

        metrics = {
            f"dice_class_{i}": dice[i].item() for i in range(C)
        }
        metrics.update({f"iou_class_{i}": iou[i].item() for i in range(C)})
        metrics.update({f"precision_class_{i}": precision[i].item() for i in range(C)})
        metrics.update({f"recall_class_{i}": recall[i].item() for i in range(C)})

        # Calculate means excluding NaNs
        metrics["dice"] = torch.nanmean(dice).item()
        metrics["iou"] = torch.nanmean(iou).item()
        metrics["precision"] = torch.nanmean(precision).item()
        metrics["recall"] = torch.nanmean(recall).item()

        return metrics