import torch

from src.utils.data import convert_to_one_hot

class SegmentationMetricsAccumulator:
    """
    Class for computing segmentation metrics for pancreas segmentation.
    """

    def __init__(
            self,
            zero_division="nan",
            smooth_factor=1e-12,
            include_background=True
        ):
        """
        Parameters
        ----------
        num_classes : int
            Number of classes in the segmentation task.
        absent_class_strategy : str
            What to do when a class is absent in both prediction and ground truth.
            Options: "nan", "one", "zero"
        smooth_factor : float, optional
            Smoothing factor to avoid division by zero, by default 1e-12
        include_background : bool, optional
            Whether to include the background class in the metrics, by default True.
        """
        if zero_division not in ["nan", "one", "zero"]:
            raise ValueError("zero_division must be 'nan', 'one', or 'zero'")
        self.zero_division_method = zero_division
        self.smooth_factor = smooth_factor
        self.include_background = include_background
        self.reset()

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
    
    @staticmethod
    def _get_metrics(
        dice_scores,
        iou_scores,
        precision_scores,
        recall_scores,
        include_background=True
    ):
        C = dice_scores.shape[1]  # Number of classes

        # Exclude background class if not needed
        start_class = 0 if include_background else 1

        # Calculate means excluding NaNs
        dice = torch.nanmean(dice_scores[:, start_class:], dim=0)
        iou = torch.nanmean(iou_scores[:, start_class:], dim=0)
        precision = torch.nanmean(precision_scores[:, start_class:], dim=0)
        recall = torch.nanmean(recall_scores[:, start_class:], dim=0)

        classes = range(start_class, C)

        metrics = {
            f"dice_class_{i}": dice[i - start_class].item() for i in classes
        }
        metrics.update({f"iou_class_{i}": iou[i - start_class].item() for i in classes})
        metrics.update({f"precision_class_{i}": precision[i - start_class].item() for i in classes})
        metrics.update({f"recall_class_{i}": recall[i - start_class].item() for i in classes})

        # Calculate means excluding NaNs
        metrics["dice"] = torch.nanmean(dice).item()
        metrics["iou"] = torch.nanmean(iou).item()
        metrics["precision"] = torch.nanmean(precision).item()
        metrics["recall"] = torch.nanmean(recall).item()

        return metrics

    def reset(self):
        """Reset the metrics accumulator."""
        self.dice_scores = []
        self.iou_scores = []
        self.precision_scores = []
        self.recall_scores = []
        self.tp = []
        self.fp = []
        self.fn = []
        self.tn = []
        
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
        # Mask for absent classes (both prediction and ground truth are zero)
        dims = tuple(range(2, y_pred_oh.ndim))
        mask = torch.sum(y_pred_oh + y_true_oh, dim=dims) == 0

        # Compute true positives, false positives, and false negatives
        tp, fp, fn, tn = self.compute_tp_fp_fn_tn(y_pred_oh, y_true_oh)

        # Compute dice
        dice = 2. * tp / (2. * tp + fp + fn + self.smooth_factor)
        dice = self._handle_zero_division(dice, mask)

        # Compute IoU
        iou = tp / (tp + fp + fn + self.smooth_factor)
        iou = self._handle_zero_division(iou, mask)

        # Compute precision
        precision = tp / (tp + fp + self.smooth_factor)
        precision = self._handle_zero_division(precision, mask)

        # Compute recall
        recall = tp / (tp + fn + self.smooth_factor)
        recall = self._handle_zero_division(recall, mask)

        # Store the results
        self.dice_scores.append(dice)
        self.iou_scores.append(iou)
        self.precision_scores.append(precision)
        self.recall_scores.append(recall)
        self.tp.append(tp)
        self.fp.append(fp)
        self.fn.append(fn)
        self.tn.append(tn)

        return SegmentationMetricsAccumulator._get_metrics(
            dice,
            iou,
            precision,
            recall,
            include_background=self.include_background
        )

    def aggregate(self):
        """
        Aggregate the metrics across all batches and classes.
            
        Returns
        -------
        dict
            A dictionary containing the aggregated metrics.
        """
        dice_tensor = torch.cat(self.dice_scores, dim=0)
        iou_tensor = torch.cat(self.iou_scores, dim=0)
        precision_tensor = torch.cat(self.precision_scores, dim=0)
        recall_tensor = torch.cat(self.recall_scores, dim=0)

        return SegmentationMetricsAccumulator._get_metrics(
            dice_tensor,
            iou_tensor,
            precision_tensor,
            recall_tensor,
            include_background=self.include_background
        )
    
    def aggregate_global_cm(self):
        tp_tensor = torch.cat(self.tp, dim=0)
        fp_tensor = torch.cat(self.fp, dim=0)
        fn_tensor = torch.cat(self.fn, dim=0)
        tn_tensor = torch.cat(self.tn, dim=0)

        C = tp_tensor.shape[1]  # Number of classes

        # Exclude background class if not needed
        start_class = 0 if self.include_background else 1

        # Calculate means excluding NaNs
        tp = torch.nansum(tp_tensor[:, start_class:], dim=0)
        fp = torch.nansum(fp_tensor[:, start_class:], dim=0)
        fn = torch.nansum(fn_tensor[:, start_class:], dim=0)
        tn = torch.nansum(tn_tensor[:, start_class:], dim=0)

        classes = range(start_class, C)

        cm = {
            f"tp_class_{i}": tp[i - start_class].item() for i in classes
        }
        cm.update({f"fp_class_{i}": fp[i - start_class].item() for i in classes})
        cm.update({f"fn_class_{i}": fn[i - start_class].item() for i in classes})
        cm.update({f"tn_class_{i}": tn[i - start_class].item() for i in classes})

        # Calculate means excluding NaNs
        cm["tp"] = torch.nansum(tp).item()
        cm["fp"] = torch.nansum(fp).item()
        cm["fn"] = torch.nansum(fn).item()
        cm["tn"] = torch.nansum(tn).item()

        return cm

