import torch
import monai.metrics as mm
import torch.nn.functional as F

class SegmentationMonaiMetrics:
    """
    A class to compute segmentation metrics using MONAI.
    This class computes the Dice score for segmentation tasks.
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
            print("Both y_pred and y_true are already one-hot encoded.")
            return y_pred, y_true
        
        # Check if the input is 2D or 3D
        if y_pred.dim() == 4 and y_true.dim() == 3: # 2D case
            print("2D case")
            B, C, H, W = y_pred.shape
            n_classes = C

            print(f"y_pred shape: {y_pred.shape}, y_true shape: {y_true.shape}")
            print(f"y_pred values: {y_pred.unique()}")

            # Convert y_pred to one-hot encoding
            y_pred_classes = torch.argmax(y_pred, dim=1, keepdim=True)
            y_pred_one_hot = torch.zeros(B, n_classes, H, W, device=y_pred.device)
            y_pred_one_hot.scatter_(1, y_pred_classes, 1)

            # Convert y_true to one-hot encoding
            y_true_one_hot = torch.zeros(B, n_classes, H, W, device=y_true.device)
            y_true_one_hot.scatter_(1, y_true.unsqueeze(1).long(), 1)

            # Ensure y_true_one_hot values are within the valid range of class indices
            print(f"y_true_one_hot shape: {y_true_one_hot.shape}, y_pred_one_hot shape: {y_pred_one_hot.shape}")
            print(f"y_true values: {y_true.unique()}")
            print(f"y_true_one_hot values: {y_true_one_hot.unique()}")
            print(f"y_pred_one_hot values: {y_pred_one_hot.unique()}")

            return y_pred_one_hot, y_true_one_hot
        
        elif y_pred.dim() == 5 and y_true.dim() == 4:   # 3D case
            print("3D case")
            B, C, D, H, W = y_pred.shape
            n_classes = C

            print(f"y_pred shape: {y_pred.shape}, y_true shape: {y_true.shape}")
            print(f"y_pred values: {y_pred.unique()}")

            # Convert y_pred to one-hot encoding
            y_pred_classes = torch.argmax(y_pred, dim=1, keepdim=True)
            y_pred_one_hot = torch.zeros(B, n_classes, D, H, W, device=y_pred.device)
            y_pred_one_hot.scatter_(1, y_pred_classes, 1)

            # Convert y_true to one-hot encoding
            y_true_one_hot = torch.zeros(B, n_classes, D, H, W, device=y_true.device)
            y_true_one_hot.scatter_(1, y_true.unsqueeze(1).long(), 1)

            print(f"y_true_one_hot shape: {y_true_one_hot.shape}, y_pred_one_hot shape: {y_pred_one_hot.shape}")
            print(f"y_true values: {y_true.unique()}")
            print(f"y_true_one_hot values: {y_true_one_hot.unique()}")
            print(f"y_pred_one_hot values: {y_pred_one_hot.unique()}")

            return y_pred_one_hot, y_true_one_hot

        else:
            raise ValueError("Input tensors must be either 2D or 3D.")
        
    @staticmethod
    def compute_dice(y_pred, y_true, include_background=True):
        """
        Compute the Dice score for the predicted and ground truth segmentation maps.

        Parameters
        ----------
        y_pred : torch.Tensor
            The predicted segmentation map.
        y_true : torch.Tensor
            The ground truth segmentation map.
        include_background : bool, optional
            Whether to include the background class in the Dice score calculation.

        Returns
        -------
        float
            The computed Dice score.
        """
        # Convert to one-hot encoding
        y_pred_one_hot, y_true_one_hot = SegmentationMonaiMetrics.convert_to_one_hot(y_pred, y_true)
        C = y_true_one_hot.size(1)

        # Compute Dice score using MONAI's function
        dice_metric = mm.DiceMetric(
            include_background=include_background,
            reduction="none"
        )

        dice_scores = dice_metric(y_pred_one_hot, y_true_one_hot)
        dice_scores = dice_scores.mean(dim=0)

        dice_dict = {f"dice_class_{i}": dice_scores[i].item() for i in range(C)}
        dice_dict["dice_mean"] = dice_scores.mean().item()
        
        return dice_scores.mean(), dice_dict

    @staticmethod
    def compute_iou(y_pred, y_true, include_background=True):
        """
        Compute the Intersection over Union (IoU) for the predicted and ground 
        truth segmentation maps.

        Parameters
        ----------
        y_pred : torch.Tensor
            The predicted segmentation map.
        y_true : torch.Tensor
            The ground truth segmentation map.
        include_background : bool, optional
            Whether to include the background class in the IoU calculation.

        Returns
        -------
        float
            The computed IoU score.
        dict
            A dictionary containing IoU scores for each class.
        """
        y_pred_one_hot, y_true_one_hot = SegmentationMonaiMetrics.convert_to_one_hot(y_pred, y_true)
        C = y_true_one_hot.size(1)

        iou_metric = mm.MeanIoU(
            include_background=include_background,
            reduction="none"
        )
        iou_scores = iou_metric(y_pred_one_hot, y_true_one_hot)
        iou_scores = iou_scores.mean(dim=0)

        iou_dict = {f"iou_class_{i}": iou_scores[i].item() for i in range(C)}
        iou_dict["iou_mean"] = iou_scores.mean().item()

        return iou_scores.mean(), iou_dict

    @staticmethod
    def compute_precision_recall(y_pred, y_true, include_background=True):
        """
        Compute the precision and recall for the predicted and ground truth
        segmentation maps.

        Parameters
        ----------
        y_pred : torch.Tensor
            The predicted segmentation map.
        y_true : torch.Tensor
            The ground truth segmentation map.
        include_background : bool, optional
            Whether to include the background class in the precision and recall 
            calculation.

        Returns
        -------
        float
            The computed precision score.
        float
            The computed recall score.
        dict
            A dictionary containing precision scores for each class.
        dict
            A dictionary containing recall scores for each class.
        """
        y_pred_one_hot, y_true_one_hot = SegmentationMonaiMetrics.convert_to_one_hot(y_pred, y_true)
        C = y_true_one_hot.size(1)

        # Compute confusion matrix
        cm = mm.get_confusion_matrix(
            y_pred=y_pred_one_hot,
            y=y_true_one_hot,
            include_background=include_background
        )
        
        # Compute precision and recall using the confusion matrix
        precision = mm.compute_confusion_matrix_metric('precision', cm).mean(dim=0)
        recall = mm.compute_confusion_matrix_metric('recall', cm).mean(dim=0)

        precision_dict = {f"precision_class_{i}": precision[i].item() for i in range(C)}
        precision_dict["precision_mean"] = precision.mean().item()

        recall_dict = {f"recall_class_{i}": recall[i].item() for i in range(C)}
        recall_dict["recall_mean"] = recall.mean().item()

        return precision.mean(), recall.mean(), precision_dict, recall_dict