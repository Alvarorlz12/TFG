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
    
def to_one_hot(tensor, n_classes):
    """
    Convert a tensor to one-hot encoding for both 2D and 3D data.

    Parameters
    ----------
    tensor : torch.Tensor
        The input tensor. Shape can be (B, H, W) for 2D or (B, D, H, W) for 3D.
    n_classes : int
        The number of classes.

    Returns
    -------
    torch.Tensor
        The one-hot encoded tensor.
    """
    if tensor.dim() == 3:  # 2D case
        B, H, W = tensor.shape
        one_hot = torch.zeros(B, n_classes, H, W, device=tensor.device)
        one_hot.scatter_(1, tensor.unsqueeze(1).long(), 1)
    elif tensor.dim() == 4:  # 3D case
        B, D, H, W = tensor.shape
        one_hot = torch.zeros(B, n_classes, D, H, W, device=tensor.device)
        one_hot.scatter_(1, tensor.unsqueeze(1).long(), 1)
    else:
        raise ValueError("Input tensor must have 3 or 4 dimensions (2D or 3D).")
    
    return one_hot

def prepare_tensors_for_visualization(predictions, ground_truth, volume=None):
    """
    Prepare tensors for visualization by converting them to NumPy arrays.

    Parameters
    ----------
    prediction : torch.Tensor
        The predicted segmentation map. It must have the following shape:
        (B, C, D, H, W) and be logits or probabilities so argmax can be applied.
    ground_truth : torch.Tensor
        The ground truth segmentation map. It must have the following shape:
        (B, D, H, W).
    volume : torch.Tensor, optional
        The original volume (if available). It must have the following shape:
        (B, C, D, H, W) where C must be 1.
        If not provided, the function will return None for the volume.

    Returns
    -------
    ndarray
        The predicted segmentation map as a NumPy array.
    ndarray
        The ground truth segmentation map as a NumPy array.
    ndarray or None
        The original volume as a NumPy array, or None if not provided.
    """
    # Ensure (D, H, W) for prediction and ground truth
    pred_np = torch.argmax(predictions[0], dim=0).cpu().numpy()
    gt_np = ground_truth[0].cpu().numpy()
    # Ensure (D, H, W) or None for volume
    volume_np = volume[0][0].cpu().numpy() if volume is not None else None

    return pred_np, gt_np, volume_np