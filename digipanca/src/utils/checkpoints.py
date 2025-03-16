import os
import torch
from tqdm import tqdm

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir="checkpoints",
                    filename="model.pth", delete_old=True):
    """
    Save the model checkpoint.

    Parameters
    ----------
    model : torch.nn.Module
        Model to save.
    optimizer : torch.optim.Optimizer
        Optimizer state to save.
    epoch : int
        Current epoch.
    loss : float
        Current loss.
    checkpoint_dir : str, optional
        Directory to save the checkpoint.
    filename : str, optional
        Name of the checkpoint file.
    delete_old : bool, optional
        Whether to delete old checkpoints before saving the new one.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Delete old checkpoints if required
    if delete_old:
        for file in os.listdir(checkpoint_dir):
            if file.startswith("best_model_epoch") and file.endswith(".pth"):
                os.remove(os.path.join(checkpoint_dir, file))

    checkpoint_path = os.path.join(checkpoint_dir, filename)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }
    torch.save(checkpoint, checkpoint_path)
    tqdm.write(f"Saved checkpoint: {checkpoint_path}")

def load_checkpoint(model, optimizer, checkpoint_path):
    """
    Load the model checkpoint.

    Parameters
    ----------
    model : torch.nn.Module
        Model to load the checkpoint into.
    optimizer : torch.optim.Optimizer
        Optimizer to load the state into.
    checkpoint_path : str
        Path to the checkpoint file.

    Returns
    -------
    int
        Epoch of the checkpoint.
    float
        Loss of the checkpoint.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print(f"Loaded checkpoint: {checkpoint_path}")

    return checkpoint["epoch"], checkpoint["loss"]