import os
from tqdm import tqdm

from src.utils.checkpoints import save_checkpoint

class ModelCheckpoint:
    """
    Save the best model checkpoint based on the given metric.
    """
    def __init__(self, checkpoint_dir, metric_name="valid_loss", mode="min"):
        self.checkpoint_dir = checkpoint_dir
        self.metric_name = metric_name
        self.mode = mode
        self.best_metric = float("inf") if mode == "min" else float("-inf")
        os.makedirs(checkpoint_dir, exist_ok=True)

    def __call__(self, model, optimizer, epoch, loss, metrics):
        """
        Save the model checkpoint if the metric improves. If the metric is not
        found in the metrics dictionary, the loss is used.

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
        metrics : dict
            Dictionary of metrics for the epoch.
        """
        if self.metric_name not in metrics:
            current_value = loss
        else:
            current_value = metrics.get(self.metric_name, None)
        if current_value is None:
            return
        
        if (self.mode == "min" and current_value < self.best_metric) or \
           (self.mode == "max" and current_value > self.best_metric):
            self.best_metric = current_value
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=loss,
                checkpoint_dir=self.checkpoint_dir,
                filename=f"best_model_epoch{epoch}.pth"
            )
            tqdm.write(f"New best model saved at epoch {epoch} with {self.metric_name}: {current_value:.6f}")
            
class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    a certain number of epochs.
    """
    def __init__(self, patience=5, metric_name="valid_loss", mode="min"):
        self.patience = patience
        self.metric_name = metric_name
        self.mode = mode
        self.counter = 0
        self.best_metric = float("inf") if mode == "min" else float("-inf")
        self.early_stop = False

    def __call__(self, current_value):
        """
        Check if the model training should stop.

        Parameters
        ----------
        current_value : float
            Current value of the monitored metric.
        """
        if (self.mode == "min" and current_value < self.best_metric) or \
           (self.mode == "max" and current_value > self.best_metric):
            self.best_metric = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False