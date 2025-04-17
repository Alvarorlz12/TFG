import os
import time
import torch
import json
from tqdm import tqdm

from src.training.callbacks import ModelCheckpoint, EarlyStopping
from src.metrics.segmentation import SegmentationMetrics
from src.metrics.sma import SegmentationMetricsAccumulator as SMA
from src.utils.checkpoints import load_checkpoint

_SUMMARY = {}

class Trainer:
    def __init__(self, model, loss_fn, optimizer, train_loader, val_loader,
                 config, experiment_dir, logger=None, notifier=None,
                 checkpoint_path=None):
        """
        Initialize the trainer with the model, loss function, optimizer, 
        data loaders, and configuration.

        Parameters
        ----------
        model : torch.nn.Module
            Model to train.
        loss_fn : torch.nn.Module
            Loss function.
        optimizer : torch.optim.Optimizer
            Optimizer to use for training.
        train_loader : torch.utils.data.DataLoader
            DataLoader for training data.
        val_loader : torch.utils.data.DataLoader
            DataLoader for validation data.
        config : dict
            Configuration dictionary.
        experiment_dir : str
            Directory of the experiment, used for saving checkpoints.
        logger : Logger, optional
            Logger for logging the training metrics.
        notifier : Notifier, optional
            Notifier for sending messages to Telegram.
        checkpoint_path : str, optional
            Path to the checkpoint file to load the model from. If None,
            the model is trained from scratch.
        """
        self.device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = config["num_epochs"]
        self.checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
        self.metrics_dir = os.path.join(experiment_dir, "metrics")
        self.logger = logger
        # self.metrics = SegmentationMetrics()
        self.metrics = SMA(
            zero_division="nan",
            include_background=config['loss_params'].get("include_background", True)
        )
        self.notifier = notifier

        # Callbacks
        self.checkpoint_callback = ModelCheckpoint(
            checkpoint_dir=self.checkpoint_dir,
            metric_name="val_loss",
            mode="min"
        )
        self.early_stopping = EarlyStopping(
            patience=config["patience"],
            metric_name="val_loss",
            mode="min"
        )

        # Create the checkpoint directory and metrics directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)

        # Metrics file
        self.metrics_file = os.path.join(self.metrics_dir, "metrics.json")
        if not os.path.exists(self.metrics_file):
            with open(self.metrics_file, "w") as f:
                json.dump({"train": [], "val": []}, f, indent=4)

        # Load checkpoint if provided
        self.start_epoch = 0
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"🔁 Resuming from checkpoint: {checkpoint_path}")
            self.start_epoch, _ = load_checkpoint(self.model, self.optimizer, checkpoint_path)

    def _save_metrics(self, epoch, train_loss, val_loss, train_metrics, 
                      val_metrics, is_best=False):
        """
        Save the metrics to the metrics file.

        Parameters
        ----------
        epoch : int
            Current epoch.
        train_loss : float
            Training loss.
        val_loss : float
            Validation loss.
        train_metrics : dict
            Training metrics.
        val_metrics : dict
            Validation metrics.
        is_best : bool, optional
            Whether the model is the best model. If True, save the metrics
            with the key "best" too.
        """
        with open(self.metrics_file, "r") as f:
            metrics_data = json.load(f)

        # Append the metrics for the current epoch
        metrics_data["train"].append({
            "epoch": epoch,
            "loss": train_loss,
            **train_metrics
        })
        metrics_data["val"].append({
            "epoch": epoch,
            "loss": val_loss,
            **val_metrics
        })

        if is_best:
            metrics_data["best"] = {
                "epoch": epoch,
                "loss": val_loss,
                **val_metrics
            }

        with open(self.metrics_file, "w") as f:
            json.dump(metrics_data, f, indent=4)

    def train_step(self, epoch):
        """
        Train the model for one epoch.

        Parameters
        ----------
        epoch : int
            Current epoch.

        Returns
        -------
        float
            Average loss for the epoch.
        dict
            Training metrics for the epoch.
        """
        self.model.train()
        train_loss = 0.0
        all_metrics = {}
        train_loop = tqdm(
            self.train_loader,
            desc=f"[Train {epoch+1}/{self.num_epochs}]",
            leave=False,
            colour="green"
        )

        for images, masks, _ in train_loop:
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            if isinstance(outputs, dict):
                outputs = outputs["out"]

            loss = self.loss_fn(outputs, masks)
            # Check if loss is CombinedLoss which returns a tuple of losses
            if isinstance(loss, tuple):
                loss = loss[0]  # Use only the first loss

            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            batch_metrics = self.metrics.update(outputs.detach(), masks)

            if self.logger:
                self.logger.log(
                    {"train_loss": loss.item(), **batch_metrics},
                    step=epoch+1
                )

            train_loop.set_postfix(
                loss=f"{loss.item():.4f}",
                dice=f"{batch_metrics.get('dice', 0):.4f}"
            )

            # Free up memory
            del images, masks, outputs, loss
            torch.cuda.empty_cache()
            
        all_metrics = self.metrics.aggregate()
        self.metrics.reset()
        train_metrics = {
            k: v for k, v in all_metrics.items()
        }
        avg_loss = train_loss / len(self.train_loader)
        return avg_loss, train_metrics

    def val_step(self, epoch):
        """
        Validate the model for one epoch.

        Parameters
        ----------
        epoch : int
            Current epoch.

        Returns
        -------
        float
            Average loss for the epoch.
        dict
            Validation metrics for the epoch.
        """
        self.model.eval()
        val_loss = 0.0
        all_metrics = {}
        val_loop = tqdm(
            self.val_loader,
            desc=f"[Valid {epoch+1}/{self.num_epochs}]",
            leave=False,
            colour="blue"
        )

        with torch.no_grad():
            for images, masks, _ in val_loop:
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(images)
                
                if isinstance(outputs, dict):
                    outputs = outputs["out"]

                loss = self.loss_fn(outputs, masks)

                # Check if loss is CombinedLoss which returns a tuple of losses
                if isinstance(loss, tuple):
                    loss = loss[0]  # Use only the first loss

                val_loss += loss.item()
                batch_metrics = self.metrics.update(outputs, masks)

                if self.logger:
                    self.logger.log(
                        {"valid_loss": loss.item(), **batch_metrics},
                        step=epoch+1
                    )

                val_loop.set_postfix(loss=loss.item())

                # Free up memory
                del images, masks, outputs, loss
                torch.cuda.empty_cache()

        all_metrics = self.metrics.aggregate()
        self.metrics.reset()
        val_metrics = {
            k: v for k, v in all_metrics.items()
        }
        avg_loss = val_loss / len(self.val_loader)
        return avg_loss, val_metrics

    def train(self):
        """
        Train the model for the specified number of epochs and return the
        training and validation losses.
        """
        loop = tqdm(
            range(self.start_epoch, self.num_epochs),
            colour="red",
            initial=self.start_epoch,
            total=self.num_epochs    
        )
        loop.set_description(f"Epoch [{self.start_epoch}/{self.num_epochs}]")
        loop.set_postfix(train_loss="N/A", valid_loss="N/A",train_dice="N/A", valid_dice="N/A")

        start_time = time.time()
        try:
            for epoch in loop:
                train_loss, train_metrics = self.train_step(epoch)
                val_loss, val_metrics = self.val_step(epoch)

                if self.notifier and (epoch + 1) % max(1, self.num_epochs // 10) == 0:
                    self.notifier.send_progress_message(
                        current_epoch=epoch + 1,
                        total_epochs=self.num_epochs,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        train_dice=train_metrics.get('dice', 0),
                        val_dice=val_metrics.get('dice', 0)
                    )

                # Save the model checkpoint if it is the best
                is_best = self.checkpoint_callback(
                    self.model,
                    self.optimizer,
                    epoch+1,
                    val_loss,
                    val_metrics
                )

                # Save the metrics. If it is the best model, save the metrics
                # with the key "best" too
                self._save_metrics(epoch+1, train_loss, val_loss, train_metrics,
                                val_metrics, is_best=is_best)

                # Early stopping
                if self.early_stopping(val_loss):
                    tqdm.write(f"Early stopping at epoch {epoch+1}")
                    _SUMMARY["completed_epochs"] = epoch + 1
                    break

                # Update progress bar
                loop.set_description(f"Epoch [{epoch+1}/{self.num_epochs}]")
                loop.set_postfix(
                    train_loss=f"{train_loss:.4f}",
                    valid_loss=f"{val_loss:.4f}",
                    train_dice=f"{train_metrics.get('dice', 0):.4f}",
                    valid_dice=f"{val_metrics.get('dice', 0):.4f}"
                )
        except Exception as e:
            tqdm.write(f"An error occurred during training: {str(e)}")
            if self.notifier is not None:
                self.notifier.send_error_message(str(e), epoch+1)
            raise e

        total_time = time.time() - start_time
        hours, rem = divmod(total_time, 3600)
        minutes, seconds = divmod(rem, 60)
        tqdm.write(f"Training completed in {int(hours):0>2}:{int(minutes):0>2}:{seconds:05.2f}")
        # Update summary for notifier
        _SUMMARY.update({
            "train_loss": train_loss,
            "train_metrics": train_metrics,
            "val_loss": val_loss,
            "metrics": val_metrics,
            "best_model": self.checkpoint_callback.get_best_model_performance(),
            "training_time": total_time
        })