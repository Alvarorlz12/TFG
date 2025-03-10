import torch
from tqdm import tqdm
from src.metrics.segmentation import SegmentationMetrics
from src.utils.logger import Logger
from src.utils.checkpoints import save_checkpoint

class Trainer:
    def __init__(self, model, optimizer, criterion, device, logger=None):
        """
        Initialize the trainer with the model, optimizer, criterion, and device.

        Parameters
        ----------
        model : torch.nn.Module
            Model to train.
        optimizer : torch.optim.Optimizer
            Optimizer to use for training.
        criterion : torch.nn.Module
            Loss function.
        device : str
            Device to use for training.
        logger : Logger, optional
            Logger for logging the training metrics.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.logger = logger
        self.metrics = SegmentationMetrics()

    def train_epoch(self, train_loader, epoch):
        """
        Train the model for one epoch.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            DataLoader for training data.
        epoch : int
            Current epoch.

        Returns
        -------
        float
            Average loss for the epoch.
        """
        self.model.train()
        train_loss = 0.0

        for images, masks, _ in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            batch_metrics = self.metrics.all_metrics(outputs.detach(), masks)

            if self.logger is not None:
                self.logger.log(
                    metrics={"train_loss": loss.item(), **batch_metrics},
                    step=epoch
                )

        return train_loss / len(train_loader)
    
    def validate_epoch(self, val_loader, epoch):
        """
        Validate the model for one epoch.

        Parameters
        ----------
        val_loader : torch.utils.data.DataLoader
            DataLoader for validation data.
        epoch : int
            Current epoch.

        Returns
        -------
        float
            Average loss for the epoch.
        """
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, masks, _ in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

                val_loss += loss.item()
                batch_metrics = self.metrics.all_metrics(outputs, masks)

                if self.logger is not None:
                    self.logger.log(
                        metrics={"val_loss": loss.item(), **batch_metrics},
                        step=epoch
                    )

        return val_loss / len(val_loader)
    
    def fit(self, train_loader, val_loader, num_epochs):
        """
        Train the model for the given number of epochs.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            DataLoader for training data.
        val_loader : torch.utils.data.DataLoader
            DataLoader for validation data.
        num_epochs : int
            Number of epochs to train the model.

        Returns
        -------
        list
            Training losses for each epoch.
        list
            Validation losses for each epoch.
        """
        train_losses, val_losses = [], []

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validate_epoch(val_loader, epoch)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

        return train_losses, val_losses
    
    def close(self):
        """
        Close the logger.
        """
        if self.logger is not None:
            self.logger.close()
    
