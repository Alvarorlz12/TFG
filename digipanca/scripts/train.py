import os
import shutil
import argparse
import torch

from pathlib import Path
from torch.utils.data import DataLoader

from src.utils.config import load_config
from src.models import UNet, CustomDeepLabV3
from src.losses import MulticlassDiceLoss, CombinedLoss
from src.data.dataset import PancreasDataset
from src.training.trainer import Trainer, _SUMMARY
from src.utils.logger import Logger
from src.utils.notifier import Notifier

def get_model(config):
    """Initialize model based on configuration."""
    model_type = config['model']['type']
    
    if model_type == 'unet':
        return UNet(
            in_channels=config['model']['in_channels'],
            out_channels=config['model']['out_channels'],
            init_features=config['model']['init_features']
        )
    elif model_type == 'deeplabv3':
        return CustomDeepLabV3(
            num_classes=config['model']['num_classes'],
            dropout_rate=config['model']['dropout_rate'],
            pretrained=config['model']['pretrained']
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def get_loss_fn(config):
    """Initialize loss function based on configuration."""
    loss_type = config['training']['loss_function']

    if loss_type == 'MulticlassDiceLoss':
        return MulticlassDiceLoss()
    elif loss_type == 'CombinedLoss':
        weights = config['training']['loss_params'].get('weights', None)
        if weights is not None:
            device = torch.device(
                config["training"]["device"] if torch.cuda.is_available() else "cpu"
            )
            weights = torch.tensor(weights).to(device)
        return CombinedLoss(
            alpha=config['training']['loss_params']['alpha'],
            beta=config['training']['loss_params']['beta'],
            class_weights=weights
        )
    else:
        raise ValueError(f"Unsupported loss function: {loss_type}")

def main():
    parser = argparse.ArgumentParser(
        description='Train pancreas segmentation model'
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--experiment', type=str, required=True,
                        help='Experiment name')
    parser.add_argument('--notify', action='store_true',
                        help='Send notification to Telegram')
    parser.add_argument('--keep', action='store_true',
                        help='Keep experiment directory if exists')
    args = parser.parse_args()

    # Initialize notifier only if enabled
    notifier = Notifier(args.experiment) if args.notify else None
    
    # Load configuration
    config = load_config(args.config)
    RAW_DIR = config['data']['raw_dir']
    
    # Show experiment information
    print("Experiment:", args.experiment)
    print("Description:", config['description'])
    
    # Set up experiment directory
    print("📂 Setting up experiment directory...")
    experiment_dir = Path(f"experiments/{args.experiment}")
    # Clear experiment directory if exists
    if experiment_dir.exists() and not args.keep:
        shutil.rmtree(experiment_dir)
        print(f"\tClearing previous data...")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize logger
    logger = Logger(log_dir=f"{experiment_dir}/logs", verbosity="INFO")
    
    # Create dataset and data loaders
    sample_dirs = [os.path.join(RAW_DIR, sd) for sd in os.listdir(RAW_DIR)]
    train_dataset = PancreasDataset(
        sample_dirs=sample_dirs,
        split_path=config['data']['split_path'],
        split_type='train',
        resize=tuple(config['data']['input_size'])
    )
    val_dataset = PancreasDataset(
        sample_dirs=sample_dirs,
        split_path=config['data']['split_path'],
        split_type='val',
        resize=tuple(config['data']['input_size'])
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers']
    )
    
    # Initialize model
    model = get_model(config)
    
    # Initialize loss function
    loss_fn = get_loss_fn(config)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config['training'],
        experiment_dir=experiment_dir,
        logger=logger,
        notifier=notifier
    )

    # Update summary for notifier
    _SUMMARY.update({
        'experiment': args.experiment,
        'description': config['description'],
        'model_type': config['model']['type'],
        'epochs': config['training']['num_epochs'],
        'batch_size': config['data']['batch_size'],
        'learning_rate': config['training']['learning_rate'],
        'optimizer': 'AdamW',
        'loss_function': config['training']['loss_function']
    })

    # Notify training start if enabled
    if args.notify:
        notifier.send_start_message(_SUMMARY)
    
    # Train model
    print("📉 Training model...")
    trainer.train()

    # Notify training end if enabled
    if args.notify:
        notifier.send_end_message(_SUMMARY)


if __name__ == '__main__':
    main()