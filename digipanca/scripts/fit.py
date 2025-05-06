import os
import shutil
import argparse
import torch
import json
import torch
import numpy as np

from pathlib import Path
from torch.utils.data import DataLoader
from datetime import datetime

from src.data.split_manager import SplitManager
from src.utils.config import load_config
from src.training.trainer import Trainer, _SUMMARY
from src.utils.logger import Logger
from src.utils.notifier import Notifier
from src.training.setup import (
    get_model,
    get_loss_fn,
    get_dataset,
    get_transforms,
    get_augment
)
from src.training.utils import set_seed

#region MAIN
def main():
    torch.cuda.empty_cache()
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Fit pancreas segmentation model'
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--experiment', type=str, required=True,
                        help='Experiment name')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs to fit the model')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path to save the model weights')
    args = parser.parse_args()

    # Start time
    start_time = datetime.now()
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    
    # Load configuration
    config = load_config(args.config)

    # Set random seed for reproducibility
    seed = config['training'].get('seed', 42)
    set_seed(seed)
    
    # Show experiment information
    print("Experiment:", args.experiment)
    print("Description:", config['description'])
    
    # Set up experiment directory
    print("📂 Setting up experiment directory...")
    experiment_root = Path(f"experiments/{args.experiment}")
    experiment_dir = experiment_root / f"{args.experiment}_{timestamp}"
    # Clear experiment directory if exists
    if experiment_dir.exists():
        shutil.rmtree(experiment_dir)
        print(f"\tClearing previous data...")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    print(f"\tExperiment directory: {experiment_dir}")
    
    # Initialize logger
    logger = Logger(log_dir=f"{experiment_dir}/logs", verbosity="INFO")

    # Load transforms and augmentations
    transform = get_transforms(config)
    augment = get_augment(config)

    # Create dataset and data loaders
    train_dataset = get_dataset(
        config=config,
        split_data=None,    # No split data for training
        split_type='train',
        transform=transform,
        augment=augment
    )

    # Drop last batch if it has only one sample. Fix BatchNorm error
    drop_last = len(train_dataset) % config['data']['batch_size'] == 1
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        persistent_workers=True,
        drop_last=drop_last
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
        val_loader=None,
        config=config['training'],
        experiment_dir=experiment_dir,
        logger=logger,
        notifier=None,
        checkpoint_path=None
    )

    # Update summary for notifier
    _SUMMARY.update({
        'experiment': args.experiment,
        'description': config['description'],
        'start_time': start_time.strftime('%d-%m-%Y %H:%M:%S'),
        'config_file': args.config,
        'model_type': config['model']['type'] + 
            (' (MONAI)' if config['model'].get('use_monai', False) else ''),
        'epochs': config['training']['num_epochs'],
        'batch_size': config['data']['batch_size'],
        'learning_rate': config['training']['learning_rate'],
        'optimizer': 'AdamW',
        'loss_function': config['training']['loss_function'] +
        (' \\(MONAI\\)' if config['training'].get('use_monai_loss', False) else ''),
        'experiment_dir': str(experiment_dir)
    })
    
    # Fit the model
    print("🚀 Fitting model...")
    trainer.fit(
        num_epochs=args.num_epochs,
        save_path=args.save_path
    )

    # End time
    _SUMMARY['end_time'] = datetime.now().strftime('%d-%m-%Y %H:%M:%S')

    # Save _SUMMARY in experiment directory
    with open(f"{experiment_dir}/summary.json", 'w') as f:
        json.dump(_SUMMARY, f, indent=4)
#endregion

if __name__ == '__main__':
    main()