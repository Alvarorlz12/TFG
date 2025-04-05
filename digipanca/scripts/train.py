import os
import shutil
import argparse
import torch
import json
import torch

from pathlib import Path
from torch.utils.data import DataLoader
from datetime import datetime

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
        description='Train pancreas segmentation model'
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--experiment', type=str, required=True,
                        help='Experiment name')
    parser.add_argument('--notify', action='store_true',
                        help='Send notification to Telegram and save results in Google Sheets')
    parser.add_argument('--only_save', action='store_true',
                        help='Only save results in Google Sheets')
    args = parser.parse_args()

    # Start time
    start_time = datetime.now()
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")

    # Initialize notifier only if enabled (by --notify or --only_save)
    #     --notify and --only_save -> only save results in Google Sheets
    #     --notify only -> send notification to Telegram and save results in Google Sheets
    #     --only_save only -> only save results in Google Sheets (same as --notify --only_save)
    if args.notify or args.only_save:
        notifier = Notifier(args.experiment, only_save=args.only_save)
    else:
        notifier = None
    
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
        split_type='train',
        transform=transform,
        augment=augment
    )
    val_dataset = get_dataset(
        config=config,
        split_type='val',
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        persistent_workers=True
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

    # Notify training start if enabled
    if args.notify and not args.only_save:
        notifier.send_start_message(_SUMMARY)
    
    # Train model
    print("📉 Training model...")
    trainer.train()

    # End time
    _SUMMARY['end_time'] = datetime.now().strftime('%d-%m-%Y %H:%M:%S')

    # Notify training end if enabled
    if args.notify and not args.only_save:
        notifier.send_end_message(_SUMMARY)

    # Save results in Google Sheets if enabled
    if args.notify or args.only_save:
        notifier.save_results(_SUMMARY)

    # Save _SUMMARY in experiment directory
    with open(f"{experiment_dir}/summary.json", 'w') as f:
        json.dump(_SUMMARY, f, indent=4)
#endregion

if __name__ == '__main__':
    main()