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

#region AUXILIARY
def collect_fold_summaries(experiment_dir):
    summary_files = sorted(Path(experiment_dir).rglob("fold_*/summary.json"))
    all_data = []

    for f in summary_files:
        with open(f, 'r') as file:
            data = json.load(file)
            all_data.append(data)

    return all_data

def average_dicts(dicts):
    """Average values of dictionaries with the same keys."""
    keys = dicts[0].keys()
    result = {}
    for key in keys:
        values = [d[key] for d in dicts]
        result[key] = float(np.mean(values))
    return result

def build_average_summary(all_data):
    num_folds = len(all_data)
    base = all_data[0]  # Use first fold as reference for non-metric fields

    avg_summary = {
        "experiment": base["experiment"],
        "description": base["description"],
        "start_time": base["start_time"],
        "config_file": base["config_file"],
        "model_type": base["model_type"],
        "epochs": base["epochs"],
        "batch_size": base["batch_size"],
        "learning_rate": base["learning_rate"],
        "optimizer": base["optimizer"],
        "loss_function": base["loss_function"],
        "experiment_dir": str(Path(base["experiment_dir"]).parent),  # directory without fold
        "train_loss": float(np.mean([f["train_loss"] for f in all_data])),
        "val_loss": float(np.mean([f["val_loss"] for f in all_data])),
        "train_metrics": average_dicts([f["train_metrics"] for f in all_data]),
        "metrics": average_dicts([f["metrics"] for f in all_data]),
        "best_model": {
            "epoch": float(np.mean([f["best_model"]["epoch"] for f in all_data])),
            "loss": float(np.mean([f["best_model"]["loss"] for f in all_data])),
            "metrics": average_dicts([f["best_model"]["metrics"] for f in all_data])
        },
        "training_time": float(np.sum([f["training_time"] for f in all_data])),
        "training_time_mean": float(np.mean([f["training_time"] for f in all_data])),
        "folds": num_folds,
        "end_time": all_data[-1]["end_time"]
    }

    return avg_summary

#endregion

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
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from a previous checkpoint (defined in config or CLI)')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to specific checkpoint to resume from (overrides config)')
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

    # Resume training if specified
    resume_path = args.resume_from or config['training'].get('checkpoint_path', None)

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

    # Create split manager
    split_manager = SplitManager(split_data=config['data']['split_file'])
    
    # Initialize logger
    logger = Logger(log_dir=f"{experiment_dir}/logs", verbosity="INFO")

    # Load transforms and augmentations
    transform = get_transforms(config)
    augment = get_augment(config)

    # Iterate over the splits and create datasets
    for i, split in enumerate(split_manager):
        print(f"📂 Starting Fold {i + 1}/{len(split_manager)}")
        split_dir = experiment_dir / f"fold_{i+1}"
        split_dir.mkdir(parents=True, exist_ok=True)
        # Save the split data in the experiment directory
        with open(split_dir / 'split_data.json', 'w') as f:
            json.dump(split, f, indent=4)

        # Create dataset and data loaders
        train_dataset = get_dataset(
            config=config,
            split_data=split,
            split_type='train',
            transform=transform,
            augment=augment
        )
        val_dataset = get_dataset(
            config=config,
            split_data=split,
            split_type='val',
            transform=transform
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
            experiment_dir=split_dir,
            logger=logger,
            notifier=notifier,
            checkpoint_path=resume_path if args.resume else None
        )

        # Update summary for notifier
        _SUMMARY.update({
            'experiment': args.experiment,
            'fold': i + 1,
            'total_folds': len(split_manager),
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

        if resume_path is not None:
            _SUMMARY['resume_path'] = resume_path

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

        # Save _SUMMARY in experiment directory
        with open(f"{split_dir}/summary.json", 'w') as f:
            json.dump(_SUMMARY, f, indent=4)

        print(f"📂 Fold {i + 1}/{len(split_manager)} completed.")

    # Collect all fold summaries
    all_data = collect_fold_summaries(experiment_dir)
    # Build average summary
    avg_summary = build_average_summary(all_data)

    # Notify avergage if enabled
    if args.notify and not args.only_save:
        notifier.send_average_message(avg_summary)

    # Save results in Google Sheets if enabled
    if args.notify or args.only_save:
        notifier.save_results(avg_summary)

    # Save _SUMMARY in experiment directory
    with open(f"{experiment_dir}/summary.json", 'w') as f:
        json.dump(avg_summary, f, indent=4)
#endregion

if __name__ == '__main__':
    main()