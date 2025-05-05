import os
import glob
import argparse
import torch

from pathlib import Path

from src.data.split_data import load_train_test_split
from src.utils.config import load_config
from src.utils.evaluation import evaluate_model, load_trained_model

def main():
    parser = argparse.ArgumentParser(description="Experiemnt evaluation")

    parser.add_argument(
        '--experiment_folder', type=str,
        help="Path to the experiment folder (contains multiple checkpoints)"
    )
    parser.add_argument(
        '--config', type=str, required=True,
        help="Configuration file"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    config_device = config['training']['device']
    device = torch.device(config_device if torch.cuda.is_available() else "cpu")

    # Train directory because we are evaluating the model on the training set
    # using the validation split
    test_dir = os.path.join(config["data"]["processed_dir"], "train")
    
    # Prepare parameters
    ckpt_paths = []
    out_dirs = []
    patient_ids = []

    # Search for fold directories
    folds = sorted(
        d for d in os.listdir(args.experiment_folder)
        if os.path.isdir(os.path.join(args.experiment_folder, d)) and d.startswith('fold_')
    )
    if not folds:
        raise FileNotFoundError("No fold directories found in the experiment folder.")
    
    for fold in folds:
        fold_dir = os.path.join(args.experiment_folder, fold)
        ckpt_dir = os.path.join(fold_dir, 'checkpoints')

        # Find best model checkpoint(s)
        candidates = glob.glob(os.path.join(ckpt_dir, 'best_model_*.pth'))
        if not candidates:
            print(f'Warning: no checkpoints found in {ckpt_dir}, skipping {fold_dir}')
            continue

        # Pick the latest epoch by numeric sort
        best_ckpt = sorted(
            candidates,
            key=lambda p: int(os.path.basename(p).split('epoch')[1].split('.pth')[0])
        )[-1]

        ckpt_paths.append(best_ckpt)
        out_dirs.append(fold_dir)
        patient_ids.append(
            load_train_test_split(
                os.path.join(fold_dir, 'split_data.json')
            )['val']
        )

    # Loop through each checkpoint and evaluate
    for ckpt, out_dir, pids in zip(ckpt_paths, out_dirs, patient_ids):
        print(f"Evaluating checkpoint: {ckpt}")
        model = load_trained_model(config, ckpt).to(device)
        evaluate_model(model, config, out_dir, device, test_dir, pids)

if __name__ == '__main__':
    main()
