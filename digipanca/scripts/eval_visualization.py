import argparse
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2
from pathlib import Path
from torch.utils.data import DataLoader

from src.utils.config import load_config
from src.data.transforms import build_transforms_from_config
from src.models import UNet, CustomDeepLabV3
from src.data.dataset import PancreasDataset

# Obtener el mejor modelo del experimento
def get_best_model_path(experiment, time_stamp):
    experiment_path = Path(f"experiments/{experiment}/{experiment}_{time_stamp}")
    experiment_path = experiment_path / "checkpoints"
    if not experiment_path.exists():
        raise FileNotFoundError(f"Not found: {experiment_path}")
    
    # List best models in the experiment (ideally, there should be only one)
    best_models = list(experiment_path.glob("best_model_epoch*.pth"))

    if not best_models:
        raise FileNotFoundError(f"No best models found in {experiment_path}")
    
    # If there are multiple best models, select the one with the highest epoch
    best_model_path = sorted(best_models)[-1]
    
    return best_model_path

# Cargar el modelo
def get_model(config, checkpoint_path):
    model_type = config['model']['type']
    
    if model_type == 'unet':
        model = UNet(
            in_channels=config['model']['in_channels'],
            out_channels=config['model']['out_channels'],
            init_features=config['model']['init_features']
        )
    elif model_type == 'deeplabv3':
        model = CustomDeepLabV3(
            num_classes=config['model']['num_classes'],
            dropout_rate=config['model']['dropout_rate'],
            pretrained=False
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_state_dict = checkpoint['model']
    model.load_state_dict(
        {k: v for k, v in model_state_dict.items() if "aux_classifier" not in k},
        strict=False
    )
    model.eval()
    return model

# Visualization
def visualize_model_predictions(model, test_loader, device, num_images=3, alpha=0.6):
    """
    Visualizes multiple model predictions in separate windows.

    Parameters
    ----------
    model : torch.nn.Module
        The trained model.
    test_loader : torch.utils.data.DataLoader
        The test data loader.
    device : torch.device
        The device to run the model on.
    num_images : int
        The number of images to visualize. Default is 3.
    alpha : float
        The transparency of the overlay. Default is 0.6.

    Returns
    -------
    None
    """
    cmap = mcolors.ListedColormap(['green', 'purple', 'red', 'blue'])
    boundaries = [0.5, 1.5, 2.5, 3.5, 4.5]
    norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)

    model.eval()
    indices = random.sample(range(len(test_loader.dataset)), num_images)

    for i, idx in enumerate(indices):
        image, mask, patient_id = test_loader.dataset[idx] 
        image, mask = image.to(device), mask.to(device)

        with torch.no_grad():
            output = model(image.unsqueeze(0))["out"]
            prediction = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
            ground_truth = mask.cpu().numpy()

        # Compute Dice Score for each class
        dice_scores = [
            2 * np.sum((prediction == class_idx) & (ground_truth == class_idx)) / 
            (np.sum(prediction == class_idx) + np.sum(ground_truth == class_idx) + 1e-6) 
            for class_idx in range(1, 5)
        ]

        # Create overlays
        mask_overlay = np.where(ground_truth > 0, ground_truth, np.nan)
        prediction_overlay = np.where(prediction > 0, prediction, np.nan)

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(12, 5))

        # Original image
        axes[0].imshow(image.squeeze().cpu(), cmap='gray')
        axes[0].set_title('Original', fontsize=12)
        axes[0].axis('off')

        # Ground Truth
        axes[1].imshow(image.squeeze().cpu(), cmap='gray')
        axes[1].imshow(mask_overlay, cmap=cmap, norm=norm, alpha=alpha)
        axes[1].set_title('Ground Truth', fontsize=12)
        axes[1].axis('off')

        # Model Prediction
        axes[2].imshow(image.squeeze().cpu(), cmap='gray')
        axes[2].imshow(prediction_overlay, cmap=cmap, norm=norm, alpha=alpha)
        axes[2].set_title('Prediction', fontsize=12)
        axes[2].axis('off')

        # Add Dice Scores in a single line below the images
        dice_text = " | ".join([f"Class {i+1}: {dice:.4f}" for i, dice in enumerate(dice_scores)])
        fig.text(0.5, 0.02, dice_text, ha='center', fontsize=11, color='black')
        fig.text(0.5, 0.06, "Dice Scores", ha='center', fontsize=12, fontweight='bold')

        # Set window title with patient ID and slice index
        slice_idx = idx - test_loader.dataset.get_initial_slice_idx(patient_id)
        fig.suptitle(f"Patient {patient_id}", fontsize=16, fontweight="bold")
        fig.text(0.5, 0.9, f"Slice {slice_idx}", ha='center', fontsize=14, fontweight="bold")

        # Colorbar (properly positioned)
        cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])  # Ajuste preciso
        colorbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
        colorbar.set_ticks([1, 2, 3, 4])
        colorbar.set_ticklabels(['Pancreas', 'Tumor', 'Arteries', 'Veins'])
        colorbar.ax.tick_params(labelsize=10)

        # Ajuste de espaciado
        plt.subplots_adjust(wspace=0.15, right=0.88, bottom=0.12, top=0.85)
        plt.show(block=(i == num_images - 1))

def main():
    parser = argparse.ArgumentParser(description="Model evaluation")
    parser.add_argument('--config', type=str, required=True, help="Configuration file")
    parser.add_argument('--experiment', type=str, required=True, help="Experiment name")
    parser.add_argument('--timestamp', type=str, required=True, help="Timestamp of the experiment")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Get the best model path
    best_model_path = get_best_model_path(args.experiment, args.timestamp)

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(config, best_model_path).to(device)

    # Build transforms pipeline
    transform = build_transforms_from_config(config.get('transforms', None))

    # Load the validation dataset
    val_dataset = PancreasDataset(
        data_dir=config['data']['raw_dir'],
        split_file=config['data']['split_path'],
        split_type='val',
        transform=transform
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )

    # Visualize model predictions
    visualize_model_predictions(model, val_loader, device, num_images=10)

if __name__ == '__main__':
    main()
