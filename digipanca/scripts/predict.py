import os
import argparse
import torch

from tqdm.auto import tqdm

from src.utils.config import load_config
from src.utils.evaluation import load_trained_model
from src.inference.predicter import Predicter3D, Predicter2D
from src.data.dataset2d import PancreasDataset2D
from src.data.dataset3d import PancreasDataset3D
from src.utils.visualization.animation import create_2d_animation
from src.training.setup.transforms_factory import get_transforms
from src.utils.data import get_patients_in_processed_folder

def main():
    parser = argparse.ArgumentParser(description="Patient prediction")

    parser.add_argument(
        '--model_path', type=str,
        help="Path to the model weights file (.pth)"
    )
    parser.add_argument(
        '--config', type=str, required=True,
        help="Configuration file"
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help="Output directory to save the predictions"
    )
    parser.add_argument(
        '--split', type=str, default='train',
        help="Split to predict on (train or test)"
    )
    parser.add_argument(
        '--patient_id', type=str, default=None,
        help="Patient ID to predict. If not provided, all patients will be predicted."
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    config_device = config['training']['device']
    device = torch.device(config_device if torch.cuda.is_available() else "cpu")

    DEFAULT_RESULTS_BASE_DIR = 'models/predictions'
    if args.output_dir is None:
        model_name = os.path.basename(args.model_path).split('.')[0]
        args.output_dir = os.path.join(DEFAULT_RESULTS_BASE_DIR, model_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # Get the data directory based on the split
    data_dir = os.path.join(config["data"]["processed_dir"], args.split)
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Test directory {data_dir} does not exist.")
    
    model = load_trained_model(config, args.model_path).to(device)

    if config['data'].get('is_3d', False):
        predicter = Predicter3D(model, config, device, data_dir)
        dataset = PancreasDataset3D(
            data_dir=data_dir,
            transform=get_transforms(config),
            verbose=False
        )
    else:
        predicter = Predicter2D(model, config, device, data_dir)
        dataset = PancreasDataset2D(
            data_dir=data_dir,
            transform=get_transforms(config),
            verbose=False
        )

    # Get the patient IDs to predict
    if args.patient_id is not None:
        patient_ids = [args.patient_id]
    else:
        patient_ids = get_patients_in_processed_folder(data_dir)

    loop = tqdm(
        patient_ids,
        desc="Predicting patients",
        unit="patient",
        colour="red",
        leave=True
    )

    # Loop through each patient ID and predict
    for patient_id in loop:
        patient_dir = os.path.join(args.output_dir, patient_id)
        os.makedirs(patient_dir, exist_ok=True)
        predictions, masks = predicter.predict_patient(patient_id)

        # Create and save the animation
        volume, _ = dataset.get_patient_volume(patient_id)
        create_2d_animation(
            predictions,
            masks,
            patient_id,
            output_dir=patient_dir,
            volume=None,
            filename=f'{patient_id}_no_volume.gif',
            alpha=0.8
        )
        create_2d_animation(
            predictions,
            masks,
            patient_id,
            output_dir=patient_dir,
            volume=volume,
            filename=f'{patient_id}_with_volume.gif'
        )

    print(f"Predictions finished for all patients. Results saved in: {args.output_dir}")    

if __name__ == '__main__':
    main()
