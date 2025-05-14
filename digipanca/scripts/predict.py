import os
import argparse
import torch

from tqdm.auto import tqdm
from nibabel.orientations import axcodes2ornt, ornt_transform

from src.data.preprocessing import reverse_preprocess_mask
from src.utils.config import load_config, read_config
from src.utils.evaluation import load_trained_model
from src.inference.predicter import Predicter3D, Predicter2D
from src.data.dataset2d import PancreasDataset2D
from src.data.dataset3d import PancreasDataset3D
from src.utils.visualization.animation import create_2d_animation
from src.training.setup.transforms_factory import get_transforms
from src.utils.data import get_original_info, get_patients_in_processed_folder, save_segmentation_mask
from src.utils.tensors import prepare_tensors_for_visualization

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
    parser.add_argument(
        '--save_masks', action='store_true',
        help="Save the segmentation masks as NIfTI files"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    config_device = config['training']['device']
    device = torch.device(config_device if torch.cuda.is_available() else "cpu")

    # Get the preprocessing config
    preprocess_config = config['data'].get('preprocess_config', None)
    if preprocess_config is None:
        # Infer the preprocessing config from the processed data directory
        folder = os.path.basename(config['data']['processed_dir'])
        preprocess_config = os.path.join(
            'configs/data',
            f'preprocess_{folder}_{args.split}.yaml')
        
    # Load the preprocessing config
    preprocess_config = read_config(preprocess_config)

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
        # volume, _ = dataset.get_patient_volume(patient_id)
        # create_2d_animation(
        #     predictions,
        #     masks,
        #     patient_id,
        #     output_dir=patient_dir,
        #     volume=None,
        #     filename=f'{patient_id}_no_volume.gif',
        #     alpha=0.8
        # )
        # create_2d_animation(
        #     predictions,
        #     masks,
        #     patient_id,
        #     output_dir=patient_dir,
        #     volume=volume,
        #     filename=f'{patient_id}_with_volume.gif'
        # )

        if not args.save_masks:
            continue

        # Get the original information
        original_affine, original_spacing, original_orientation = get_original_info(
            config['data']['raw_dir'],
            patient_id
        )
        print(f"Original affine: {original_affine}")
        print(f"Original spacing: {original_spacing}")
        print(f"Original orientation: {original_orientation}")
        # Get the orientation transformation matrix
        orientation_transform = None
        current_orientation = preprocess_config.get('target_orientation', None)
        if current_orientation is not None:
            orientation_transform = ornt_transform(
                axcodes2ornt(tuple(current_orientation)),
                axcodes2ornt(tuple(original_orientation))
            )

        # Prepare the predictions and masks for saving
        predictions_np, masks_np, _ = prepare_tensors_for_visualization(
            predictions,
            masks,
            volume=None
        )

        # Apply reverse preprocessing
        current_spacing = preprocess_config.get('target_spacing', None)
        print(f"Current spacing: {current_spacing}")
        if current_spacing is not None:
            current_spacing = tuple(current_spacing)
        roi = preprocess_config['data'].get('roi', None)
        if current_spacing is None:
            current_spacing = original_spacing

        print(f"Current spacing: {current_spacing}")
        r_pred, r_affine = reverse_preprocess_mask(
            mask=predictions_np,
            current_spacing=current_spacing,
            original_affine=original_affine,
            h_min=roi['h_min'] if roi is not None else 0,
            h_max=roi['h_max'] if roi is not None else 512,
            w_min=roi['w_min'] if roi is not None else 0,
            w_max=roi['w_max'] if roi is not None else 512,
            orientation_transform=orientation_transform,
            original_spacing=original_spacing
        )
        file_path = os.path.join(
            patient_dir,
            f"{patient_id}_predictions.nii.gz"
        )
        print(f"updated affine: {r_affine}")
        # Save the predictions as NIfTI files
        save_segmentation_mask(
            mask=r_pred,
            affine=r_affine,
            file_path=file_path
        )
        r_mask, r_affine = reverse_preprocess_mask(
            mask=masks_np,
            current_spacing=current_spacing,
            original_affine=original_affine,
            h_min=roi['h_min'] if roi is not None else 0,
            h_max=roi['h_max'] if roi is not None else 512,
            w_min=roi['w_min'] if roi is not None else 0,
            w_max=roi['w_max'] if roi is not None else 512,
            orientation_transform=orientation_transform,
            original_spacing=original_spacing
        )
        file_path = os.path.join(
            patient_dir,
            f"{patient_id}_masks.nii.gz"
        )
        # Save the predictions as NIfTI files
        save_segmentation_mask(
            mask=r_mask,
            affine=r_affine,
            file_path=file_path
        )

    print(f"Predictions finished for all patients. Results saved in: {args.output_dir}")    

if __name__ == '__main__':
    main()
