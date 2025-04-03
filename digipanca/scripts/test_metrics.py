import torch
import csv
import numpy as np
from tqdm import tqdm
from src.models import CustomDeepLabV3 as CDL
from src.utils.config import load_config
from src.data.transforms import build_transforms_from_config
from src.data.dataset2d import PancreasDataset2D
from src.metrics.segmentation import SegmentationMetrics as SM
from src.metrics.segmentation_bak import SegmentationMetrics as SMbak

def test(csv_file='saved_metrics_test.csv'):
    MODEL_PATH = 'experiments/deep_aug_randcrop/deep_aug_randcrop_20250319_074330/checkpoints/best_model_epoch21.pth'
    CONFIG_PATH = 'configs/experiments/deep_aug_randcrop.yaml'
    NUM_CLASSES = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(CONFIG_PATH)
    transform = build_transforms_from_config(config.get('transforms', None))

    model = CDL(num_classes=NUM_CLASSES, dropout_rate=0.2, pretrained=False)
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')['model']
    model.load_state_dict(
        {k: v for k, v in checkpoint.items() if "aux_classifier" not in k},
        strict=False
    )
    model.eval()
    model.to(device)

    # Dataset
    train_dataset = PancreasDataset2D(
        data_dir='data/processed/2d/train',
        transform=transform,
        load_into_memory=False
    )

    val_dataset = PancreasDataset2D(
        data_dir='data/processed/2d/val',
        transform=transform,
        load_into_memory=False
    )

    # Eval and save Dice Scores
    # [Patient ID, Slice, Class, Softmax, Argmax, Difference]
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(['patient_id', 'slice', 'class', 'softmax', 'argmax', 'difference'])

        model.eval()
    
        for idx in tqdm(range(len(train_dataset))):
            image, mask, patient_id = train_dataset[idx] 
            image, mask = image.to(device), mask.to(device)
            mask_un = mask.unsqueeze(0)
    
            with torch.no_grad():
                output = model(image.unsqueeze(0))["out"]
    
            _, dice_scores_actual = SM.dice_coefficient(output, mask_un)
            _, dice_scores_bak = SMbak.dice_coefficient(output, mask_un)
    
            vol_slice = train_dataset.get_volume_slice_idx(idx)

            for class_idx in range(NUM_CLASSES):
                key = f'dice_class_{class_idx}'
                actual = np.round(dice_scores_actual[key], 8)
                previous = np.round(dice_scores_bak[key], 8)
                difference = np.round(actual - previous, 8)

                writer.writerow([patient_id, vol_slice, class_idx, previous, actual, difference])

        for idx in tqdm(range(len(val_dataset))):
            image, mask, patient_id = val_dataset[idx] 
            image, mask = image.to(device), mask.to(device)
            mask_un = mask.unsqueeze(0)
    
            with torch.no_grad():
                output = model(image.unsqueeze(0))["out"]
    
            _, dice_scores_actual = SM.dice_coefficient(output, mask_un)
            _, dice_scores_bak = SMbak.dice_coefficient(output, mask_un)
    
            vol_slice = val_dataset.get_volume_slice_idx(idx)

            for class_idx in range(NUM_CLASSES):
                key = f'dice_class_{class_idx}'
                actual = np.round(dice_scores_actual[key], 8)
                previous = np.round(dice_scores_bak[key], 8)
                difference = np.round(actual - previous, 8)

                writer.writerow([patient_id, vol_slice, class_idx, previous, actual, difference])

    print(f"Finished. Saved: {csv_file}")

if __name__ == "__main__":
    test()