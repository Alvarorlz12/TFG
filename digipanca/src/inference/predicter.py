import os
import torch

try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.data.dataset2d import PancreasDataset2D
from src.data.dataset3d import PancreasDataset3D
from src.training.setup.transforms_factory import get_transforms

class Predicter:
    def __init__(self, model, config, device, test_dir, transform=None):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.test_dir = test_dir
        self.transform = transform if transform else get_transforms(config)

    def predict_patient(self, patient_id):
        raise NotImplementedError("Implemented in subclass.")
    
class Predicter2D(Predicter):
    def predict_patient(self, patient_id):
        self.model.eval()

        p_dataset = PancreasDataset2D(
            data_dir=self.test_dir,
            transform=self.transform,
            load_into_memory=False,
            patient_ids=[patient_id],
            verbose=False
        )

        p_dl = DataLoader(
            p_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True
        )

        patient_loop = tqdm(
            p_dl,
            leave=False,
            colour="blue"
        )
        patient_loop.set_description(f"Patient {patient_id}")

        all_preds, all_gts = [], []

        with torch.no_grad():
            for images, masks, _ in patient_loop:
                images, masks = images.to(self.device), masks.to(self.device)
                preds = self.model(images)
                if isinstance(preds, dict):
                    preds = preds['out']
                all_preds.append(preds.cpu())
                all_gts.append(masks.cpu())

        return (
            torch.cat(all_preds, dim=0).permute(1, 0, 2, 3).unsqueeze(0).to(self.device),
            torch.cat(all_gts, dim=0).unsqueeze(0).to(self.device)
        )
    
class Predicter3D(Predicter):
    def predict_patient(self, patient_id):
        self.model.eval()

        p_dataset = PancreasDataset3D(
            data_dir=self.test_dir,
            transform=self.transform,
            load_into_memory=False,
            patient_ids=[patient_id],
            verbose=False
        )

        p_dl = DataLoader(
            p_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True
        )

        patient_loop = tqdm(
            p_dl,
            leave=False,
            colour="blue"
        )
        patient_loop.set_description(f"Patient {patient_id}")

        all_preds = []
        all_slices = p_dataset.get_patient_subvolumes_slices(patient_id)
        D = all_slices[-1][1] + 1

        with torch.no_grad():
            for images, masks, _ in patient_loop:
                images, masks = images.to(self.device), masks.to(self.device)
                preds = self.model(images)
                if isinstance(preds, dict):
                    preds = preds['out']
                all_preds.append(F.softmax(preds, dim=1))

        # Post-process: get a single 3D volume for the patient
        all_preds = torch.cat(all_preds, dim=0)
        C, _, H, W = all_preds.shape[1:]
        sum_probs = torch.zeros((C, D, H, W), dtype=torch.float64, device=all_preds.device)
        count = torch.zeros((D, H, W), dtype=torch.int8, device=all_preds.device)

        for i, (start, end) in enumerate(all_slices):
            sum_probs[:, start:end+1, :, :] += all_preds[i]
            count[start:end+1, :, :] += 1

        avg_probs = sum_probs / count.unsqueeze(0)

        return (
            avg_probs.unsqueeze(0).to(self.device),
            (p_dataset.get_patient_volume(patient_id)[1]).to(self.device)
        )