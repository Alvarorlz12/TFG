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
from src.metrics.sma import SegmentationMetricsAccumulator as SMA
from src.training.setup.transforms_factory import get_transforms
from src.utils.export import write_csv_from_dict
from src.utils.data import get_patients_in_processed_folder

class Evaluator:
    def __init__(self, model, config, test_dir, device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.test_dir = test_dir
        self.transform = get_transforms(config)
        self.sma_patient = SMA(include_background=False)
        self.sma_global = SMA(include_background=False)

    def _export_results_to_csv(self, output_folder, metrics, cm):
        os.makedirs(output_folder, exist_ok=True)
    
        # Export metrics
        metrics_file = os.path.join(output_folder, 'evaluation_metrics.csv')
        metric_headers = [
            'patient_id',
            'dice_class_1', 'dice_class_2', 'dice_class_3', 'dice_class_4', 'dice_mean'
        ]
        metric_fields = ['dice_class_1', 'dice_class_2', 'dice_class_3', 'dice_class_4', 'dice']
        write_csv_from_dict(metrics_file, metrics, metric_fields, metric_headers)
    
        # Export confusion matrix
        cm_file = os.path.join(output_folder, 'evaluation_cm.csv')
        cm_headers = [
            'patient_id',
            'tp_class_1', 'tp_class_2', 'tp_class_3', 'tp_class_4', 'tp',
            'fp_class_1', 'fp_class_2', 'fp_class_3', 'fp_class_4', 'fp',
            'fn_class_1', 'fn_class_2', 'fn_class_3', 'fn_class_4', 'fn'
        ]
        cm_fields = [
            'tp_class_1', 'tp_class_2', 'tp_class_3', 'tp_class_4', 'tp',
            'fp_class_1', 'fp_class_2', 'fp_class_3', 'fp_class_4', 'fp',
            'fn_class_1', 'fn_class_2', 'fn_class_3', 'fn_class_4', 'fn'
        ]
        write_csv_from_dict(cm_file, cm, cm_fields, cm_headers)

        print(f"Results saved in: {output_folder}")

    def evaluate(self, patient_ids=None, csv_folder=None):
        """
        Evaluate the model on the test dataset.

        Parameters
        ----------
        patient_ids : list of str, optional
            List of patient IDs to evaluate. If None, all patients in the test
            directory will be evaluated.
        csv_folder : str, optional
            Path to the folder where the evaluation results will be saved as CSV.
            If None, results will not be saved.

        Returns
        -------
        dict
            Dictionary containing the evaluation metrics for each patient.
        dict
            Dictionary containing the confusion matrix for each patient.
        """
        if patient_ids is None:
            patient_ids = get_patients_in_processed_folder(self.test_dir)
        
        loop = tqdm(
            patient_ids,
            colour="red",
            leave=True
        )
        loop.set_description(f"Evaluating patients")

        all_metrics, all_cms = {}, {}
        
        for patient_id in loop:
            p_metrics, p_cm = self.evaluate_patient(patient_id)
            all_metrics[patient_id] = p_metrics
            all_cms[patient_id] = p_cm

        # Global results
        all_metrics['global'] = self.sma_global.aggregate()
        all_cms['global'] = self.sma_global.aggregate_global_cm()

        # Save data on CSV file if specified
        if csv_folder is not None:
            self._export_results_to_csv(csv_folder, all_metrics, all_cms)

        return all_metrics, all_cms

    def evaluate_patient(self, patient_id):
        raise NotImplementedError("Implemented in subclasses")
    
class Evaluator2D(Evaluator):
    def evaluate_patient(self, patient_id):
        self.model.eval()

        p_dataset = PancreasDataset2D(
            data_dir=self.test_dir,
            transform=self.transform,
            load_into_memory=False,
            patient_ids=[patient_id],
            verbose=False
        );

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
        all_gts = []

        with torch.no_grad():
            for images, masks, _ in patient_loop:
                images, masks = images.to(self.device), masks.to(self.device)

                outputs = self.model(images)
                
                if isinstance(outputs, dict):
                    outputs = outputs["out"]

                all_preds.append(outputs)
                all_gts.append(masks)

        # Concatenate predictions and ground truths (from 2D slices to a single 3D volume)
        all_preds = torch.cat(all_preds, dim=0).permute(1, 0, 2, 3).unsqueeze(0)
        all_gts = torch.cat(all_gts, dim=0).unsqueeze(0)

        # Update metrics
        _ = self.sma_patient.update(all_preds, all_gts) # Patient accumulator
        _ = self.sma_global.update(all_preds, all_gts)  # Global accumulator

        # Get aggregated scores and confusion matrix
        p_metrics = self.sma_patient.aggregate()
        p_cm = self.sma_patient.aggregate_global_cm()
        
        self.sma_patient.reset() # Reset patient accumulator

        return p_metrics, p_cm
    
class Evaluator3D(Evaluator):
    def evaluate_patient(self, patient_id):
        self.model.eval()

        p_dataset = PancreasDataset3D(
            data_dir=self.test_dir,
            transform=self.transform,
            load_into_memory=False,
            patient_ids=[patient_id],
            verbose=False
        );

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
        D = all_slices[-1][1] + 1 # add 1 to the total number of slices

        with torch.no_grad():
            for images, masks, _ in patient_loop:
                images, masks = images.to(self.device), masks.to(self.device)

                outputs = self.model(images)
                
                if isinstance(outputs, dict):
                    outputs = outputs["out"]

                all_preds.append(F.softmax(outputs, dim=1))

        # Post-process: get a single 3D volume for the patient
        all_preds = torch.cat(all_preds, dim=0)
        print(all_preds.shape)
        C, _, H, W = all_preds.shape[1:]
        sum_probs = torch.zeros((C, D, H, W), dtype=torch.float64)
        count = torch.zeros((D, H, W), dtype=torch.int8)

        for i, (start, end) in enumerate(all_slices):
            sum_probs[:, start:end+1, :, :] += all_preds[i]
            count[start:end+1, :, :] += 1

        avg_probs = sum_probs / count.unsqueeze(0)
        pred_vol = avg_probs.unsqueeze(0) # B, C, D, H, W

        # Get mask reconstruction
        _, recon_mask = p_dataset.get_patient_volume(patient_id)

        # Update metrics
        _ = self.sma_patient.update(pred_vol, recon_mask) # Patient accumulator
        _ = self.sma_global.update(pred_vol, recon_mask)  # Global accumulator

        # Get aggregated scores and confusion matrix
        p_metrics = self.sma_patient.aggregate()
        p_cm = self.sma_patient.aggregate_global_cm()
        
        self.sma_patient.reset() # Reset patient accumulator

        return p_metrics, p_cm