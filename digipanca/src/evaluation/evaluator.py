import os

try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm

from src.metrics.sma import SegmentationMetricsAccumulator as SMA
from src.training.setup.transforms_factory import get_transforms
from src.utils.export import write_csv_from_dict
from src.utils.data import get_patients_in_processed_folder
from src.inference.predicter import Predicter2D, Predicter3D

class Evaluator:
    def __init__(self, model, config, test_dir, device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.test_dir = test_dir
        self.transform = get_transforms(config)
        self.sma_patient = SMA(include_background=False)
        self.sma_global = SMA(include_background=False)
        self.test_dir = test_dir
        if config['data'].get('is_3d', False):
            self.predicter = Predicter3D(model, config, device, test_dir)
        else:
            self.predicter = Predicter2D(model, config, device, test_dir)

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
        # Get predictions and ground truth
        pred_volume, gt_volume = self.predicter.predict_patient(patient_id)

        # Update metrics
        _ = self.sma_patient.update(pred_volume, gt_volume) # Patient accumulator
        _ = self.sma_global.update(pred_volume, gt_volume)  # Global accumulator

        # Get aggregated scores and confusion matrix
        p_metrics = self.sma_patient.aggregate()
        p_cm = self.sma_patient.aggregate_global_cm()
        
        self.sma_patient.reset() # Reset patient accumulator

        return p_metrics, p_cm