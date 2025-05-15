import os
import torch

from src.training.setup.model_factory import get_model
from src.evaluation.evaluator import Evaluator

def evaluate_model(model, config, output_dir, device, test_dir=None, patient_ids=None):
    """
    Evaluate the model on the test dataset and save the results in the specified
    directory. The evaluation metrics are calculated and saved as CSV files.

    Parameters
    ----------
    model : torch.nn.Module
        The trained model to evaluate.
    config : dict
        Configuration dictionary containing evaluation parameters.
    output_dir : str
        Path to the directory where the evaluation results will be saved.
    device : torch.device
        The device to run the evaluation on (CPU or GPU).
    test_dir : str, optional
        Path to the test directory. If None, the test directory is taken from the
        configuration file.
    patient_ids : list of str, optional
        List of patient IDs to evaluate. If None, all patients in the test
        directory will be evaluated.
    """
    model.to(device)
    model.eval()

    # Get the test directory
    if test_dir is None:
        test_dir = os.path.join(config["data"]["processed_dir"], "test")

    evaluator = Evaluator(model, config, test_dir, device)
    evaluator.evaluate(patient_ids=patient_ids, csv_folder=output_dir)

    print("Evaluation completed.")

def load_trained_model(config, checkpoint_path):
    """
    Load the model from the checkpoint path. The model is initialized with the
    configuration parameters and the state_dict is loaded from the checkpoint.
    The model is set to evaluation mode.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing model parameters.
    checkpoint_path : str
        Path to the model checkpoint.

    Returns
    -------
    torch.nn.Module
        The loaded model.
    """
    # Set pretrained=False and load model
    config['model']['pretrained'] = False
    model = get_model(config)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_state_dict = checkpoint['model']
    model.load_state_dict(
        {k: v for k, v in model_state_dict.items() if "aux_classifier" not in k},
        strict=False
    )
    model.eval()
    return model

def load_fitted_model(config, weights_path):
    """
    Load the model from the fitted weights path. The model is initialized with the
    configuration parameters and the state_dict is loaded from the weights file.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing model parameters.
    weights_path : str
        Path to the model weights file.

    Returns
    -------
    torch.nn.Module
        The loaded model.
    """
    # Set pretrained=False and load model
    config['model']['pretrained'] = False
    model = get_model(config)

    model_state_dict = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(
        {k: v for k, v in model_state_dict.items() if "aux_classifier" not in k},
        strict=False
    )
    model.eval()
    return model