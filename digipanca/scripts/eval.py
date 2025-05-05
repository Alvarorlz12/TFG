import os
import argparse
import torch

from src.utils.config import load_config
from src.utils.evaluation import evaluate_model, load_trained_model

def main():
    parser = argparse.ArgumentParser(description="Model evaluation on test set")

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
        help="Output directory for evaluation results"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    config_device = config['training']['device']
    device = torch.device(config_device if torch.cuda.is_available() else "cpu")

    DEFAULT_RESULTS_BASE_DIR = 'models/evaluation_results'
    if args.output_dir is None:
        model_name = os.path.basename(args.model_path).split('.')[0]
        args.output_dir = os.path.join(DEFAULT_RESULTS_BASE_DIR, model_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # Evaluate the model on the test set
    test_dir = os.path.join(config["data"]["processed_dir"], "test")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory {test_dir} does not exist.")
    
    model = load_trained_model(config, args.model_path).to(device)
    evaluate_model(model, config, args.output_dir, device, test_dir)

if __name__ == '__main__':
    main()
