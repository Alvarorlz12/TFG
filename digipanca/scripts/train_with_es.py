import argparse

from ignite.engine import Events
from monai.bundle import ConfigParser
from monai.handlers import EarlyStopHandler

def main():
    parser = argparse.ArgumentParser(description="Training with Early Stopping")

    parser.add_argument(
        '--config_file', type=str, required=True,
        help="Path to the config file (MONAI bundle format)"
    )
    args = parser.parse_args()

    config = ConfigParser()
    config.read_config(args.config_file)

    trainer = config.get_parsed_content("train#trainer")
    evaluator = config.get_parsed_content("validate#evaluator")
    patience = config.get("train#patience", None)
    if patience is None:
        patience = 10
    
    es_handler = EarlyStopHandler(
        patience=patience,
        score_function=lambda eng: eng.state.metrics['val_mean_dice'],
        epoch_level=True,
        min_delta=0.0
    )

    es_handler.set_trainer(trainer)

    evaluator.add_event_handler(Events.COMPLETED, es_handler)

    trainer.run()

if __name__ == '__main__':
    main()
