import os
import logging

class Logger:
    """Logger class for logging"""

    LEVELS = {"DEBUG": logging.DEBUG, "INFO": logging.INFO,
              "WARNING": logging.WARNING, "ERROR": logging.ERROR}

    def __init__(self, log_dir = "experiments/logs", verbosity = "INFO",
                 console = False):
        """
        Initialize the logger and create log directory.

        Parameters
        ----------
        log_dir : str, optional
            Directory to save the logs.
        verbosity : str, optional
            Verbosity level of the logger
        console : bool, optional
            Whether to log to console.
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "training.log")

        # Convert verbosity to logging level
        verbosity_level = self.LEVELS.get(verbosity.upper(), logging.INFO)

        # Initialize logger
        self.logger = logging.getLogger("TrainingLogger")
        self.logger.setLevel(verbosity_level)

        # Check if logger already has handlers
        if not self.logger.hasHandlers():
            # File handler
            file_handler = logging.FileHandler(log_file, mode="a")
            file_handler.setLevel(verbosity_level)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )

            # Add handler to the logger
            self.logger.addHandler(file_handler)
            
            if console:
                # Console handler
                console_handler = logging.StreamHandler()
                console_handler.setLevel(verbosity_level)
                console_handler.setFormatter(
                    logging.Formatter("%(levelname)s - %(message)s")
                )
                self.logger.addHandler(console_handler)

            # Save handlers for closing
            if console:
                self.handlers = [file_handler, console_handler]
            else:
                self.handlers = [file_handler]

    def log(self, metrics, step = None):
        """
        Log the metrics.

        Parameters
        ----------
        metrics : dict
            Dictionary containing the metrics.
        step : int, optional
            Step number.
        """
        log_msg = f"Step {step}: " if step is not None else ""
        log_msg += ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(log_msg)

    def log_message(self, message, level = "INFO"):
        """
        Log a message.

        Parameters
        ----------
        message : str
            Message to log.
        level : str, optional
            Log level.
        """
        level = level.upper()
        if level in self.LEVELS:
            self.logger.log(self.LEVELS[level], message)

    def close(self):
        """Close the logger"""
        for handler in self.handlers:
            self.logger.removeHandler(handler)
            handler.close()