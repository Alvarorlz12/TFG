import os
import yaml

_PROJECT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..')
)

def load_config(config_path="configs/data/default.yaml"):
    """
    Load the configuration file from the given path. If the path is not given,
    the default path is used.

    Parameters
    ----------
    config_path : str, optional
        Path to the configuration file (default: "configs/data/default.yaml").

    Returns
    -------
    dict
        Configuration dictionary.
    """
    config_path = os.path.join(_PROJECT_DIR, config_path)
    
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Make the root directory an absolute path
    config["data"]["root_dir"] = os.path.join(_PROJECT_DIR,
                                              config["data"]["root_dir"])

    # Make all the data paths absolute
    for key, value in config["data"].items():
        if isinstance(value, str):  # Only strings are paths
            config["data"][key] = os.path.join(config["data"]["root_dir"], 
                                               value)

    return config