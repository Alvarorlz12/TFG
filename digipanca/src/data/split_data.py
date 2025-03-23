import os
import json
import random
from sklearn.model_selection import train_test_split

def create_train_test_split(
    sample_dirs,
    test_size=0.15,
    val_size=0.15,
    shuffle=True,
    random_state=42,
    split_dir="data/splits",
    filename="train_test_split.json"
):
    """
    Create a train-test split for the samples and a validation split from the 
    training set. Save the split to a JSON file.

    Parameters
    ----------
    sample_dirs : list
        List of directories containing the samples.
    test_size : float
        Proportion of the samples to include in the test split.
    val_size : float
        Proportion of the training samples to include in the validation split.
    random_state : int
        Random seed for reproducibility.
    split_dir : str
        Directory to save the split JSON file.
    filename : str
        Name of the split JSON file.

    Returns
    -------
    dict
        Dictionary containing the train-test split.
    """
    os.makedirs(split_dir, exist_ok=True)
    split_path = os.path.normpath(os.path.join(split_dir, filename))

    patient_ids = [os.path.basename(d) for d in sample_dirs]
    random.seed(random_state)

    if test_size != 0.0:
        train_ids, test_ids = train_test_split(
            patient_ids,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle
        )
    else:
        train_ids = patient_ids
        test_ids = []

    train_ids, val_ids = train_test_split(
        train_ids,
        test_size=val_size,
        random_state=random_state,
        shuffle=shuffle
    )

    split_dict = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids
    }

    with open(split_path, "w") as f:
        json.dump(split_dict, f, indent=4)

    print(f"✅ Train-test split saved to {split_path}")

    return split_dict

def load_train_test_split(split_path="data/splits/train_test_split.json"):
    """
    Load the train-test split from a JSON file.

    Parameters
    ----------
    split_path : str
        Path to the split JSON file.

    Returns
    -------
    dict
        Dictionary containing the train-test split.
    """
    with open(os.path.normpath(split_path), "r") as f:
        return json.load(f)