import os

from src.data.split_data import create_train_test_split
from src.utils.config import load_config

def main():
    config = load_config()
    # config = load_config("configs/data/default_no_test.yaml")
    RAW_DATA_DIR = config["data"]["raw_dir"]
    TEST_SIZE = config["data"]["test_split"]
    VAL_SIZE = config["data"]["val_split"]
    SHUFFLE = config["data"]["shuffle"]

    sample_dirs = [
        os.path.join(RAW_DATA_DIR, d) for d in os.listdir(RAW_DATA_DIR)
    ]
    create_train_test_split(
        sample_dirs=sample_dirs,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        random_state=42,
        shuffle=SHUFFLE
    )

if __name__ == "__main__":
    main()