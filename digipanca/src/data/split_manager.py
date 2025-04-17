import json

class SplitManager:
    def __init__(self, split_data):
        """
        Initialize the SplitManager with the path to the split file.

        Parameters
        ----------
        split_data : str
            Path to the JSON file containing the train-test split.
        """
        self.split_data = split_data
        if isinstance(split_data, str):
            with open(split_data, 'r') as f:
                self.splits = json.load(f)
        else:
            self.splits = split_data

        # If it is not a list, convert it to a list
        if not isinstance(self.splits, list):
            self.splits = [self.splits]

    def __len__(self):
        """
        Get the number of splits.

        Returns
        -------
        int
            Number of splits.
        """
        return len(self.splits)
    
    def __getitem__(self, idx):
        """
        Get the split at the specified index.

        Parameters
        ----------
        idx : int
            Index of the split.

        Returns
        -------
        dict
            Dictionary containing the train-test split.
        """
        return self.splits[idx]
    
    def __iter__(self):
        """
        Iterate over the splits.

        Yields
        ------
        dict
            Dictionary containing the train-test split.
        """
        for split in self.splits:
            yield split
