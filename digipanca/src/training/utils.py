import random
import numpy as np
import torch

def set_seed(seed=42, deterministic=True):
    """
    Set random seed for reproducibility.
    
    Parameters
    ----------
    seed : int
        The seed value to set for random number generation.
    deterministic : bool
        If True, sets the random seed for deterministic behavior.
        If False, allows for non-deterministic behavior.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic