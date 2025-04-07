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

def worker_init_fn(worker_id):
    """
    Worker initialization function for DataLoader.
    
    Parameters
    ----------
    worker_id : int
        The ID of the worker process.
    """
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)