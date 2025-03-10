import time
import torch
import psutil

def profile_function(func, *args, **kwargs):
    """
    Profile the runtime and memory usage of a function.

    Parameters
    ----------
    func : callable
        Function to profile.
    *args
        Positional arguments to pass to the function.
    **kwargs
        Keyword arguments to pass to the function.

    Returns
    -------
    Any
        Result of the function.
    """
    process = psutil.Process()
    mem_before = process.memory_info().rss / 2**20
    start_time = time.time()

    result = func(*args, **kwargs)

    end_time = time.time()
    mem_after = process.memory_info().rss / 2**20

    print(f"Memory used: {mem_after - mem_before:.2f} MB")
    print(f"Runtime: {end_time - start_time:.2f} seconds")

    return result

