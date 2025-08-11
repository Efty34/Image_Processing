import numpy as np

def gaussian_smoothing_kernel(size, sigma) -> np.ndarray:

    if size % 2 == 0:
        raise ValueError("Size must be an odd number.")
    
    center = size // 2

    x, y = np.meshgrid(np.arange(size) - center, np.arange(size) - center)

    kernel = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (sigma**2))

    return kernel 