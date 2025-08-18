import numpy as np

def sharpening_kernel(size, sigma) -> np.ndarray:

    if size % 2 == 0:
        raise ValueError("Size must be an odd number.")
    
    center = (size-1) // 2
    
    x, y = np.meshgrid(np.arange(size) - center, np.arange(size) - center)

    x2y2 = x**2 + y**2
    
    kernel = ((x2y2 - (2 * sigma**2)) / sigma**4) * (np.exp(-x2y2 / (2 * sigma**2)))

    # kernel = kernel / np.sum(kernel) 
    
    return kernel