import numpy as np

def sharpening_kernel(size, sigma) -> np.ndarray:

    if size % 2 == 0:
        raise ValueError("Size must be an odd number.")
    
    center = size // 2
    
    x, y = np.meshgrid(np.arange(size) - center, np.arange(size) - center)
    
    u_squared_plus_v_squared = x**2 + y**2
    
    kernel = ((u_squared_plus_v_squared - 2 * sigma**2) / sigma**4) * np.exp(-u_squared_plus_v_squared / (2 * sigma**2))
    
    return kernel