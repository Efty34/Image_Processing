import numpy as np
import cv2

def convolution2D(image, kernel):
    h, w = image.shape
    k = np.flip(kernel)
    kh, kw = k.shape
    
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("Kernel dimensions must be odd.")
    
    pad = (kh - 1) // 2
    img = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    result = np.zeros_like(img, dtype=np.float32)
    
    for i in range(pad, h + pad):
        for j in range(pad, w + pad):
            y = i - pad
            x = j - pad
            roi = img[y:y+kh, x:x+kw]
            val = np.sum(roi * k)
            result[i, j] = val
    
    raw_result_cropped = result[pad:h+pad, pad:w+pad]
    norm = np.round(cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)
    final = norm[pad:h+pad, pad:w+pad]
    
    return {
        "padding": pad,
        "border_img": img,
        "raw_result": result,
        "raw_result_cropped": raw_result_cropped,
        "normalized": norm,
        "final": final
    }

def convolve2D_2(image : np.array ,  kernel : np.array) -> np.array:
    kernel = np.flip(kernel)
    ih, iw = image.shape
    kh , kw = kernel.shape
    border_size = kh//2
    img_bordered = cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT)
    bh,bw = img_bordered.shape
    result = np.zeros((ih, iw), dtype=np.float32)
    for i in range(ih):
        for j in range(iw):
            region = img_bordered[i:i+kh ,j:j+kw]
            mul = np.multiply(region,kernel)
            val = np.sum(mul)
            result[i,j] = val  
    return result 


def gaussianFunction(x,y,sigma):
    return (1/(2*np.pi*sigma**2))*np.exp(-(x**2 + y**2) /(2*sigma**2))
    

def gaussian_kernel(size, sigma) -> np.ndarray:

    k = size // 2
    coords = np.arange(-k, k + 1)
    x,y = np.meshgrid(coords, coords)
    kernel = gaussianFunction(x, y, sigma)
    kernel /= np.sum(kernel)

    return kernel

def gaussian_X_derivative_kernel(size, sigma) -> np.ndarray:

    k = size // 2
    coords=np.arange(-k, k + 1)
    x,y = np.meshgrid(coords, coords)
    gaussValue=gaussianFunction(x, y, sigma)
    kernel = -(x / sigma**2) * gaussValue
    kernel /= np.sum(kernel)

    return kernel

def gaussian_Y_derivative_kernel(size, sigma) -> np.ndarray:

    k = size // 2
    coords=np.arange(-k, k + 1)
    x,y=np.meshgrid(coords, coords)
    gaussValue=gaussianFunction(x, y, sigma)
    kernel = -(y / sigma**2) * gaussValue
    kernel /= np.sum(kernel)

    return kernel

def logFunction(x,y,sigma):
    x2y2 = x**2 + y**2
    return -(1 / (np.pi * sigma**4)) * (1 - x2y2 / (2 * sigma**2)) * np.exp(-x2y2 / (2 * sigma**2))

def log_kernel(size, sigma) -> np.ndarray:

    k=size//2
    coords=np.arange(-k, k + 1)
    x,y=np.meshgrid(coords, coords)
    kernel = logFunction(x, y, sigma)

    return kernel

def int_kernel(kernel, min_val, max_val):

    k_min = np.min(kernel)
    k_max = np.max(kernel)
    norm_kernel = (kernel - k_min) / (k_max - k_min) 
    int_kernel = norm_kernel * (max_val - min_val) + min_val
    int_kernel = np.round(int_kernel).astype(int)
    return int_kernel

def double_threshold(image, low_thresh, high_thresh, mid_val):
    output = np.zeros_like(image, dtype=np.uint8)
    output[image>=high_thresh]=255
    output[image<low_thresh]=0
    mask =(image >=low_thresh)&(image<high_thresh)
    output[mask] = mid_val

    return output

def hysteresis_thresholding(image, low_thresh, high_thresh, strong_val, weak_val):

    strong_pixel_val = strong_val
    weak_pixel_val = weak_val

    output = np.zeros_like(image, dtype=np.uint8)

    strong_pixels = image >= high_thresh
    weak_pixels = (image < high_thresh) & (image >= low_thresh)

    output[strong_pixels] = strong_pixel_val
    output[weak_pixels] = weak_pixel_val

    pad_output = np.pad(output, pad_width=1, mode='constant', constant_values=0)

    rows, cols = image.shape
    for _ in range(5): 
        for r in range(rows):
            for c in range(cols):
                pr,pc=r+1,c+1
                if pad_output[pr, pc] == weak_pixel_val:
                    neighborhood = pad_output[pr-1:pr+2, pc-1:pc+2]
                    if np.max(neighborhood) == strong_pixel_val:
                        pad_output[pr, pc] = strong_pixel_val

    output = pad_output[1:-1, 1:-1]

    output[output == weak_pixel_val] = 0

    return output
    

def show_image(image, title='Image'):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()