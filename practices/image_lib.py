import matplotlib.pyplot as plt
import numpy as np
import cv2

def sharp_image(image):
    sharpening_kernel = Gaussian_Sharpenning_kernel(size=5, sigma=1.0)
    b_channel ,g_channel ,r_channel = cv2.split(image)
    sharp_convolved_b = convolve(b_channel, sharpening_kernel)
    sharp_convolved_g = convolve(g_channel, sharpening_kernel)
    sharp_convolved_r = convolve(r_channel, sharpening_kernel)
    sharp_convolved_image = cv2.merge((sharp_convolved_b, sharp_convolved_g, sharp_convolved_r))
    normalized_result = cv2.normalize(sharp_convolved_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return normalized_result

def convolve(image : np.array ,  kernel : np.array) -> np.array:
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
            # rh , rw = region.shape 
            # sum = 0 
            # for k in range(rh):
            #     for l in range(rw):
            #         mul = region[k,l] * kernel[k,l]
            #         sum = sum + mul
            # result[i,j] = sum 
    return result 

def Gaussian_Smoothing_Function(u,v,sigma):
    return (1/(2*np.pi*sigma**2))*np.exp(-(u**2 + v**2) /(sigma**2))

def Gaussian_Sharp_Function(x,y,sigma):
    # return (-1/np.pi*sigma**4) * (1 - (u**2 + v**2)/(2*sigma**2)) * np.exp(-(u**2 + v**2)/(2*sigma**2))
    x2y2 = x**2 + y**2
    return -(1 / (np.pi * sigma**4)) * (1 - x2y2 / (2 * sigma**2)) * np.exp(-x2y2 / (2 * sigma**2))
def Gaussian_Smoothing_kernel(size , sigma):
    k = size // 2
    kernel = np.zeros((size,size ) , dtype=np.float32)
    for i in range(size):
        for j in range(size):
            u = i - k
            v = j - k 
            kernel[i,j] = Gaussian_Smoothing_Function(u,v,sigma)
    kernel /= np.sum(kernel)
    return kernel 

def Gaussian_Sharpenning_kernel(size , sigma):
    k = size // 2
    kernel = np.zeros((size,size ) , dtype=np.float32)
    for i in range(size):
        for j in range(size):
            u = i - k
            v = j - k 
            kernel[i,j] = Gaussian_Sharp_Function(u,v,sigma)
    kernel = kernel - kernel.mean()  # Normalize to zero mean
    return kernel 
def show_image(image, title='Image'):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()