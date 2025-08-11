import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale images
box = cv2.imread('Lab_1/assets/box.jpg', cv2.IMREAD_GRAYSCALE)

# getting h, w dimensions of the images
h, w = box.shape

# Task - 1 : Working with Kernel 1

# Gaussian kernel
kernel1 = np.array([
    [1,  4,  6,  4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1,  4,  6,  4, 1]
], dtype=np.float32)

kh, kw = kernel1.shape

# flipping the kernel for pure convolution, if not flipped then that will be cross-correlation.
kernel1 = np.flip(kernel1)

# we want same output as input, so we need to pad the input image, this formula only works for symmetric kernels
padding = (kernel1.shape[0] - 1 ) // 2

# Add borders to the images. For the purpose of padding
box_bordered = cv2.copyMakeBorder(box, padding, padding, padding, padding, cv2.BORDER_CONSTANT)

# output place holder for the convolution
box_conv = np.zeros_like(box_bordered, dtype=np.float32)

# Use float32 depth to keep values beyond 0–255
# img_conv = cv2.filter2D(lena_bordered, ddepth=cv2.CV_32F, kernel= kernel1)

# Manual convolution operation using kernel1
for i in range(padding, h + padding):
    for j in range(padding, w + padding):
        result = 0
        for m in range(kh):
            for n in range(kw):
                result += kernel1[m, n] * box_bordered[i - padding + m, j - padding + n]
        box_conv[i, j] = result

# Normalize the float result to 0–255 and convert to uint8
box_norm = np.round(cv2.normalize(box_conv, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)
box_norm_cropped = box_norm[2:h+2, 2:w+2]

# Show all images
cv2.imshow('Original Grayscale Image', box)
cv2.imshow('Bordered Image', box_bordered)
cv2.imshow('Convolution Image', box_conv)
cv2.imshow('Normalized Image', box_norm)
cv2.imshow('Normalized Cropped Image', box_norm_cropped)




# Taks - 2 : Working with Kernel 2

# In kernel 2, the center is at (0, 0) or top left corner
kernel2 = np.array([[-1, 0, 1],
                    [-1, 0, 1],
                    [-1, 0, 1]])

# flipping the kernel for convolution
# After flipping the center is at bottom right
kernel2 = np.flip(kernel2)

kh2, kw2 = kernel2.shape

# Padding for kernel2, no formula can be applied, intution only
box_bordered_2 = cv2.copyMakeBorder(box, 2, 0, 2, 0, cv2.BORDER_CONSTANT)

box_conv_2 = np.zeros_like(box_bordered_2, dtype=np.float32)



for i in range(2, h+2):
    for j in range(2, w+2):
        result = 0
        for m in range(kh2):
            for n in range(kw2):
                result += kernel2[m, n] * box_bordered_2[i+m-2,j+n-2]
        box_conv_2[i , j] = result

print("Convolution with Kernel 2 completed.", box_conv_2)

# Normalize and display results for kernel2
box_norm_2 = np.round(cv2.normalize(box_conv_2, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)
box_norm_cropped_2 = box_norm_2[2:h+2, 2:w+2]

# Show kernel2 results
cv2.imshow('Kernel2 - Original Grayscale Image', box)
cv2.imshow('Kernel2 - Bordered Image', box_bordered_2)
cv2.imshow('Kernel2 - Convolution Result', box_conv_2)
cv2.imshow('Kernel2 - Normalized Result', box_norm_2)
cv2.imshow('Kernel2 - Normalized Cropped', box_norm_cropped_2)



cv2.waitKey(0)
cv2.destroyAllWindows()