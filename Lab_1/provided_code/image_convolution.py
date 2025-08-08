import numpy as np
import cv2

# Load grayscale image
img = cv2.imread('Lab_1/assets/box.jpg', cv2.IMREAD_GRAYSCALE)
h, w = img.shape
# Add border to preserve edge during convolution
img_bordered = cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_CONSTANT)

# Gaussian kernel
kernel1 = np.array([
    [1,  4,  6,  4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1,  4,  6,  4, 1]
], dtype=np.float32)

kernel1 = np.flip(kernel1)  # Flip the kernel for convolution

'''
kernel2 = np.array([[-1, 0, 1],
                    [-1, 0, 1],
                    [-1, 0, 1]])

kernel3 = np.array([[-1, -1, -1],
                    [0, 0, 0],
                    [1, 1, 1]])

'''
# Use float32 depth to keep values beyond 0–255
img_conv = cv2.filter2D(img_bordered, ddepth=cv2.CV_32F, kernel= kernel1)

print(img_conv.shape)

# Normalize the float result to 0–255 and convert to uint8
norm = np.round(cv2.normalize(img_conv, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)
norm_cropped = norm [2:h+2, 2:w+2]

# Show all images
cv2.imshow('Original Grayscale Image', img)
cv2.imshow('Bordered Image', img_bordered)
cv2.imshow('Convolution Image', img_conv)
cv2.imshow('Normalized Image', norm)
cv2.imshow('Normalized Cropped Image', norm_cropped)

cv2.waitKey(0)
cv2.destroyAllWindows()
