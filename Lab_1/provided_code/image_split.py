import cv2
import numpy as np
import matplotlib.pyplot as plt

img_color = cv2.imread('Lab_1/assets/Lena.jpg',1)
img_gray = cv2.imread('Lab_1/assets/Lena.jpg',cv2.IMREAD_GRAYSCALE) # 2nd parameter 0--> gray image 1--> color image
print(img_color.shape)

cv2.imshow('Original image',img_color)
cv2.imshow('Gray_scale image',img_gray)
cv2.waitKey(0)

#%%
b1,g1,r1 = cv2.split(img_color)

# cv2.imshow("Green",g1)
# cv2.imshow("Red",r1)
# cv2.imshow("Blue",b1)
# cv2.waitKey(0)

b = img_color.copy()
# set green and red channels to 0
b[:, :, 1] = 0
b[:, :, 2] = 0

g = img_color.copy()
# set blue and red channels to 0
g[:, :, 0] = 0
g[:, :, 2] = 0

r = img_color.copy()
# set blue and green channels to 0
r[:, :, 0] = 0
r[:, :, 1] = 0
# RGB - Blue
cv2.imshow('B-RGB', b)
# RGB - Green
cv2.imshow('G-RGB', g)
# RGB - Red
cv2.imshow('R-RGB', r)
cv2.waitKey(0)

merged = cv2.merge((b1,g1,r1))
cv2.imshow("Merged",merged )
cv2.waitKey(0)

#%%
# Convert BGR to HSV
img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)

# Split HSV channels
h, s, v = cv2.split(img_hsv)

# Show each HSV channel as grayscale image
cv2.imshow('Value Channel', v)
cv2.imshow('Saturation Channel', s)

# Merge channels back
merged_hsv = cv2.merge([h, s, v])
cv2.imshow('Merged HSV', merged_hsv)

# Convert merged HSV back to BGR for display
img_merged_bgr = cv2.cvtColor(merged_hsv, cv2.COLOR_HSV2BGR)
cv2.imshow('Merged HSV Image', img_merged_bgr)
#%%

# Plot using matplotlib
# plt.figure(figsize=(10, 4))

# rgb_img = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
# plt.subplot(1, 2, 1)
# plt.imshow(rgb_img)
# plt.title('Original Image (RGB)')
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.imshow(img_gray, cmap='gray')
# plt.title('Grayscale Image')
# plt.axis('off')

# plt.tight_layout()
# plt.show()

#%%

cv2.waitKey(0)
cv2.destroyAllWindows()
