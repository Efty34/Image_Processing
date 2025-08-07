import numpy as np
import cv2

img1 = np.array([
                [1,1,0,1,0],
                [0,0,1,0,0],
                [0,1,0,0,1],
                [1,0,1,0,1],
                [0,1,1,1,0]], dtype="uint8")
print(img1)

img2 = np.array([
    [   12,  26,  33,  26,  12],
    [   26,  55,  71,  55,  26],
    [   33,  71,  91,  71,  33],
    [   26,  55,  71,  55,  26],
    [   12,  26,  33,  26,  12],
   
], dtype=np.float32)

img3 = cv2.resize(img2, None, fx = 100, fy = 100, interpolation = cv2.INTER_NEAREST)
img3 = img3*255
norm = np.round(cv2.normalize(img3, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)
# norm(i, j) = ((input(i, j) - min) / (max - min)) * 255

cv2.imshow('Interpolation image',norm)
cv2.waitKey(0)
cv2.destroyAllWindows()