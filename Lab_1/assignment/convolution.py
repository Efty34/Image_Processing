import numpy as np
import cv2

def apply_convolution(image, kernel):
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
    
    norm = np.round(cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)
    final = norm[pad:h+pad, pad:w+pad]
    
    return {
        "padding": pad,
        "border_img": img,
        "raw_result": result,
        "normalized": norm,
        "final": final
    }