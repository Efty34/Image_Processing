import numpy as np
import cv2

def apply_convolution(image: np.ndarray, kernel: np.ndarray):

    h, w = image.shape
    
    kernel_flip = np.flip(kernel)

    kh, kw = kernel_flip.shape

    padding = (kh - 1) // 2

    border_image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT)

    convo_op = np.zeros_like(border_image, dtype=np.float32)

    for i in range(padding, h + padding):
        for j in range(padding, w + padding):
            result = 0
            for m in range(kh):
                for n in range(kw):
                    result += border_image[i + m - padding, j + n - padding] * kernel_flip[m, n]
            convo_op[i, j] = result
    
    norm = np.round(cv2.normalize(convo_op, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)
    norm_cropped = norm[padding:h + padding, padding:w + padding]

    return {
        "padding": padding,
        "border_image": border_image,
        "convo_op": convo_op,
        "norm": norm,
        "norm_cropped": norm_cropped
    }