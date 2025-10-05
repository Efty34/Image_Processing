import numpy as np
import matplotlib.pyplot as plt

def manual_vertical_glcm_fn(image, levels=256):
    glcm = np.zeros((levels, levels), dtype=int)
    h, w = image.shape
    for y in range(h - 1): 
        for x in range(w):
            i = image[y, x]
            j = image[y + 1, x]
            glcm[i, j] += 1
    # glcm = glcm / glcm.sum()        
    return glcm

def manual_horizontal_glcm_fn(image, levels=256):
    glcm = np.zeros((levels, levels), dtype=int)
    h, w = image.shape
    for y in range(h): 
        for x in range(w - 1):
            i = image[y, x]
            j = image[y, x + 1]
            glcm[i, j] += 1
    # glcm = glcm / glcm.sum()        
    return glcm

def manual_diagonal_glcm_fn(image, levels=256):
    glcm = np.zeros((levels, levels), dtype=int)
    h, w = image.shape
    for y in range(h - 1): 
        for x in range(w - 1):
            i = image[y, x]
            j = image[y + 1, x + 1]
            glcm[i, j] += 1
    # glcm = glcm / glcm.sum()        
    return glcm

def plot_glcm(img, glcm1, glcm2, glcm3,title0, title1, title2, title3):
    plt.figure(figsize=(20, 16))

    ax0=plt.subplot(1, 4, 1)
    im0 = ax0.imshow(img, cmap='gray')
    ax0.set_title(title0)
    ax0.set_xlabel('X axis')
    ax0.set_ylabel('Y axis')

    ax1 = plt.subplot(1, 4, 2)
    im1 = ax1.imshow(glcm1, cmap='hot', interpolation='nearest')
    ax1.set_title(title1)
    ax1.set_xlabel('Gray Level j')
    ax1.set_ylabel('Gray Level i')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='Probability')

    ax2 = plt.subplot(1, 4, 3)
    im2 = ax2.imshow(glcm2, cmap='hot', interpolation='nearest')
    ax2.set_title(title2)
    ax2.set_xlabel('Gray Level j')
    ax2.set_ylabel('Gray Level i')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='Probability')

    ax3 = plt.subplot(1, 4, 4)
    im3 = ax3.imshow(glcm3, cmap='hot', interpolation='nearest')
    ax3.set_title(title3)
    ax3.set_xlabel('Gray Level j')
    ax3.set_ylabel('Gray Level i')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label='Probability')

    plt.tight_layout()
    plt.show()

def maxm_probability(glcm):
    glcm_n= glcm / glcm.sum()
    return np.max(glcm_n)

# def energy(glcm):
#     h, w = glcm.shape
#     total = 0.0
#     for y in range(h):
#         for x in range(w):
#             total += glcm[y, x] ** 2
#     return total
def energy(glcm):
    glcm_n= glcm / glcm.sum()
    return np.sum(glcm_n ** 2)

def entropy(glcm):
    h, w = glcm.shape
    glcm_n= glcm / glcm.sum()
    total = 0.0
    for y in range(h):
        for x in range(w):
            if glcm_n[y, x] > 0:
                total += glcm_n[y, x] * np.log2(glcm_n[y, x])
    return -total

def contrast(glcm):
    h,w=glcm.shape
    glcm_n= glcm / glcm.sum()
    total=0.0
    for y in range(h):
        for x in range(w):
            total=((y-x)**2)*glcm_n[y,x]
    return total

def homogenity(glcm):
    h,w=glcm.shape
    glcm_n= glcm / glcm.sum()
    total=0.0
    for y in range(h):
        for x in range(w):
            total=(glcm_n[y,x])/(1+np.abs(y-x))
    return total