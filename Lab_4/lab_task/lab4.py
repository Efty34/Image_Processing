# %%

import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi

# %%
def img_area(bin_img):
    area=np.count_nonzero(bin_img)
    return area
# %%
def img_perimeter(border_img):
    perimeter=np.count_nonzero(border_img)
    return perimeter
# %%
def find_max_d(bin_img):
    min_x=min_y=1000000
    max_x=max_y=0
    h,w=bin_img.shape
    for x in range(h):
        for y in range(w):
            if(bin_img[x,y]<=0):
                continue
            min_x=min(min_x,x)
            min_y=min(min_y,y)
            max_x=max(max_x,x)
            max_y=max(max_y,y)
    
    return (max_x-min_x,max_y-min_y)

# %%
def calc_descriptors(binary_img,i):

    se=np.ones((3,3),np.uint8)
    eroded=cv2.erode(binary_img,se,iterations=1)
    border_img=binary_img-eroded

    # cv2.imshow(f'border_{i}',border_img)
    # cv2.imshow(f'binary_{i}',binary_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    area=img_area(binary_img)
    perimeter=img_perimeter(border_img)
    
    max_d_tup=find_max_d(binary_img)
    max_d=max(max_d_tup)

    compactness=(perimeter**2)/area
    form_factor=(4*pi*area)/(perimeter**2)
    roundness=(4*area)/(pi*max_d**2)

    return form_factor,roundness,compactness

# %%
def euc_distt(t1, t2):
    abs_f=(t1[0]-t2[0])**2
    abs_r=(t1[1]-t2[1])**2
    abs_c=(t1[2]-t2[2])**2
    
    return sqrt(abs_f+abs_r+abs_c)

# %%
def sim_matrix(train_images,test_images):
    train_descriptors=[]
    test_descriptors=[]

    for i,img in enumerate(train_images):
        ff,rs,cs=calc_descriptors(img,i)
        train_descriptors.append((ff,rs,cs))
    
    for i, img in enumerate(test_images):
        ff,rs,cs=calc_descriptors(img,i)
        test_descriptors.append((ff,rs,cs))
    
    sim_mat=[]
    for i,test_d in enumerate(test_descriptors):
        sim_row=[]
        for j,train_d in enumerate(train_descriptors):
            euc_dist=euc_distt(test_d,train_d)
            sim_row.append(euc_dist)
        sim_mat.append(sim_row)
    
    print("train",train_descriptors)
    print("test",test_descriptors)
    print("sim",sim_mat)
    
    print("Similarity Matrix\n")
    print("\t",end="")
    for j in range(len(train_images)):
        print(f"GT{j + 1}",end="\t")
    print()

    for i in range(len(test_images)):
        print(f"Test {i + 1}\t",end="")
        for j in range(len(train_images)):
            similarity_val=sim_mat[i][j]
            print(f"{similarity_val:.5f}",end="\t")
        print()
    
    return sim_matrix

# %%
train_images=[
    cv2.imread('../assets/c1.jpg',0),
    cv2.imread('../assets/t1.jpg',0),
    cv2.imread('../assets/p1.png',0)
]
test_images=[
    cv2.imread('../assets/c2.jpg',0),
    cv2.imread('../assets/t2.jpg',0),
    cv2.imread('../assets/p2.png',0),
    cv2.imread('../assets/st.jpg',0),
]
result=sim_matrix(train_images,test_images)
