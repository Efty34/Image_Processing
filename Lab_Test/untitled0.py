import cv2
import numpy as np

def img_area(bin_img):
    area=np.count_nonzero(bin_img)
    return area

def img_perimeter(border_img):
    perimeter=np.count_nonzero(border_img)
    return perimeter

def find_ab(bin_img):
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) >= 5:
        (x, y), (MA, ma), angel = cv2.fitEllipse(cnt)
        aa=max(MA,ma)
        bb=min(MA,ma)
    else:
        print("404")

    return aa,bb

def clac_descriptors(bin_img,i):
    k=np.ones((3,3), dtype=np.uint8)
    eroded=cv2.erode(bin_img,k,iterations=1)
    bor_img=bin_img-eroded

    area=img_area(bin_img)
    peri=img_perimeter(bor_img)
    a,b=find_ab(bin_img)

    comp=(peri**2)/area
    form_fact=(4*np.pi*area)/(peri**2)
    eccentricity=np.sqrt(1-(b/a)**2)

    return comp,form_fact,eccentricity

def distp(t0,t1):
    abs_c=np.abs(t0[0]-t1[0])
    abs_ff=np.abs(t0[1]-t1[1])
    abs_ecc=np.abs(t0[2]-t1[2])

    return abs_c+abs_ff+abs_ecc

def sim_matrix(train_images,test_images):
    train_descriptors=[]
    test_descriptors=[]

    for i,img in enumerate(train_images):
        co,ff,rd=clac_descriptors(img,i)
        train_descriptors.append((co,ff,rd))

    for i, img in enumerate(test_images):
        co,ff,rd=clac_descriptors(img,i)
        test_descriptors.append((co,ff,rd))

    sim_mat=[]
    for i,test_d in enumerate(test_descriptors):
        sim_row=[]
        for j,train_d in enumerate(train_descriptors):
            dist=distp(test_d,train_d)
            sim_row.append(dist)
        sim_mat.append(sim_row)

    print("train",train_descriptors)
    print("test",test_descriptors)
    print("sim",sim_mat)

    print("Similarity Matrix\n")
    print("\t",end="")
    for j in range(len(train_images)):
        print(f"Train{j + 1}",end="\t")
    print()

    for i in range(len(test_images)):
        print(f"Test {i + 1}\t",end="")
        for j in range(len(train_images)):
            similarity_val=sim_mat[i][j]
            print(f"{similarity_val:.5f}",end="\t")
        print()

    return sim_mat

train_images=[
    cv2.imread('c1.jpg',0),
    #cv2.imread('p1.png',0),
    #cv2.imread('t1.jpg',0)
]
test_images=[
    #cv2.imread('c2.jpg',0),
    #cv2.imread('p2.png',0),
    #cv2.imread('t2.jpg',0),
    cv2.imread('st.jpg',0),
]

result=sim_matrix(train_images,test_images)
