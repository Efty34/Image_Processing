import cv2
import numpy as np
import matplotlib.pyplot as plt

def notch_filter(img,notch_centers,radius,filter_type="reject"):
    h,w=img.shape
    u,v=np.meshgrid(np.arange(w),np.arange(h))
    notch_mask=np.ones((h, w),dtype=np.float32)
    center_u,center_v=h//2,w//2
    for center in notch_centers:
        u_notch, v_notch = center
        D1=np.sqrt((u-(center_v+u_notch-center_v))**2 + 
                     (v-(center_u+v_notch-center_u))**2)
        D2=np.sqrt((u-(center_v-(u_notch-center_v)))**2 + 
                     (v -(center_u-(v_notch -center_u)))**2)
        
        if filter_type=='reject':
            notch_mask=notch_mask*(D1>radius)*(D2>radius)
        else:
            print("404")
            
            
    return notch_mask


def apply(r):
    duck=cv2.imread("pnois2.jpg",0)
    ft = np.fft.fft2(duck)
    ft_shift=np.fft.fftshift(ft)
    
    notch_centers=[(261,261)]
    radius=r
    
    notch_mask=notch_filter(duck,notch_centers,radius,'reject')
    
    print(notch_mask)
    
    ft_filtered=ft_shift*notch_mask
    
    ft_ishift=np.fft.ifftshift(ft_filtered)
    img_filtered=np.fft.ifft2(ft_ishift)
    img_filtered=np.abs(img_filtered)
    
    phase=np.angle(ft_shift)
    phase_ =cv2.normalize(phase, None, 0, 255, cv2.NORM_MINMAX,dtype=cv2.CV_8U)
    
    fig,axes=plt.subplots(2,3,figsize=(15,10))
    
    axes[0,0].imshow(duck,cmap='gray')
    axes[0,0].set_title('Org image')
    axes[0,0].axis('off')
    
    magnitude_original=20*np.log(np.abs(ft_shift)+1)
    axes[0,1].imshow(magnitude_original, cmap='gray')
    axes[0,1].set_title('pectrum 2')
    axes[0,1].axis('off')
    
    axes[0,2].imshow(notch_mask,cmap='gray')
    axes[0,2].set_title('Notch Filter Mask')
    axes[0,2].axis('off')
    
    magnitude_filtered=20*np.log(np.abs(ft_filtered) + 1)
    axes[1,0].imshow(magnitude_filtered, cmap='gray')
    axes[1,0].set_title('spectrum 2')
    axes[1,0].axis('off')
    
    axes[1,1].imshow(img_filtered, cmap='gray')
    axes[1,1].set_title('Filtered Image')
    axes[1,1].axis('off')
    
    axes[1,2].imshow(phase_,cmap='gray')
    axes[1,2].set_title('phase')
    axes[1,2].axis('off')
    
    
    plt.tight_layout()
    plt.show()
    
    return img_filtered, notch_mask

filtered_duck,filter_mask=apply(r=5)


    
    
    
    