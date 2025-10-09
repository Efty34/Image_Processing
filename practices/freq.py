duck=cv2.imread("./test_assets/duck.png",0)

plt.imshow(duck, cmap="gray")
plt.axis("off")
plt.show()

# Forward transform
ft = np.fft.fft2(duck)
ft_shift = np.fft.fftshift(ft)

# Power spectrum
magnitude_spectrum_ac = np.abs(ft_shift)
magnitude_spectrum = 20 * np.log( magnitude_spectrum_ac + 1 ) # +1 to remove 0, 20 = scaling factor
magnitude_spectrum = cv2.normalize(magnitude_spectrum, None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)

plt.imshow(magnitude_spectrum, cmap='gray')
plt.title("Magnitude Spectrum")
plt.show()

phase = np.angle(ft_shift)
phase_ = cv2.normalize(phase, None, 0, 255, cv2.NORM_MINMAX,dtype=cv2.CV_8U)

plt.imshow(phase_, cmap='gray')
plt.title("Phase Spectrum")
plt.show()

def notch_filter(img, notch_centers, radius, filter_type='reject'):
    """
    Create a notch filter to reject or pass specific frequencies
    
    Parameters:
    img: input image (2D array)
    notch_centers: list of tuples [(u1,v1), (u2,v2), ...] - notch center positions
    radius: radius of each notch
    filter_type: 'reject' (default) or 'pass'
    
    Returns:
    notch_mask: filter mask
    """
    h, w = img.shape
    
    # Create coordinate grids
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # Initialize filter as all ones (pass everything)
    notch_mask = np.ones((h, w), dtype=np.float32)
    
    # Center of frequency domain
    center_u, center_v = h // 2, w // 2
    
    for center in notch_centers:
        u_notch, v_notch = center
        
        # Create symmetric notch pairs
        # Positive notch at (u_notch, v_notch)
        D1 = np.sqrt((u - (center_v + u_notch - center_v))**2 + 
                     (v - (center_u + v_notch - center_u))**2)
        
        # Symmetric notch at (-u_notch, -v_notch)
        D2 = np.sqrt((u - (center_v - (u_notch - center_v)))**2 + 
                     (v - (center_u - (v_notch - center_u)))**2)
        
        if filter_type == 'reject':
            # Notch reject: set to 0 where distance <= radius
            notch_mask = notch_mask * (D1 > radius) * (D2 > radius)
        else:  # filter_type == 'pass'
            # Notch pass: set to 0 everywhere except where distance <= radius
            pass_mask = (D1 <= radius) | (D2 <= radius)
            notch_mask = notch_mask * pass_mask
    
    return notch_mask

# Example usage for your specific case:
def apply_notch_filter_to_duck():
    # Load duck image
    duck = cv2.imread("./test_assets/duck.png", 0)
    
    # Forward transform
    ft = np.fft.fft2(duck)
    ft_shift = np.fft.fftshift(ft)
    
    # Define notch centers and radius
    notch_centers = [(272, 256), (262, 261)]  # Your specific coordinates
    radius = 5
    
    # Create notch reject filter
    notch_mask = notch_filter(duck, notch_centers, radius, 'reject')
    
    # Apply filter
    ft_filtered = ft_shift * notch_mask
    
    # Inverse transform
    ft_ishift = np.fft.ifftshift(ft_filtered)
    img_filtered = np.fft.ifft2(ft_ishift)
    img_filtered = np.abs(img_filtered)
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(duck, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Original spectrum
    magnitude_original = 20 * np.log(np.abs(ft_shift) + 1)
    axes[0, 1].imshow(magnitude_original, cmap='gray')
    axes[0, 1].set_title('Original Spectrum')
    axes[0, 1].axis('off')
    
    # Notch filter mask
    axes[0, 2].imshow(notch_mask, cmap='gray')
    axes[0, 2].set_title('Notch Filter Mask')
    axes[0, 2].axis('off')
    
    # Filtered spectrum
    magnitude_filtered = 20 * np.log(np.abs(ft_filtered) + 1)
    axes[1, 0].imshow(magnitude_filtered, cmap='gray')
    axes[1, 0].set_title('Filtered Spectrum')
    axes[1, 0].axis('off')
    
    # Filtered image
    axes[1, 1].imshow(img_filtered, cmap='gray')
    axes[1, 1].set_title('Filtered Image')
    axes[1, 1].axis('off')
    
    # Difference
    axes[1, 2].imshow(np.abs(duck.astype(float) - img_filtered), cmap='gray')
    axes[1, 2].set_title('Difference')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return img_filtered, notch_mask

# Run the example
filtered_duck, filter_mask = apply_notch_filter_to_duck()

def butterworth_notch_filter(img, notch_centers, D0, n=2, filter_type='reject'):
    """
    Create a Butterworth notch filter to reject or pass specific frequencies
    
    Parameters:
    img: input image (2D array)
    notch_centers: list of tuples [(uk,vk), ...] - notch center positions
    D0: cutoff frequency (radius) for each notch
    n: order of the Butterworth filter (default=2)
    filter_type: 'reject' (default) or 'pass'
    
    Returns:
    notch_mask: Butterworth notch filter mask
    """
    h, w = img.shape
    
    # Create coordinate grids
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # Center of frequency domain
    center_u, center_v = h // 2, w // 2
    
    # Initialize filter as all ones
    H_NR = np.ones((h, w), dtype=np.float32)
    
    for center in notch_centers:
        uk, vk = center
        
        # Distance from positive notch center (uk, vk)
        Dk = np.sqrt((u - (center_v + uk - center_v))**2 + 
                     (v - (center_u + vk - center_u))**2)
        
        # Distance from negative notch center (-uk, -vk)
        D_k = np.sqrt((u - (center_v - (uk - center_v)))**2 + 
                      (v - (center_u - (vk - center_u)))**2)
        
        # Butterworth notch reject filter formula
        # H_NR(u,v) = [1/(1+(D0/Dk)^2n)] * [1/(1+(D0/D_k)^2n)]
        
        # Avoid division by zero
        Dk = np.where(Dk == 0, 1e-10, Dk)
        D_k = np.where(D_k == 0, 1e-10, D_k)
        
        if filter_type == 'reject':
            # Butterworth notch reject
            H1 = 1 / (1 + (D0 / Dk)**(2*n))
            H2 = 1 / (1 + (D0 / D_k)**(2*n))
            H_NR = H_NR * H1 * H2
        else:  # filter_type == 'pass'
            # Butterworth notch pass (complement of reject)
            H1 = 1 / (1 + (Dk / D0)**(2*n))
            H2 = 1 / (1 + (D_k / D0)**(2*n))
            H_NR = H_NR * H1 * H2
    
    return H_NR

def apply_butterworth_notch_filter():
    """Apply Butterworth notch filter to duck image"""
    # Load duck image
    duck = cv2.imread("./test_assets/duck.png", 0)
    
    # Forward transform
    ft = np.fft.fft2(duck)
    ft_shift = np.fft.fftshift(ft)
    
    # Define notch centers and parameters
    notch_centers = [(272, 256), (262, 261)]  # Your specific coordinates
    D0 = 10  # Cutoff frequency
    n = 2    # Butterworth order
    
    # Create Butterworth notch reject filter
    butterworth_mask = butterworth_notch_filter(duck, notch_centers, D0, n, 'reject')
    
    # Apply filter
    ft_filtered = ft_shift * butterworth_mask
    
    # Inverse transform
    ft_ishift = np.fft.ifftshift(ft_filtered)
    img_filtered = np.fft.ifft2(ft_ishift)
    img_filtered = np.abs(img_filtered)
    
    # Visualization
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Original image
    axes[0, 0].imshow(duck, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Original spectrum
    magnitude_original = 20 * np.log(np.abs(ft_shift) + 1)
    axes[0, 1].imshow(magnitude_original, cmap='gray')
    axes[0, 1].set_title('Original Spectrum')
    axes[0, 1].axis('off')
    
    # Butterworth notch filter mask
    axes[0, 2].imshow(butterworth_mask, cmap='gray')
    axes[0, 2].set_title('Butterworth Notch Filter')
    axes[0, 2].axis('off')
    
    # Filtered spectrum
    magnitude_filtered = 20 * np.log(np.abs(ft_filtered) + 1)
    axes[0, 3].imshow(magnitude_filtered, cmap='gray')
    axes[0, 3].set_title('Filtered Spectrum')
    axes[0, 3].axis('off')
    
    # Filtered image
    axes[1, 0].imshow(img_filtered, cmap='gray')
    axes[1, 0].set_title('Butterworth Filtered Image')
    axes[1, 0].axis('off')
    
    # Difference
    axes[1, 1].imshow(np.abs(duck.astype(float) - img_filtered), cmap='gray')
    axes[1, 1].set_title('Difference')
    axes[1, 1].axis('off')
    
    # Compare with ideal notch filter
    ideal_mask = notch_filter(duck, notch_centers, 5, 'reject')
    ft_ideal = ft_shift * ideal_mask
    ft_ideal_ishift = np.fft.ifftshift(ft_ideal)
    img_ideal = np.abs(np.fft.ifft2(ft_ideal_ishift))
    
    axes[1, 2].imshow(img_ideal, cmap='gray')
    axes[1, 2].set_title('Ideal Notch Filtered')
    axes[1, 2].axis('off')
    
    # Filter comparison
    axes[1, 3].imshow(ideal_mask, cmap='gray')
    axes[1, 3].set_title('Ideal Notch Filter')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return img_filtered, butterworth_mask

# Test different Butterworth orders
def compare_butterworth_orders():
    """Compare different orders of Butterworth notch filter"""
    duck = cv2.imread("./test_assets/duck.png", 0)
    ft = np.fft.fft2(duck)
    ft_shift = np.fft.fftshift(ft)
    
    notch_centers = [(272, 256), (262, 261)]
    D0 = 10
    orders = [1, 2, 4, 8]
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    for i, n in enumerate(orders):
        # Create filter
        mask = butterworth_notch_filter(duck, notch_centers, D0, n, 'reject')
        
        # Apply filter
        ft_filtered = ft_shift * mask
        ft_ishift = np.fft.ifftshift(ft_filtered)
        img_filtered = np.abs(np.fft.ifft2(ft_ishift))
        
        # Display filter
        axes[0, i].imshow(mask, cmap='gray')
        axes[0, i].set_title(f'Butterworth n={n}')
        axes[0, i].axis('off')
        
        # Display filtered image
        axes[1, i].imshow(img_filtered, cmap='gray')
        axes[1, i].set_title(f'Filtered (n={n})')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Run the examples
filtered_duck_butterworth, butterworth_filter_mask = apply_butterworth_notch_filter()
compare_butterworth_orders()