import numpy as np
from imageio.v3 import imread

def gaussian_kernel(size=5, sigma=1.0):
    ax = np.linspace(-(size//2), size//2, size)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2+yy**2)/(2*sigma**2))
    return k/k.sum()

def sobel_x(): return np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
def sobel_y(): return np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

def median_filter(img, size=3):
    from scipy.ndimage import median_filter as mf
    return mf(img, size=size)

def bilateral_filter(img, d=9, sigma_color=75, sigma_space=75):
    h, w = img.shape[:2]; out = np.zeros_like(img, dtype=np.float64)
    img_f = img.astype(np.float64)
    half = d//2
    ax = np.linspace(-half, half, d)
    xx, yy = np.meshgrid(ax, ax)
    spatial_k = np.exp(-(xx**2+yy**2)/(2*sigma_space**2))
    for ch in range(3) if img.ndim==3 else [None]:
        src = img_f[:,:,ch] if img.ndim==3 else img_f
        for i in range(half, h-half):
            for j in range(half, w-half):
                patch = src[i-half:i+half+1, j-half:j+half+1]
                range_k = np.exp(-((patch-src[i,j])**2)/(2*sigma_color**2))
                weight = spatial_k*range_k
                out[i,j,ch if img.ndim==3 else 0] = np.sum(patch*weight)/weight.sum() if img.ndim==3 else np.sum(patch*weight)/weight.sum()
    return np.clip(out, 0, 255).astype(np.uint8)

def build_conv_demo(upload_path):
    img = imread(upload_path)
    if img.ndim==3 and img.shape[2]==4: img=img[:,:,:3]
    results = []
    # Identity
    results.append(('identity','单位冲激核',img))
    # Box blur
    k = np.ones((5,5))/25
    if img.ndim==3:
        blurred=np.zeros_like(img);[exec(f'blurred[:,:,{c}]=np.convolve(img[:,:,{c}].flatten(),k.flatten(),"same").reshape(img.shape[:2])') for c in range(3)]
    else: blurred=np.convolve(img.flatten(),k.flatten(),'same').reshape(img.shape)
    results.append(('box_blur','方框模糊(5×5)',np.clip(blurred,0,255).astype(np.uint8)))
    # Gaussian blur
    gk = gaussian_kernel(5, 1.5)
    if img.ndim==3:
        gb=np.zeros_like(img);[exec(f'gb[:,:,{c}]=np.convolve(img[:,:,{c}].flatten(),gk.flatten(),"same").reshape(img.shape[:2])') for c in range(3)]
    else: gb=np.convolve(img.flatten(),gk.flatten(),'same').reshape(img.shape)
    results.append(('gaussian','高斯模糊(σ=1.5)',np.clip(gb,0,255).astype(np.uint8)))
    # Sobel
    gray=img if img.ndim==2 else (0.299*img[:,:,0]+0.587*img[:,:,1]+0.114*img[:,:,2])
    sx=np.convolve(gray.flatten(),sobel_x().flatten(),'same').reshape(gray.shape)
    sy=np.convolve(gray.flatten(),sobel_y().flatten(),'same').reshape(gray.shape)
    mag=np.clip(np.sqrt(sx**2+sy**2),0,255).astype(np.uint8)
    results.append(('sobel','Sobel梯度幅值',mag))
    # Sharpen
    sharpen_k=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    if img.ndim==3:
        sh=np.zeros_like(img);[exec(f'sh[:,:,{c}]=np.clip(np.convolve(img[:,:,{c}].flatten(),sharpen_k.flatten(),"same").reshape(img.shape[:2]),0,255)') for c in range(3)]
    else: sh=np.clip(np.convolve(img.flatten(),sharpen_k.flatten(),'same').reshape(img.shape),0,255)
    results.append(('sharpen','锐化',sh.astype(np.uint8)))
    return results
