"""Demo processor for 频域分析."""
import numpy as np
import imageio.v3 as iio
from app.modules.phase1_fundamentals.grayscale.algorithm import to_uint8
from app.modules.phase3_intermediate.frequency.algorithm import compute_fft_spectrum,apply_lowpass_filter,apply_highpass_filter


def _to_uint8_heat(arr):
    """Float array -> uint8 heatmap."""
    arr=np.asarray(arr,dtype=np.float64)
    m=arr.max() if arr.size else 1.0
    if m<=1e-12: return np.zeros(arr.shape,dtype=np.uint8)
    return np.clip(arr/m*255,0,255).astype(np.uint8)


def build_pipeline(image_path=None, **kwargs):
    img_u8 = to_uint8(iio.imread(image_path)) if image_path else np.zeros((64,64,3), dtype=np.uint8)
    
    img=iio.imread(image_path) if image_path else np.zeros((64,64,3),dtype=np.uint8)
    img_u8=to_uint8(img)
    if img_u8.ndim==3: gray=np.round(img_u8[:,:,0]*0.299+img_u8[:,:,1]*0.587+img_u8[:,:,2]*0.114).astype(np.uint8)
    else: gray=img_u8
    fft_s,mag,log_mag=compute_fft_spectrum(gray.astype(np.float64))
    lowpass=apply_lowpass_filter(gray.astype(np.float64),cutoff_ratio=0.05)
    highpass=apply_highpass_filter(gray.astype(np.float64),cutoff_ratio=0.01)
    mag_vis=np.clip(log_mag/log_mag.max()*255,0,255).astype(np.uint8)
    low_vis=np.clip(lowpass/lowpass.max()*255,0,255).astype(np.uint8) if lowpass.max()>0 else np.zeros_like(gray,dtype=np.uint8)
    high_vis=np.clip(np.abs(highpass)/np.abs(highpass).max()*255,0,255).astype(np.uint8) if np.abs(highpass).max()>0 else np.zeros_like(gray,dtype=np.uint8)
    steps=[{"id":"original","name":"原图","image":gray,"explanation":"输入灰度图"},
           {"id":"magnitude","name":"幅值谱","image":mag_vis,"explanation":"中心=低频,边缘=高频.对数尺度压缩动态范围"},
           {"id":"lowpass","name":"低通滤波","image":low_vis,"explanation":"只保留低频=图像模糊,边缘和细节消失"},
           {"id":"highpass","name":"高通滤波","image":high_vis,"explanation":"只保留高频=只显示边缘和纹理"}]
    return {"steps":steps,"metrics":{}}
