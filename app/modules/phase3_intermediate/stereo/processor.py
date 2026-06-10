"""Demo processor for 立体匹配."""
import numpy as np
import imageio.v3 as iio
from app.modules.phase1_fundamentals.grayscale.algorithm import to_uint8
from app.modules.phase3_intermediate.stereo.algorithm import block_matching_disparity,disparity_to_color


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
    # Simulate right image by shifting left slightly (creates synthetic disparity)
    h,w=gray.shape; right_img=np.roll(gray,int(w*0.02),axis=1)
    right_img[:,:int(w*0.02)]=right_img[:,int(w*0.02):int(w*0.02)*2]
    disp=block_matching_disparity(gray,right_img,block_size=7,max_disparity=32)
    disp_color=disparity_to_color(disp)
    steps=[{"id":"left","name":"左视图","image":gray,"explanation":"左相机图像"},
           {"id":"right","name":"右视图","image":right_img,"explanation":"右相机图像(模拟视差)"},
           {"id":"disparity","name":"视差图","image":disp_color,"explanation":"红=近(大视差),蓝=远(小视差)"}]
    return {"steps":steps,"metrics":{"max_disparity":float(disp.max()),"mean_disparity":round(float(disp[disp>0].mean()),2)}}
