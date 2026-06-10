"""Demo processor for HOG 特征."""
import numpy as np
import imageio.v3 as iio
from app.modules.phase1_fundamentals.grayscale.algorithm import to_uint8
from app.modules.phase3_intermediate.hog_svm.algorithm import extract_hog_features,visualize_hog_cells


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
    features,cell_hists,(mag,ang)=extract_hog_features(gray.astype(np.float64),cell_size=8,block_size=2,num_bins=9)
    hog_vis=visualize_hog_cells(gray,cell_hists)
    mag_vis=np.clip(mag/mag.max()*255,0,255).astype(np.uint8) if mag.max()>0 else np.zeros_like(gray,dtype=np.uint8)
    steps=[{"id":"original","name":"原图","image":img_u8,"explanation":"输入图像"},
           {"id":"gradient","name":"梯度幅值","image":mag_vis,"explanation":"每个像素的边缘强度"},
           {"id":"hog","name":"HOG 特征可视化","image":hog_vis,"explanation":"每个8×8 cell:线段=梯度主方向,长度=该方向强度。共"+str(len(features))+"维特征"}]
    return {"steps":steps,"metrics":{"feature_dim":int(len(features)),"cell_size":8,"num_cells":f"{cell_hists.shape[0]}x{cell_hists.shape[1]}"}}
