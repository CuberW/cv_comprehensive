"""Demo processor for 语义分割."""
import numpy as np
import imageio.v3 as iio
from app.modules.phase1_fundamentals.grayscale.algorithm import to_uint8
from app.modules.phase4_deep_learning.semantic.algorithm import simple_segmentation_demo,colorize_segmentation


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
    if img_u8.ndim==2: img_u8=np.stack([img_u8]*3,axis=-1)
    nc=int(kwargs.get("num_classes",5))
    seg=simple_segmentation_demo(img_u8,num_classes=nc)
    seg_color=colorize_segmentation(seg,num_classes=nc)
    alpha=0.4; overlay=(img_u8*(1-alpha)+seg_color*alpha).astype(np.uint8)
    steps=[{"id":"original","name":"原图","image":img_u8,"explanation":"输入图像"},
           {"id":"segmentation","name":str(nc)+"类分割结果","image":overlay,"explanation":"半透明颜色叠加:不同颜色=不同语义类别"}]
    return {"steps":steps,"metrics":{"num_classes":nc,"unique_labels":len(np.unique(seg))}}
