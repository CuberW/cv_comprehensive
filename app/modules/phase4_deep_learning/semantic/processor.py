"""Demo processor for 语义分割."""
import numpy as np
from app.utils.image_utils import load_image_u8
from app.modules.phase4_deep_learning.semantic.algorithm import simple_segmentation_demo,colorize_segmentation


def _to_uint8_heat(arr):
    """Float array -> uint8 heatmap."""
    arr=np.asarray(arr,dtype=np.float64)
    m=arr.max() if arr.size else 1.0
    if m<=1e-12: return np.zeros(arr.shape,dtype=np.uint8)
    return np.clip(arr/m*255,0,255).astype(np.uint8)


def build_pipeline(image_path=None, **kwargs):
    img_u8 = load_image_u8(image_path, mode='rgb', max_side=512) if image_path else np.zeros((64,64,3), dtype=np.uint8)
    nc=int(kwargs.get("num_classes",5))
    seg=simple_segmentation_demo(img_u8,num_classes=nc)
    seg_color=colorize_segmentation(seg,num_classes=nc)
    alpha=0.4; overlay=(img_u8*(1-alpha)+seg_color*alpha).astype(np.uint8)
    steps=[{"id":"original","name":"原图","image":img_u8,"explanation":"输入图像"},
           {"id":"segmentation","name":str(nc)+"类分割结果","image":overlay,"explanation":"半透明颜色叠加:不同颜色=不同语义类别"}]
    return {"steps":steps,"metrics":{"num_classes":nc,"unique_labels":len(np.unique(seg))}}
