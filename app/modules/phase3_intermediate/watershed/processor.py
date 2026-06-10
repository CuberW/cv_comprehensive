"""Demo processor for 分水岭分割."""
import numpy as np
import imageio.v3 as iio
from app.modules.phase1_fundamentals.grayscale.algorithm import to_uint8
from app.modules.phase3_intermediate.watershed.algorithm import watershed_segmentation,make_markers_using_distance
from app.modules.phase2_classical.edge.algorithm import sobel_gradients,to_gray


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
    gray=to_gray(img_u8) if img_u8.ndim==3 else img_u8
    _,_,grad,_=sobel_gradients(gray.astype(np.float32))
    try:
        from app.modules.phase1_fundamentals.threshold.algorithm import otsu_threshold,global_threshold
        t=otsu_threshold(gray); binary=global_threshold(gray,t)
        markers=make_markers_using_distance(binary,min_dist=15)
        labels=watershed_segmentation(grad,markers)
        n_labels=len(np.unique(labels))-1
        colors=np.array([[220,38,38],[37,99,235],[5,150,105],[217,119,6],[124,58,237],[251,191,36]],dtype=np.uint8)
        h,w=labels.shape; seg_vis=np.zeros((h,w,3),dtype=np.uint8)
        for c in range(1,min(n_labels+1,7)):
            seg_vis[labels==c]=colors[(c-1)%6]
        boundary=np.zeros((h,w),dtype=bool)
        for y in range(1,h-1):
            for x in range(1,w-1):
                p=labels[y,x]
                if p>0 and (labels[y-1,x]!=p or labels[y+1,x]!=p or labels[y,x-1]!=p or labels[y,x+1]!=p):
                    boundary[y,x]=True
        seg_vis[boundary]=[255,255,255]
    except Exception:
        seg_vis=img_u8 if img_u8.ndim==3 else np.stack([img_u8]*3,axis=-1)
        n_labels=0
    steps=[{"id":"original","name":"原图","image":img_u8,"explanation":"输入图像"},
           {"id":"gradient","name":"梯度幅值(地形)","image":np.clip(grad/grad.max()*255,0,255).astype(np.uint8),"explanation":"看作地形:亮=山脊,暗=盆地"},
           {"id":"markers","name":"种子标记","image":np.where(markers>0,255,0).astype(np.uint8),"explanation":"确定各区域的起始位置"},
           {"id":"result","name":"分割结果","image":seg_vis,"explanation":"不同颜色=不同区域,白线=分水岭边界"}]
    return {"steps":steps,"metrics":{"regions":int(n_labels)}}
