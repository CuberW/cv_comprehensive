"""Demo processor for GrabCut 前景提取."""
import numpy as np
import imageio.v3 as iio
from app.modules.phase1_fundamentals.grayscale.algorithm import to_uint8
from app.modules.phase3_intermediate.grabcut.algorithm import grabcut_segment


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
    h,w=img_u8.shape[:2]
    x=int(kwargs.get("x",w//5)); y=int(kwargs.get("y",h//5))
    rw=int(kwargs.get("w",w*3//5)); rh=int(kwargs.get("h",h*3//5))
    mask=grabcut_segment(img_u8,(x,y,rw,rh))
    overlay=img_u8.copy() if img_u8.ndim==3 else np.stack([img_u8]*3,axis=-1)
    overlay[mask>0]=overlay[mask>0]//2+np.array([0,100,0],dtype=np.uint8)//2
    rect_vis=img_u8.copy() if img_u8.ndim==3 else np.stack([img_u8]*3,axis=-1)
    cv2_rect=rect_vis.copy()
    cv2_rect[y:y+2,x:x+rw]=[34,197,94]; cv2_rect[y+rh:y+rh+2,x:x+rw]=[34,197,94]
    cv2_rect[y:y+rh,x:x+2]=[34,197,94]; cv2_rect[y:y+rh,x+rw:x+rw+2]=[34,197,94]
    fg_pct=round(float(mask.sum()/255/mask.size)*100,1)
    steps=[{"id":"original","name":"原图","image":img_u8,"explanation":"输入图像"},
           {"id":"rect","name":"框选目标","image":cv2_rect,"explanation":"绿色矩形框=用户标记的目标区域"},
           {"id":"result","name":"分割结果","image":overlay,"explanation":"绿色半透明="+str(fg_pct)+"%的前景像素"}]
    return {"steps":steps,"metrics":{"foreground_pct":fg_pct,"rect":[x,y,rw,rh]}}
