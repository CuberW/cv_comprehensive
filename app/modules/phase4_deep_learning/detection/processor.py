"""Demo processor for 目标检测."""
import numpy as np
import imageio.v3 as iio
from app.modules.phase1_fundamentals.grayscale.algorithm import to_uint8
from app.modules.phase4_deep_learning.detection.algorithm import grid_detection_demo,non_max_suppression


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
    h,w=img_u8.shape[:2]
    dets=grid_detection_demo(img_u8.shape,grid_size=7,num_boxes=2,num_classes=3)
    class_names=["人","车","动物"]
    colors_det=[(239,68,68),(34,197,94),(59,130,246)]
    vis=img_u8.copy()
    for d in dets:
        col=colors_det[d["class_id"]%3]
        x,y,bw,bh=int(d["x"]-d["w"]/2),int(d["y"]-d["h"]/2),int(d["w"]),int(d["h"])
        x,y=max(0,x),max(0,y); bw, bh = min(bw, w-x), min(bh, h-y)
        if bw>0 and bh>0:
            vis[y:y+2,x:x+bw]=col; vis[y+bh:y+bh+2,x:x+bw]=col
            vis[y:y+bh,x:x+2]=col; vis[y:y+bh,x+bw:x+bw+2]=col
    grid_vis=img_u8.copy(); cell_h,cell_w=h//7,w//7
    for gy in range(8):
        y=gy*cell_h
        if y<h: grid_vis[y:y+1,:]=[100,100,100]
    for gx in range(8):
        x=gx*cell_w
        if x<w: grid_vis[:,x:x+1]=[100,100,100]
    steps=[{"id":"original","name":"原图","image":img_u8,"explanation":"输入图像"},
           {"id":"grid","name":"7×7网格","image":grid_vis,"explanation":"每个cell预测"+str(2)+"个边界框"},
           {"id":"detections","name":"检测结果","image":vis,"explanation":"检测到"+str(len(dets))+"个物体(NMS后)"}]
    return {"steps":steps,"metrics":{"detections":len(dets),"grid":"7x7","nms_threshold":0.4}}
