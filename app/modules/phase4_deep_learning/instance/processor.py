"""Demo processor for 实例分割."""
import numpy as np
from app.utils.image_utils import load_image_u8
from app.modules.phase4_deep_learning.instance.algorithm import simple_instance_demo,colorize_instances


def _to_uint8_heat(arr):
    """Float array -> uint8 heatmap."""
    arr=np.asarray(arr,dtype=np.float64)
    m=arr.max() if arr.size else 1.0
    if m<=1e-12: return np.zeros(arr.shape,dtype=np.uint8)
    return np.clip(arr/m*255,0,255).astype(np.uint8)


def build_pipeline(image_path=None, **kwargs):
    img_u8 = load_image_u8(image_path, mode='rgb', max_side=512) if image_path else np.zeros((64,64,3), dtype=np.uint8)
    h,w=img_u8.shape[:2]
    ni=int(kwargs.get("num_instances",3))
    insts=simple_instance_demo(img_u8,num_instances=ni)
    inst_vis=colorize_instances(insts,img_u8.shape)
    alpha=0.4; overlay=(img_u8*(1-alpha)+inst_vis*alpha).astype(np.uint8)
    box_vis=img_u8.copy()
    colors_box=[(239,68,68),(34,197,94),(59,130,246)]
    for i,inst in enumerate(insts):
        col=colors_box[i%3]; x,y,bw,bh=[int(v) for v in inst["box"]]
        x,y=max(0,x),max(0,y); bw,bh=min(bw,w-x),min(bh,h-y)
        if bw>0 and bh>0:
            box_vis[y:y+2,x:x+bw]=col; box_vis[y+bh:y+bh+2,x:x+bw]=col
            box_vis[y:y+bh,x:x+2]=col; box_vis[y:y+bh,x+bw:x+bw+2]=col
    steps=[{"id":"original","name":"原图","image":img_u8,"explanation":"输入图像"},
           {"id":"boxes","name":"检测框","image":box_vis,"explanation":"每个彩色框=一个检测到的实例"},
           {"id":"instances","name":"实例掩码","image":overlay,"explanation":"不同颜色叠加="+str(len(insts))+"个不同实例的分割掩码"}]
    return {"steps":steps,"metrics":{"instances":len(insts),"classes":len(set(i["class_id"] for i in insts))}}
