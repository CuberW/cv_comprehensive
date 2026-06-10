"""Demo processor for 扩散模型."""
import numpy as np
from app.utils.image_utils import load_image_u8
from app.modules.phase4_deep_learning.diffusion.algorithm import forward_diffusion


def _to_uint8_heat(arr):
    """Float array -> uint8 heatmap."""
    arr=np.asarray(arr,dtype=np.float64)
    m=arr.max() if arr.size else 1.0
    if m<=1e-12: return np.zeros(arr.shape,dtype=np.uint8)
    return np.clip(arr/m*255,0,255).astype(np.uint8)


def build_pipeline(image_path=None, **kwargs):
    gray = load_image_u8(image_path, mode='gray', max_side=256) if image_path else np.zeros((64,64), dtype=np.uint8)
    small=gray[::2,::2].astype(np.float64)/255.0
    ns=int(kwargs.get("num_steps",50))
    steps_out=forward_diffusion(small,num_steps=200)
    # Pick key steps for display
    key=[0,2,5,10,20,50,100,150,199]
    key=[k for k in key if k<len(steps_out)]
    h_s,w_s=small.shape; grid=np.zeros((h_s,len(key)*h_s),dtype=np.uint8)
    for i,k in enumerate(key):
        img_k=steps_out[k]["image"]
        img_k=np.clip(img_k,0,1)
        grid[:,i*h_s:(i+1)*h_s]=(img_k*255).astype(np.uint8)
    steps=[{"id":"forward","name":"前向加噪过程(t=0->199)","image":grid,"explanation":"从左到右:逐步加入高斯噪声,原图逐渐变为纯噪声"},
           {"id":"reverse","name":"逆向去噪","image":gray,"explanation":"真实扩散模型用UNet预测噪声并逐步去除(此处为概念展示)"}]
    return {"steps":steps,"metrics":{"total_steps":len(steps_out),"displayed":len(key)}}
