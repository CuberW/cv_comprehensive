"""Sobel gradient — uses old CV proven edge_optimized.sobel."""
import numpy as np
from imageio.v3 import imread
import io, base64
from PIL import Image
from app.modules.phase2_classical.edge.edge_optimized import sobel
from app.modules.phase2_classical.edge.edge import to_gray
def _b64(arr):
    buf=io.BytesIO();Image.fromarray(np.clip(arr,0,255).astype(np.uint8)).save(buf,'PNG')
    return base64.b64encode(buf.getvalue()).decode()
def build_pipeline(upload_path):
    img=imread(upload_path)
    if img.ndim==3 and img.shape[2]==4:img=img[:,:,:3]
    gray=to_gray(img);sx,sy,mag,ang=sobel(gray.astype(np.float32))
    gxv=np.clip(np.abs(sx),0,255).astype(np.uint8);gyv=np.clip(np.abs(sy),0,255).astype(np.uint8)
    magv=np.clip(mag,0,255).astype(np.uint8) if mag.max()<=255 else ((mag/mag.max())*255).clip(0,255).astype(np.uint8)
    angv=((ang+180)/360*255).astype(np.uint8)
    return {'steps':[
        {'id':'original','name':'原始图像','image_base64':_b64(img)},
        {'id':'gx','name':'水平梯度 Gx','image':gxv},
        {'id':'gy','name':'垂直梯度 Gy','image':gyv},
        {'id':'magnitude','name':'梯度幅值','image':magv},
        {'id':'angle','name':'梯度方向','image':angv},
    ]}
