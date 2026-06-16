
import numpy as np
from imageio.v3 import imread
import io,base64
from PIL import Image
def bilateral_filter(img,d=5,sigma_color=50,sigma_space=10):
    h,w=img.shape[:2];img_f=img.astype(np.float64);out=np.zeros_like(img_f)
    half=d//2
    ax=np.linspace(-half,half,d);xx,yy=np.meshgrid(ax,ax)
    sk=np.exp(-(xx**2+yy**2)/(2*sigma_space**2))
    for ch in range(3) if img.ndim==3 else [0]:
        src=img_f[:,:,ch] if img.ndim==3 else img_f
        for i in range(half,h-half):
            for j in range(half,w-half):
                patch=src[i-half:i+half+1,j-half:j+half+1]
                rk=np.exp(-((patch-src[i,j])**2)/(2*sigma_color**2))
                wgt=sk*rk
                if img.ndim==3:out[i,j,ch]=np.sum(patch*wgt)/wgt.sum()
                else:out[i,j]=np.sum(patch*wgt)/wgt.sum()
    return np.clip(out,0,255).astype(np.uint8)
def _b64(arr):b=io.BytesIO();Image.fromarray(arr).save(b,'PNG');return base64.b64encode(b.getvalue()).decode()
def build_pipeline(upload_path):
    img=imread(upload_path)
    if img.ndim==3 and img.shape[2]==4:img=img[:,:,:3]
    bf=bilateral_filter(img)
    return {'steps':[{'id':'original','name':'原始图像','image_base64':_b64(img)},{'id':'bilateral','name':'双边滤波 (d=5, color=50, space=10)','image_base64':_b64(bf)}]}
