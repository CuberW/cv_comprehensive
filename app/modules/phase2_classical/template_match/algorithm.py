
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from imageio.v3 import imread
import io, base64
from PIL import Image, ImageDraw

def ncc_match(img, template):
    """Vectorized NCC template matching using sliding_window_view."""
    h, w = img.shape[:2]
    th, tw = template.shape[:2]
    # Extract all windows: shape (H-th+1, W-tw+1, th, tw)
    windows = sliding_window_view(img, (th, tw))
    n_windows = (h - th + 1) * (w - tw + 1)
    windows_flat = windows.reshape(n_windows, th * tw).astype(np.float64)

    tm_flat = template.ravel().astype(np.float64)
    tm_mean = tm_flat.mean()
    tm_std = float(tm_flat.std()) or 1e-8
    tm_norm = (tm_flat - tm_mean) / tm_std

    p_mean = windows_flat.mean(axis=1, keepdims=True)
    p_std = windows_flat.std(axis=1, keepdims=True)
    p_std = np.maximum(p_std, 1e-8)
    p_norm = (windows_flat - p_mean) / p_std

    ncc = (p_norm * tm_norm).sum(axis=1) / (th * tw)
    return ncc.reshape(h - th + 1, w - tw + 1)
def _b64(arr):b=io.BytesIO();Image.fromarray(arr).save(b,'PNG');return base64.b64encode(b.getvalue()).decode()
def build_pipeline(upload_path):
    img=imread(upload_path)
    if img.ndim==3 and img.shape[2]==4:img=img[:,:,:3]
    gray=img if img.ndim==2 else (0.299*img[:,:,0]+0.587*img[:,:,1]+0.114*img[:,:,2]).astype(np.uint8)
    h,w=gray.shape;ch,cw=h//2,w//2;ts=60
    tpl=gray[ch-ts//2:ch+ts//2,cw-ts//2:cw+ts//2]
    r=ncc_match(gray,tpl)
    rv=np.clip((r-r.min())/(r.max()-r.min()+1e-8)*255,0,255).astype(np.uint8)
    viz=img.copy();by,bx=np.unravel_index(r.argmax(),r.shape)
    pi=Image.fromarray(viz);d=ImageDraw.Draw(pi)
    d.rectangle([bx,by,bx+ts,by+ts],outline=(0,255,0),width=2)
    return {'steps':[{'id':'original','name':'原始图像','image_base64':_b64(img)},{'id':'template','name':'模板 (60x60)','image_base64':_b64(tpl)},{'id':'ncc','name':'NCC响应图','image_base64':_b64(rv)},{'id':'result','name':'最佳匹配位置','image_base64':_b64(np.array(pi))}]}
