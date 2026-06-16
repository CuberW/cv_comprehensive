
import numpy as np
from imageio.v3 import imread
import io,base64
from PIL import Image,ImageDraw
def ncc_match(img,template):
    h,w=img.shape[:2];th,tw=template.shape[:2];result=np.zeros((h-th+1,w-tw+1))
    tm=template.mean();ts=template.std() or 1e-8
    for i in range(h-th+1):
        for j in range(w-tw+1):
            p=img[i:i+th,j:j+tw];pm=p.mean();ps=p.std() or 1e-8
            result[i,j]=((p-pm)*(template-tm)).mean()/(ps*ts)
    return result
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
