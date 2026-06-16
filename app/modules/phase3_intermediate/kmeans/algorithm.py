
import numpy as np
from imageio.v3 import imread
import io,base64
from PIL import Image
def kmeans_segment(img,k=4,max_iter=15):
    h,w,c=img.shape;data=img.reshape(-1,3).astype(np.float64)
    rng=np.random.default_rng(42);cents=data[rng.choice(len(data),k,replace=False)]
    for _ in range(max_iter):
        dists=np.linalg.norm(data[:,None]-cents[None],axis=2);labels=np.argmin(dists,axis=1)
        new_cents=np.array([data[labels==i].mean(axis=0) if (labels==i).sum()>0 else data[rng.choice(len(data))] for i in range(k)])
        if np.allclose(cents,new_cents):break
        cents=new_cents
    colors=np.array([[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],[0,255,255],[128,0,0],[0,128,0]])
    return colors[labels%len(colors)].reshape(h,w,3).astype(np.uint8)
def _b64(arr):b=io.BytesIO();Image.fromarray(arr).save(b,'PNG');return base64.b64encode(b.getvalue()).decode()
def build_pipeline(upload_path):
    img=imread(upload_path)
    if img.ndim==3 and img.shape[2]==4:img=img[:,:,:3]
    r=[{'id':'original','name':'原始图像','image_base64':_b64(img)}]
    for k in[3,5,7]:
        r.append({'id':f'k{k}','name':f'K-Means (K={k})','image_base64':_b64(kmeans_segment(img,k))})
    return {'steps':r}
