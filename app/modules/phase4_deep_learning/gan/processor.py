"""Demo processor for GAN 生成对抗网络."""
import numpy as np
import imageio.v3 as iio
from app.modules.phase1_fundamentals.grayscale.algorithm import to_uint8
from app.modules.phase4_deep_learning.gan.algorithm import SimpleGAN


def _to_uint8_heat(arr):
    """Float array -> uint8 heatmap."""
    arr=np.asarray(arr,dtype=np.float64)
    m=arr.max() if arr.size else 1.0
    if m<=1e-12: return np.zeros(arr.shape,dtype=np.uint8)
    return np.clip(arr/m*255,0,255).astype(np.uint8)


def build_pipeline(image_path=None, **kwargs):
    img_u8 = to_uint8(iio.imread(image_path)) if image_path else np.zeros((64,64,3), dtype=np.uint8)
    
    gan=SimpleGAN(noise_dim=10,image_dim=784)
    rng=np.random.default_rng(42)
    noise=rng.normal(0,1,(8,10)).astype(np.float64)
    fake=gan.generate(noise)
    fake_grid=np.zeros((28*2+4,28*4+8),dtype=np.uint8)
    for i in range(8):
        row,col=i//4,i%4
        y0=row*30; x0=col*30
        img_28=fake[i].reshape(28,28)
        img_28=(img_28-img_28.min())/(img_28.max()-img_28.min()+1e-8)*255
        fake_grid[y0:y0+28,x0:x0+28]=img_28.astype(np.uint8)
    noise_vis=np.zeros((10,100,3),dtype=np.uint8)
    for i in range(8):
        for j in range(10):
            v=int((noise[i,j]+3)/6*255)
            noise_vis[j,i*12:i*12+10]=[max(0,min(255,v)),max(0,min(255,128-int(v/2))),max(0,min(255,255-v))]
    steps=[{"id":"noise","name":"随机噪声(z)","image":noise_vis,"explanation":"8个10维噪声向量,每个将生成一张图像"},
           {"id":"generate","name":"生成器输出 G(z)","image":fake_grid,"explanation":"生成的8张28×28假图像"}]
    return {"steps":steps,"metrics":{"noise_dim":10,"generated":8}}
