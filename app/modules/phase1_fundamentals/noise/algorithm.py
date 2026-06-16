import numpy as np
from imageio.v3 import imread
import io, base64
from PIL import Image

def add_salt_pepper(img, amount=0.02):
    out = img.copy()
    h, w = img.shape[:2]
    n = int(h * w * amount)
    for _ in range(n):
        y, x = np.random.randint(0, h), np.random.randint(0, w)
        out[y, x] = [255, 255, 255] if np.random.random() > 0.5 else [0, 0, 0]
    return out

def add_gaussian_noise(img, sigma=25):
    noise = np.random.normal(0, sigma, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def _b64(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, 'PNG')
    return base64.b64encode(buf.getvalue()).decode()

def build_pipeline(upload_path):
    img = imread(upload_path)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    sp = add_salt_pepper(img, 0.03)
    gn = add_gaussian_noise(img, 25)
    return {
        'steps': [
            {'id': 'original', 'name': '原始图像', 'image_base64': _b64(img)},
            {'id': 'salt_pepper', 'name': '椒盐噪声 (3%)', 'image_base64': _b64(sp)},
            {'id': 'gaussian_noise', 'name': '高斯噪声 (sigma=25)', 'image_base64': _b64(gn)},
        ]
    }
