import numpy as np
from imageio.v3 import imread
import io, base64
from PIL import Image

def gaussian_kernel(size=5, sigma=1.0):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return k / k.sum()

def apply_gaussian(img, ksize=5, sigma=1.5):
    k = gaussian_kernel(ksize, sigma)
    h, w = img.shape[:2]
    pad = ksize // 2
    if img.ndim == 3:
        out = np.zeros_like(img, dtype=np.float64)
        for c in range(3):
            p = np.pad(img[:, :, c], pad, mode='reflect')
            for i in range(h):
                for j in range(w):
                    out[i, j, c] = np.sum(p[i:i+ksize, j:j+ksize] * k)
        return np.clip(out, 0, 255).astype(np.uint8)
    p = np.pad(img, pad, mode='reflect')
    out = np.zeros_like(img, dtype=np.float64)
    for i in range(h):
        for j in range(w):
            out[i, j] = np.sum(p[i:i+ksize, j:j+ksize] * k)
    return np.clip(out, 0, 255).astype(np.uint8)

def _b64(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, 'PNG')
    return base64.b64encode(buf.getvalue()).decode()

def build_pipeline(upload_path):
    img = imread(upload_path)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    results = [{'id': 'original', 'name': '原始图像', 'image_base64': _b64(img)}]
    for ksize, sigma in [(3, 0.8), (5, 1.5), (7, 2.5)]:
        results.append({
            'id': f'gaussian_{ksize}x{ksize}_s{sigma}',
            'name': f'高斯模糊 ({ksize}x{ksize}, sigma={sigma})',
            'image_base64': _b64(apply_gaussian(img, ksize, sigma))
        })
    return {'steps': results}
