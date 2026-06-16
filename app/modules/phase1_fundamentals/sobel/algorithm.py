import numpy as np
from imageio.v3 import imread
import io, base64
from PIL import Image

def sobel_x():
    return np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

def sobel_y():
    return np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

def apply_sobel(img):
    if img.ndim == 3:
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    else:
        gray = img.astype(np.float64)
    h, w = gray.shape
    ks = 3
    p = np.pad(gray, 1, mode='edge')
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    kx = sobel_x()
    ky = sobel_y()
    for i in range(h):
        for j in range(w):
            gx[i, j] = np.sum(p[i:i+ks, j:j+ks] * kx)
            gy[i, j] = np.sum(p[i:i+ks, j:j+ks] * ky)
    mag = np.clip(np.sqrt(gx**2 + gy**2), 0, 255).astype(np.uint8)
    ang = np.arctan2(gy, gx) * 180 / np.pi
    return mag, ang, gx, gy

def _b64(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, 'PNG')
    return base64.b64encode(buf.getvalue()).decode()

def build_pipeline(upload_path):
    img = imread(upload_path)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    mag, ang, gx, gy = apply_sobel(img)
    gxv = np.clip(np.abs(gx), 0, 255).astype(np.uint8)
    gyv = np.clip(np.abs(gy), 0, 255).astype(np.uint8)
    angv = ((ang + 180) / 360 * 255).astype(np.uint8)
    return {
        'steps': [
            {'id': 'original', 'name': '原始图像', 'image_base64': _b64(img)},
            {'id': 'gx', 'name': '水平梯度 Gx', 'image_base64': _b64(gxv)},
            {'id': 'gy', 'name': '垂直梯度 Gy', 'image_base64': _b64(gyv)},
            {'id': 'magnitude', 'name': '梯度幅值 sqrt(Gx^2+Gy^2)', 'image_base64': _b64(mag)},
            {'id': 'angle', 'name': '梯度方向', 'image_base64': _b64(angv)},
        ]
    }
