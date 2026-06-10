"""
图像工具纯函数。
所有函数保证：输入校验 → 处理 → 输出，无副作用。
"""
import numpy as np
import imageio.v3 as iio
import base64
import io
from PIL import Image


def check_shape(arr, name='array', min_dim=2, max_dim=4):
    """防御性断言：确保 NumPy 数组维度在合理范围内。"""
    assert isinstance(arr, np.ndarray), f'{name} 必须是 np.ndarray，实际为 {type(arr).__name__}'
    assert min_dim <= arr.ndim <= max_dim, \
        f'{name} 维度应为 {min_dim}-{max_dim}，实际为 {arr.ndim}'
    return arr


def load_image(path, mode='rgb'):
    """
    从磁盘读取图像，统一转为 RGB 的 float32 数组 [0, 1]。
    mode: 'rgb' | 'gray'
    """
    img = iio.imread(path)
    img = np.asarray(img, dtype=np.float32) / 255.0
    check_shape(img, 'loaded image')

    if mode == 'gray' and img.ndim == 3:
        # RGB → Gray: 加权平均，符合人眼感知
        img = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    elif mode == 'rgb' and img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    return img.astype(np.float32)


def save_image(arr, path):
    """将 float32 数组 [0, 1] 保存为 PNG。"""
    arr = to_uint8(arr)
    iio.imwrite(path, arr)


def to_uint8(arr):
    """float32 [0, 1] → uint8 [0, 255]，安全裁剪。"""
    arr = np.clip(np.asarray(arr, dtype=np.float64), 0.0, 1.0)
    return (arr * 255.0).round().astype(np.uint8)


def to_float32(arr):
    """任意图像数组 → float32 [0, 1]。"""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.max() > 1.0:
        arr /= 255.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def ensure_3channel(arr):
    """若为单通道 (H,W)，复制为三通道 (H,W,3)，方便统一渲染。"""
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return np.stack([arr, arr, arr], axis=-1)
    return arr


def to_base64(arr):
    """NumPy 图像数组 → base64 PNG 字符串（用于 API 返回）。"""
    arr = ensure_3channel(arr)
    arr_u8 = to_uint8(arr)
    img = Image.fromarray(arr_u8)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('ascii')
