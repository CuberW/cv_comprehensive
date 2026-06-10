"""Threshold pipeline builder with images."""
import numpy as np
from app.modules.phase1_fundamentals.threshold.algorithm import (
    global_threshold, adaptive_mean_threshold, otsu_threshold,
)
from app.utils.image_utils import load_image_u8, ensure_gray


def _draw_histogram_with_threshold(hist, threshold, width=400, height=200):
    """Draw histogram with a red threshold line."""
    chart = np.ones((height, width, 3), dtype=np.uint8) * 248
    hist = np.asarray(hist, dtype=np.float64)
    if hist.max() <= 0:
        return chart
    hist_norm = hist / hist.max()
    bar_w = max(1, width // len(hist))
    for i, v in enumerate(hist_norm):
        x0 = i * bar_w
        x1 = min(x0 + bar_w, width)
        bar_h = int(v * (height - 30))
        y0 = height - 30 - bar_h
        color = [59, 130, 246] if i < threshold else [147, 197, 253]
        chart[y0:height-30, x0:x1] = color
    chart[height-28:height-26, :] = [100, 100, 100]
    chart[:, :2] = [100, 100, 100]
    tx = int(threshold / 256 * width)
    chart[5:height-30, max(0,tx-1):min(width,tx+2)] = [239, 68, 68]
    return chart


def build_pipeline(image_path=None, method='otsu', threshold=128, block_size=11, C=2):
    """Build thresholding pipeline steps."""
    if image_path is None:
        img_u8 = np.zeros((64, 64, 3), dtype=np.uint8)
        gray = np.zeros((64, 64), dtype=np.uint8)
    else:
        img_u8 = load_image_u8(image_path, mode='rgb', max_side=1024)
        gray = ensure_gray(img_u8)

    if img_u8.ndim == 2:
        img_u8_rgb = np.stack([img_u8]*3, axis=-1)
    else:
        img_u8_rgb = img_u8[..., :3]

    if method == 'otsu':
        t = otsu_threshold(gray)
    elif method == 'adaptive':
        t = None
    else:
        t = threshold

    if method == 'otsu' or method == 'global':
        result = global_threshold(gray, t)
    elif method == 'adaptive':
        result = adaptive_mean_threshold(gray, block_size, C)
    else:
        result = global_threshold(gray, threshold)

    result_rgb = np.stack([result]*3, axis=-1)
    from app.modules.phase1_fundamentals.histogram.algorithm import compute_histogram
    hist = compute_histogram(gray)
    hist_chart = _draw_histogram_with_threshold(hist, t if t else 128)

    steps = [
        {'id': 'original', 'name': '原图', 'image': img_u8_rgb,
         'explanation': '输入图像'},
        {'id': 'gray', 'name': '灰度图', 'image': gray,
         'explanation': '阈值化需要先将彩色图转为灰度图'},
        {'id': 'histogram', 'name': f'直方图 (Otsu阈值={t})' if t else f'自适应阈值 (block={block_size})',
         'image': hist_chart,
         'explanation': f'蓝色=背景区域, 浅蓝=前景区域, 红色线=阈值={t}' if t
                        else f'每个像素用局部均值作为阈值'},
        {'id': 'result', 'name': '二值化结果', 'image': result_rgb,
         'explanation': '白色=前景(255), 黑色=背景(0)'},
    ]

    return {
        'steps': steps,
        'metrics': {
            'method': method, 'threshold': t,
            'foreground_pct': round(float(result.sum()/255/result.size)*100, 1) if result.size > 0 else 0,
        },
    }
