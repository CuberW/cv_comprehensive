"""Histogram pipeline builder with chart images."""
import numpy as np
import imageio.v3 as iio
from app.modules.phase1_fundamentals.histogram.algorithm import (
    compute_histogram, compute_rgb_histograms, histogram_equalization,
)
from app.utils.image_utils import to_uint8 as _to_uint8


def _draw_histogram_chart(hist, width=400, height=200, highlight_t=None):
    """Draw a histogram bar chart as a NumPy image array."""
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
        y1 = height - 30
        color = [59, 130, 246]  # Blue bars
        if highlight_t is not None and i == highlight_t:
            color = [239, 68, 68]  # Red for threshold line
        chart[y0:y1, x0:x1] = color
    # Draw axis
    chart[height-28:height-26, :] = [100, 100, 100]
    chart[:, :2] = [100, 100, 100]
    if highlight_t is not None:
        tx = int(highlight_t / 256 * width)
        chart[5:height-30, max(0,tx-1):min(width,tx+2)] = [239, 68, 68]
    return chart


def _draw_cdf_chart(hist, width=400, height=200):
    """Draw CDF curve as a NumPy image array."""
    chart = np.ones((height, width, 3), dtype=np.uint8) * 248
    cdf = np.cumsum(hist)
    if cdf[-1] <= 0:
        return chart
    cdf_norm = cdf / cdf[-1]
    xs = np.linspace(0, width-1, len(cdf_norm)).astype(np.int32)
    ys = (height - 30 - cdf_norm * (height - 35)).astype(np.int32)
    for i in range(len(xs) - 1):
        x0, y0 = xs[i], ys[i]
        x1, y1 = xs[i+1], ys[i+1]
        n = max(abs(x1-x0), abs(y1-y0), 1)
        for t in np.linspace(0, 1, n):
            xx = int(x0 + (x1-x0)*t)
            yy = int(y0 + (y1-y0)*t)
            if 0 <= yy < height and 0 <= xx < width:
                chart[yy, xx] = [34, 197, 94]
    chart[height-28:height-26, :] = [100, 100, 100]
    chart[:, :2] = [100, 100, 100]
    return chart


def build_pipeline(image_path=None):
    """Build histogram analysis pipeline."""
    if image_path is None:
        img_u8 = np.zeros((64, 64, 3), dtype=np.uint8)
        gray = np.zeros((64, 64), dtype=np.uint8)
    else:
        img = iio.imread(image_path)
        img_u8 = _to_uint8(img)
        if img_u8.ndim == 2:
            gray = img_u8
        else:
            gray = np.round(img_u8[:,:,0]*0.299 + img_u8[:,:,1]*0.587 + img_u8[:,:,2]*0.114).astype(np.uint8)

    if img_u8.ndim == 2:
        img_u8_rgb = np.stack([img_u8]*3, axis=-1)
    else:
        img_u8_rgb = img_u8[..., :3]

    hist = compute_histogram(gray)
    equalized = histogram_equalization(gray)
    eq_rgb = np.stack([equalized]*3, axis=-1) if equalized.ndim == 2 else equalized
    hist_chart = _draw_histogram_chart(hist)
    cdf_chart = _draw_cdf_chart(hist)

    peak_bin = int(np.argmax(hist))
    dark_pct = round(float(hist[:64].sum() / hist.sum() * 100), 1) if hist.sum() > 0 else 0
    bright_pct = round(float(hist[192:].sum() / hist.sum() * 100), 1) if hist.sum() > 0 else 0

    steps = [
        {'id': 'original', 'name': '原图', 'image': img_u8_rgb,
         'explanation': '输入图像'},
        {'id': 'gray', 'name': '灰度图', 'image': gray,
         'explanation': '转为灰度以计算亮度直方图'},
        {'id': 'histogram', 'name': '亮度直方图', 'image': hist_chart,
         'explanation': f'X轴: 0-255亮度, Y轴: 像素数。峰值在{peak_bin}。暗部({dark_pct}%) 亮部({bright_pct}%)'},
        {'id': 'cdf', 'name': '累计分布函数(CDF)', 'image': cdf_chart,
         'explanation': 'CDF曲线。直方图均衡化将CDF拉成直线'},
        {'id': 'equalized', 'name': '均衡化结果', 'image': eq_rgb,
         'explanation': '均衡化后直方图分布更均匀，图像对比度增强'},
    ]

    return {
        'steps': steps,
        'metrics': {
            'peak_bin': peak_bin, 'dark_pct': dark_pct, 'bright_pct': bright_pct,
            'mean': round(float(gray.mean()), 1), 'std': round(float(gray.std()), 1),
        },
    }
