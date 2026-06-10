"""Histogram pipeline builder."""
import numpy as np
import imageio.v3 as iio
from app.modules.phase1_fundamentals.histogram.algorithm import (
    compute_histogram, compute_rgb_histograms, histogram_equalization,
)
from app.utils.image_utils import to_uint8 as _to_uint8


def build_pipeline(image_path):
    """Build histogram analysis pipeline steps."""
    img = iio.imread(image_path)
    img_u8 = _to_uint8(img)
    if img_u8.ndim == 2:
        gray = img_u8
    else:
        gray = np.round(img_u8[:,:,0]*0.299 + img_u8[:,:,1]*0.587 + img_u8[:,:,2]*0.114).astype(np.uint8)

    hist = compute_histogram(gray)
    equalized = histogram_equalization(gray)
    rgb_hists = compute_rgb_histograms(img_u8) if img_u8.ndim == 3 else None

    peak_bin = int(np.argmax(hist))
    dark_pct = round(float(hist[:64].sum() / hist.sum() * 100), 1)
    bright_pct = round(float(hist[192:].sum() / hist.sum() * 100), 1)

    steps = [
        {'id': 'original', 'name': '原图', 'explanation': '输入图像'},
        {'id': 'gray', 'name': '灰度图', 'explanation': '转为灰度以计算亮度直方图'},
        {'id': 'histogram', 'name': '亮度直方图', 'explanation': f'X轴: 0-255亮度, Y轴: 像素数。峰值在{peak_bin}。暗部({dark_pct}%) 亮部({bright_pct}%)'},
        {'id': 'cdf', 'name': '累计分布函数(CDF)', 'explanation': 'CDF曲线。直方图均衡化将CDF拉成直线'},
        {'id': 'equalized', 'name': '均衡化结果', 'explanation': '均衡化后直方图分布更均匀，图像对比度增强'},
    ]

    return {
        'steps': steps, 'gray': gray, 'equalized': equalized,
        'histogram': hist.tolist(),
        'rgb_histograms': [h.tolist() for h in rgb_hists] if rgb_hists else None,
        'metrics': {
            'peak_bin': peak_bin, 'dark_pct': dark_pct, 'bright_pct': bright_pct,
            'mean': round(float(gray.mean()), 1), 'std': round(float(gray.std()), 1),
        },
    }
