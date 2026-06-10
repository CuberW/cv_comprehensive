"""Threshold pipeline builder."""
import numpy as np
import imageio.v3 as iio
from app.modules.phase1_fundamentals.threshold.algorithm import (
    global_threshold, adaptive_mean_threshold, otsu_threshold,
)


def build_pipeline(image_path, method='otsu', threshold=128, block_size=11, C=2):
    """Build thresholding pipeline steps."""
    img = iio.imread(image_path)
    if img.ndim == 3:
        gray = np.round(img[:,:,0]*0.299 + img[:,:,1]*0.587 + img[:,:,2]*0.114).astype(np.uint8)
    else:
        gray = np.asarray(img, dtype=np.uint8)

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

    steps = [
        {'id': 'original', 'name': '原图', 'explanation': '输入图像'},
        {'id': 'gray', 'name': '灰度图', 'explanation': '阈值化需要先将彩色图转为灰度图'},
        {'id': 'histogram', 'name': '直方图 + 阈值线',
         'explanation': f'Otsu自动计算阈值={t}, 左侧=背景区域, 右侧=前景区域' if t
                        else f'自适应阈值 (block={block_size}, C={C}): 每个像素使用局部邻域均值作为阈值'},
        {'id': 'result', 'name': '二值化结果', 'explanation': '白色=前景(255), 黑色=背景(0)'},
    ]

    return {
        'steps': steps, 'gray': gray, 'result': result,
        'threshold': t,
        'metrics': {
            'method': method, 'threshold': t,
            'foreground_pct': round(float(result.sum()/255/result.size)*100, 1) if result.size > 0 else 0,
        },
    }
