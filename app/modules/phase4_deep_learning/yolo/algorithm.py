"""YOLO-style one-stage detection mechanism implemented with NumPy/Pillow.

This module intentionally does not claim to be a pretrained YOLO checkpoint.
It computes a real grid/objectness/box pipeline on the input image so the page
can teach the YOLO idea without returning frontend-only fake boxes.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw

from app.modules.offline_teaching import _load_or_fixture
from app.utils.image_utils import ensure_gray, load_image_u8


def build_pipeline(image_path=None, grid_size=7, score_threshold=0.34, iou_threshold=0.35, **kwargs):
    grid_size = int(kwargs.get('grid_size', grid_size))
    grid_size = max(4, min(12, grid_size))
    score_threshold = float(kwargs.get('score_threshold', kwargs.get('threshold', score_threshold)))
    iou_threshold = float(kwargs.get('iou_threshold', iou_threshold))

    img = _load_image(image_path)
    gray = ensure_gray(img).astype(np.float32) / 255.0
    rgb = img.astype(np.float32) / 255.0
    h, w = gray.shape

    smooth = _gaussian_blur(gray, sigma=1.2)
    gy, gx = np.gradient(smooth)
    edge_energy = _norm01(np.sqrt(gx * gx + gy * gy))
    local_mean = _gaussian_blur(rgb, sigma=4.0)
    color_contrast = _norm01(np.linalg.norm(rgb - local_mean, axis=2))
    objectness = _norm01(0.62 * edge_energy + 0.38 * color_contrast)

    cells, candidates = _grid_predictions(objectness, grid_size, score_threshold)
    kept = _nms(candidates, iou_threshold)

    grid_vis = _draw_grid(img, grid_size, cells)
    objectness_vis = _overlay_heatmap(img, objectness)
    candidate_vis = _draw_boxes(img, candidates[:32], color=(250, 204, 21), show_score=False)
    kept_vis = _draw_boxes(img, kept, color=(34, 197, 94), show_score=True)
    score_vis = _score_chart(candidates, kept, score_threshold)

    return {
        'module_id': 'yolo',
        'steps': [
            {
                'id': 'input',
                'name': '输入图像',
                'image': img,
                'formula': 'I in R^{H x W x 3}',
                'explanation': 'YOLO 的核心思想是整张图只前向一次，直接在空间位置上预测目标框、置信度和类别。本教学版使用同一张输入图像做真实后端计算。',
                'data': {'height': int(h), 'width': int(w), 'channels': 3},
            },
            {
                'id': 'grid',
                'name': f'{grid_size}x{grid_size} 网格划分',
                'image': grid_vis,
                'formula': 'cell(i,j)=I[y_i:y_{i+1}, x_j:x_{j+1}]',
                'explanation': '单阶段检测会把空间位置和预测绑定起来。这里把图像划成网格，并统计每个网格的目标性分数，模拟 YOLO “哪个格子负责哪个目标”的直觉。',
                'data': {'grid_size': grid_size, 'cell_count': grid_size * grid_size},
            },
            {
                'id': 'objectness',
                'name': '目标性特征图',
                'image': objectness_vis,
                'formula': 'obj = normalize(0.62 |grad(G_sigma*I)| + 0.38 ||I-G_rho*I||)',
                'explanation': '真实 YOLO 用卷积网络学习 objectness。本教学实现不伪造训练权重，而是用梯度能量和局部颜色对比构造一个可解释的目标性图，亮处更可能包含物体结构。',
                'data': {
                    'mean_objectness': round(float(objectness.mean()), 4),
                    'max_objectness': round(float(objectness.max()), 4),
                },
            },
            {
                'id': 'raw_boxes',
                'name': '每格一次性预测候选框',
                'image': candidate_vis,
                'formula': 'b=(c_x,c_y,w,h,obj), obj > tau',
                'explanation': '每个高分网格直接回归一个候选框。框的位置来自该格内高目标性像素的包围盒，分数来自该格的最大值和平均值。',
                'data': {
                    'threshold': round(score_threshold, 3),
                    'raw_candidates': len(candidates),
                    'top_candidates': _public_boxes(candidates[:8]),
                },
            },
            {
                'id': 'score_filter',
                'name': '分数过滤与重叠抑制',
                'image': score_vis,
                'formula': 'keep = NMS({b_i | obj_i > tau}, IoU < gamma)',
                'explanation': '单阶段检测通常会产生多个相近框。这里按分数排序并抑制高度重叠框，只保留更稳定的预测。',
                'data': {
                    'iou_threshold': round(iou_threshold, 3),
                    'kept': len(kept),
                    'suppressed': max(0, len(candidates) - len(kept)),
                },
            },
            {
                'id': 'detections',
                'name': '最终一阶段检测结果',
                'image': kept_vis,
                'formula': 'D={(box_i, object, score_i)}',
                'explanation': '最终结果来自后端对输入图像的真实计算。它展示 YOLO 范式的一次性网格预测流程，但不是预训练 YOLO 权重推理。',
                'data': {'detections': _public_boxes(kept)},
            },
        ],
        'metrics': {
            'status': 'local_mechanism',
            'backend': 'NumPy/Pillow',
            'algorithm': 'YOLO-style one-stage grid detector',
            'real_pretrained_yolo': False,
            'grid_size': grid_size,
            'raw_candidates': len(candidates),
            'detections': len(kept),
            'note': '本页展示 YOLO 的单阶段检测机制；不冒充官方 YOLO 预训练权重。',
        },
    }


def _load_image(image_path):
    if image_path:
        return load_image_u8(image_path, mode='rgb', max_side=512)
    return _load_or_fixture(image_path=image_path)


def _gaussian_blur(arr, sigma=1.0):
    arr = np.asarray(arr, dtype=np.float32)
    radius = max(1, int(math.ceil(3 * sigma)))
    xs = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(xs * xs) / (2 * sigma * sigma))
    kernel /= kernel.sum()
    out = _convolve_axis(arr, kernel, axis=0)
    out = _convolve_axis(out, kernel, axis=1)
    return out.astype(np.float32)


def _convolve_axis(arr, kernel, axis):
    pad = len(kernel) // 2
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (pad, pad)
    padded = np.pad(arr, pad_width, mode='edge')
    out = np.zeros_like(arr, dtype=np.float32)
    for k, weight in enumerate(kernel):
        sl = [slice(None)] * arr.ndim
        sl[axis] = slice(k, k + arr.shape[axis])
        out += float(weight) * padded[tuple(sl)]
    return out


def _norm01(x):
    x = np.asarray(x, dtype=np.float32)
    mn = float(x.min())
    mx = float(x.max())
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn)).astype(np.float32)


def _grid_predictions(objectness, grid_size, threshold):
    h, w = objectness.shape
    cell_h = h / grid_size
    cell_w = w / grid_size
    cells = []
    candidates = []
    for gy in range(grid_size):
        for gx in range(grid_size):
            y0 = int(round(gy * cell_h))
            y1 = int(round((gy + 1) * cell_h))
            x0 = int(round(gx * cell_w))
            x1 = int(round((gx + 1) * cell_w))
            patch = objectness[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            mean_score = float(patch.mean())
            max_score = float(patch.max())
            score = 0.55 * max_score + 0.45 * mean_score
            cells.append({'grid': [gx, gy], 'box': [x0, y0, x1, y1], 'score': round(score, 4)})
            if score < threshold:
                continue
            active = patch >= max(float(np.percentile(patch, 72)), threshold * 0.75)
            yy, xx = np.where(active)
            if len(xx) >= 3 and len(yy) >= 3:
                bx0 = x0 + int(xx.min())
                bx1 = x0 + int(xx.max()) + 1
                by0 = y0 + int(yy.min())
                by1 = y0 + int(yy.max()) + 1
            else:
                bx0, by0, bx1, by1 = x0, y0, x1, y1
            pad_x = max(3, int((bx1 - bx0) * 0.22))
            pad_y = max(3, int((by1 - by0) * 0.22))
            box = [
                max(0, bx0 - pad_x),
                max(0, by0 - pad_y),
                min(w - 1, bx1 + pad_x),
                min(h - 1, by1 + pad_y),
            ]
            if box[2] - box[0] >= 5 and box[3] - box[1] >= 5:
                candidates.append({
                    'box': box,
                    'score': round(float(score), 4),
                    'label': 'object',
                    'grid': [gx, gy],
                })
    candidates.sort(key=lambda row: row['score'], reverse=True)
    return cells, candidates


def _nms(candidates, iou_threshold):
    kept = []
    for candidate in candidates:
        if all(_iou(candidate['box'], item['box']) < iou_threshold for item in kept):
            kept.append(candidate)
        if len(kept) >= 12:
            break
    return kept


def _iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _draw_grid(img, grid_size, cells):
    canvas = Image.fromarray(img.copy())
    draw = ImageDraw.Draw(canvas, 'RGBA')
    w, h = canvas.size
    max_score = max([cell['score'] for cell in cells] or [1.0])
    for cell in cells:
        x0, y0, x1, y1 = cell['box']
        alpha = int(30 + 110 * (cell['score'] / max_score))
        draw.rectangle((x0, y0, x1, y1), fill=(59, 130, 246, alpha))
    for i in range(grid_size + 1):
        x = int(round(i * w / grid_size))
        y = int(round(i * h / grid_size))
        draw.line((x, 0, x, h), fill=(226, 232, 240, 150), width=1)
        draw.line((0, y, w, y), fill=(226, 232, 240, 150), width=1)
    return np.array(canvas.convert('RGB'))


def _overlay_heatmap(img, heat):
    heat = np.clip(heat, 0, 1)
    color = np.zeros((*heat.shape, 3), dtype=np.float32)
    color[..., 0] = 245 * heat + 15 * (1 - heat)
    color[..., 1] = 158 * heat + 23 * (1 - heat)
    color[..., 2] = 11 * heat + 42 * (1 - heat)
    out = img.astype(np.float32) * 0.48 + color * 0.52
    return np.clip(out, 0, 255).astype(np.uint8)


def _draw_boxes(img, rows: Iterable[dict], color=(34, 197, 94), show_score=True):
    canvas = Image.fromarray(img.copy())
    draw = ImageDraw.Draw(canvas, 'RGBA')
    for row in rows:
        x0, y0, x1, y1 = [int(round(v)) for v in row['box']]
        draw.rectangle((x0, y0, x1, y1), outline=(*color, 255), width=3)
        if show_score:
            text = f"{row.get('label', 'object')} {float(row.get('score', 0)):.2f}"
            draw.rectangle((x0, max(0, y0 - 22), min(canvas.size[0], x0 + 112), y0), fill=(*color, 220))
            draw.text((x0 + 5, max(0, y0 - 19)), text, fill=(255, 255, 255, 255))
    return np.array(canvas.convert('RGB'))


def _score_chart(candidates, kept, threshold, width=640, height=260):
    canvas = Image.new('RGB', (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 14), 'Objectness scores: raw candidates -> kept boxes', fill=(15, 23, 42))
    rows = candidates[:12]
    if not rows:
        draw.text((18, 118), 'No cell score passed the threshold.', fill=(100, 116, 139))
        return np.array(canvas)
    kept_ids = {tuple(row['box']) for row in kept}
    for i, row in enumerate(rows):
        y = 48 + i * 16
        score = float(row['score'])
        col = (34, 197, 94) if tuple(row['box']) in kept_ids else (245, 158, 11)
        draw.rectangle((160, y, 160 + int(390 * score), y + 10), fill=col)
        draw.text((18, y - 2), f"grid {row.get('grid', '')}", fill=(71, 85, 105))
        draw.text((560, y - 2), f'{score:.2f}', fill=(15, 23, 42))
    tx = 160 + int(390 * threshold)
    draw.line((tx, 42, tx, height - 24), fill=(239, 68, 68), width=2)
    draw.text((tx + 4, height - 38), f'tau={threshold:.2f}', fill=(239, 68, 68))
    return np.array(canvas)


def _public_boxes(rows):
    return [
        {
            'box': [round(float(v), 1) for v in row['box']],
            'score': round(float(row.get('score', 0)), 4),
            'label': row.get('label', 'object'),
            'grid': row.get('grid'),
        }
        for row in rows
    ]
