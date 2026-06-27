"""Pipeline builder for real DETR inference and teaching steps."""
from __future__ import annotations

import numpy as np

from app.modules.phase5_frontier.detr.algorithm import (
    detect_objects,
    visualize_query_attention,
)
from app.utils.image_utils import load_image_u8


def build_pipeline(image_path=None, threshold=0.5, **kwargs):
    th = float(kwargs.get('threshold', threshold))
    if not image_path:
        return {
            'error': 'DETR needs an input image so the real pretrained model can run.',
            'steps': [],
            'metrics': {'status': 'missing_input'},
        }

    try:
        img_u8 = load_image_u8(image_path, mode='rgb', max_side=768)
        result = detect_objects(img_u8, threshold=th)
    except Exception as exc:
        return {
            'error': f'DETR inference failed: {exc}',
            'steps': [],
            'metrics': {'status': 'model_not_available', 'error_type': type(exc).__name__},
        }

    detections = result['detections']
    h, w = img_u8.shape[:2]
    query_predictions = result.get('query_predictions', [])
    preprocess_vis = _tensor_preview(result.get('preprocess_tensor'))
    raw_query_vis = _draw_query_predictions(img_u8, query_predictions)
    score_vis = _query_score_chart(query_predictions)
    box_vis = _draw_detection_boxes(img_u8, detections)

    query_overlay = None
    if result.get('cross_attentions') is not None:
        maps = result['cross_attentions']['maps']
        query_vars = maps.mean(axis=0).var(axis=1)
        best_query = int(np.argmax(query_vars))
        query_hm = visualize_query_attention(result['cross_attentions'], query_idx=best_query)
        query_overlay = _overlay_heatmap_on_image(img_u8, query_hm) if query_hm is not None else None

    label_counts = {}
    for det in detections:
        label_counts[det['label']] = label_counts.get(det['label'], 0) + 1
    summary = ', '.join([f'{k} x {v}' for k, v in list(label_counts.items())[:8]]) or 'no object above threshold'

    steps = [
        {
            'id': 'input',
            'name': '输入图像',
            'image': img_u8,
            'formula': 'I in R^{H x W x 3}',
            'explanation': 'DETR 把目标检测改写成集合预测问题：整张图进入 CNN backbone 和 Transformer，固定数量的 object query 直接输出类别和边界框。',
            'data': {'height': int(h), 'width': int(w), 'channels': 3},
        },
        {
            'id': 'preprocess',
            'name': '预处理张量',
            'image': preprocess_vis,
            'formula': 'x = normalize(resize(I))',
            'explanation': 'HuggingFace DetrImageProcessor 会把图片 resize、转成 CxHxW 张量，并按模型训练时的均值和方差归一化。这一步是真实模型收到的输入表示。',
            'data': {'tensor_shape': list(np.asarray(result.get('preprocess_tensor')).shape) if result.get('preprocess_tensor') is not None else []},
        },
        {
            'id': 'query_candidates',
            'name': 'Object Query 原始候选',
            'image': raw_query_vis,
            'formula': '(class_i, box_i) = Head(q_i), i=1..100',
            'explanation': 'DETR 不使用 anchor 或滑窗候选框，而是让 100 个 object query 各自提出一个“我看到的物体”。图中展示模型真实最高分 query 的候选框。',
            'data': {'top_queries': query_predictions},
        },
        {
            'id': 'query_scores',
            'name': 'Query 分数过滤',
            'image': score_vis,
            'formula': 'score_i = (1 - p_i(no-object)) max_c p_i(c)',
            'explanation': '大多数 query 会学成 no-object。这里把高分 query 的 objectness 和类别概率合成分数，解释为什么只有少量 query 会留下来。',
        },
    ]

    if query_overlay is not None:
        steps.append({
            'id': 'cross_attention',
            'name': 'Object Query 交叉注意力',
            'image': query_overlay,
            'formula': 'CrossAttn(q_i, image_tokens)',
            'explanation': '解码器中的 object query 会通过交叉注意力“看向”图像 token。热区表示某个 query 在真实 DETR 中更关注的位置。',
        })

    steps.extend([
        {
            'id': 'detections',
            'name': f'阈值后检测结果 ({len(detections)} 个物体)',
            'image': box_vis,
            'formula': 'keep_i = score_i >= tau',
            'explanation': f'保留置信度高于 {th:.2f} 的真实模型输出，并把类别、分数和边界框叠加到原图上。',
            'data': {'detections': detections[:10]},
        },
        {
            'id': 'matching_principle',
            'name': '集合预测与匈牙利匹配',
            'image': score_vis,
            'formula': 'min_perm sum_i cost(y_i, y_hat_{perm(i)})',
            'explanation': f'检测摘要：{summary}。训练时 DETR 用匈牙利算法把预测集合和真值集合一一匹配，因此推理时不需要传统检测器那种大量 anchor 后处理。',
        },
    ])

    return {
        'steps': steps,
        'metrics': {
            'status': 'pretrained_model',
            'backend': 'transformers',
            'model': 'facebook/detr-resnet-50',
            'num_detections': len(detections),
            'num_queries': result['num_queries'],
            'threshold': th,
            'unique_labels': len(label_counts),
        },
        'detections': detections,
    }


def _tensor_preview(tensor):
    if tensor is None:
        return np.zeros((224, 224, 3), dtype=np.uint8) + 240
    arr = np.asarray(tensor, dtype=np.float32)
    if arr.ndim != 3:
        return np.zeros((224, 224, 3), dtype=np.uint8) + 240
    arr = np.transpose(arr, (1, 2, 0))
    lo = np.percentile(arr, 1, axis=(0, 1), keepdims=True)
    hi = np.percentile(arr, 99, axis=(0, 1), keepdims=True)
    vis = (arr - lo) / np.maximum(hi - lo, 1e-8)
    return np.clip(vis * 255, 0, 255).astype(np.uint8)


def _cxcywh_to_xyxy(box, w, h):
    cx, cy, bw, bh = [float(v) for v in box]
    return [
        (cx - bw / 2) * w,
        (cy - bh / 2) * h,
        (cx + bw / 2) * w,
        (cy + bh / 2) * h,
    ]


def _draw_box(out, box, color, width=3):
    h, w = out.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return
    out[y1:y1 + width, x1:x2] = color
    out[max(y1, y2 - width + 1):y2 + 1, x1:x2] = color
    out[y1:y2, x1:x1 + width] = color
    out[y1:y2, max(x1, x2 - width + 1):x2 + 1] = color


def _draw_detection_boxes(img, detections):
    out = img.copy()
    colors = [(239, 68, 68), (34, 197, 94), (59, 130, 246), (217, 119, 6), (124, 58, 237)]
    for i, det in enumerate(detections):
        _draw_box(out, det['box'], colors[i % len(colors)], width=3)
    return out


def _draw_query_predictions(img, query_predictions):
    out = img.copy()
    h, w = out.shape[:2]
    colors = [(234, 88, 12), (37, 99, 235), (22, 163, 74), (147, 51, 234), (220, 38, 38)]
    for i, q in enumerate(query_predictions[:12]):
        box = _cxcywh_to_xyxy(q.get('box_cxcywh', [0, 0, 0, 0]), w, h)
        _draw_box(out, box, colors[i % len(colors)], width=2)
    return out


def _query_score_chart(query_predictions, width=640, height=280):
    from PIL import Image, ImageDraw

    img = Image.new('RGB', (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(img)
    draw.text((18, 12), 'Top object-query scores before final filtering', fill=(15, 23, 42))
    if not query_predictions:
        draw.text((18, 60), 'No query data returned.', fill=(100, 116, 139))
        return np.array(img)
    max_score = max(float(q.get('score', 0)) for q in query_predictions) or 1.0
    bar_h = 17
    for i, q in enumerate(query_predictions[:10]):
        y = 50 + i * 22
        score = float(q.get('score', 0))
        bw = int(score / max_score * (width - 250))
        draw.rectangle((160, y, 160 + bw, y + bar_h), fill=(37, 99, 235))
        draw.text((18, y), f"q{q.get('query')} {str(q.get('label'))[:15]}", fill=(51, 65, 85))
        draw.text((170 + bw, y), f'{score:.3f}', fill=(71, 85, 105))
    return np.array(img)


def _overlay_heatmap_on_image(img, heatmap_2d):
    from PIL import Image

    h, w = img.shape[:2]
    hm = np.asarray(heatmap_2d, dtype=np.float64)
    hm = np.clip(hm, 0, 1)
    hm_img = np.array(Image.fromarray((hm * 255).astype(np.uint8)).resize((w, h), Image.LANCZOS))
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    hm_f = hm_img.astype(np.float32) / 255.0
    overlay[:, :, 0] = (hm_f * 255).astype(np.uint8)
    overlay[:, :, 1] = (hm_f * 80).astype(np.uint8)
    return np.clip(img.astype(np.float32) * 0.45 + overlay.astype(np.float32) * 0.55, 0, 255).astype(np.uint8)
