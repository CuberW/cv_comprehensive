"""Pipeline builder for DETR module."""
import numpy as np
from app.utils.image_utils import load_image_u8
from app.modules.phase5_frontier.detr.algorithm import (
    detect_objects, visualize_query_attention,
)


def build_pipeline(image_path=None, threshold=0.5, **kwargs):
    th = float(kwargs.get('threshold', threshold))
    if not image_path:
        return {
            'error': 'DETR demo requires an uploaded image.',
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
            'metrics': {'status': 'model_not_available'},
        }
    detections = result['detections']
    h, w = img_u8.shape[:2]

    # Draw detection boxes
    box_vis = img_u8.copy()
    colors = [(239, 68, 68), (34, 197, 94), (59, 130, 246), (217, 119, 6), (124, 58, 237)]
    for i, det in enumerate(detections):
        col = colors[i % len(colors)]
        x1, y1, x2, y2 = [int(v) for v in det['box']]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w-1, x2), min(h-1, y2)
        if x2 > x1 and y2 > y1:
            bw, bh = x2-x1, y2-y1
            box_vis[y1:y1+3, x1:x2] = col
            box_vis[y2-2:y2+1, x1:x2] = col
            box_vis[y1:y2, x1:x1+3] = col
            box_vis[y1:y2, x2-2:x2+1] = col

    # Query attention visualization
    query_hm = None
    if result['cross_attentions'] is not None:
        # Find query with highest attention variance (most "focused")
        maps = result['cross_attentions']['maps']  # (heads, queries, seq_len)
        query_vars = maps.mean(axis=0).var(axis=1)  # variance across positions
        best_query = int(np.argmax(query_vars))
        query_hm = visualize_query_attention(result['cross_attentions'], query_idx=best_query)
        query_overlay = _overlay_heatmap_on_image(img_u8, query_hm) if query_hm is not None else None
    else:
        query_overlay = None

    # Classification summary
    label_counts = {}
    for det in detections:
        lb = det['label']
        label_counts[lb] = label_counts.get(lb, 0) + 1
    summary = ', '.join([f"{k}×{v}" for k, v in list(label_counts.items())[:8]])

    steps = [
        {'id': 'original', 'name': '输入图像', 'image': img_u8,
         'explanation': '原始图像。DETR 用 CNN Backbone + Transformer Encoder-Decoder 端到端做检测——不需要 Anchor、NMS、Region Proposal'},
        {'id': 'detections', 'name': f'检测结果 ({len(detections)} 个物体)', 'image': box_vis,
         'explanation': f'置信度阈值 > {th}。每个框直接由 100 个 Object Query 通过 Transformer Decoder 预测出来。DETR 用匈牙利算法做二分匹配来配对预测与真值'},
    ]

    if query_overlay is not None:
        steps.append({
            'id': 'query_attn', 'name': 'Object Query 交叉注意力',
            'image': query_overlay,
            'explanation': f'某个 Object Query 在图像上的关注区域（解码器交叉注意力）。高亮处 = 这个 Query "在找"的目标位置。DETR 的核心创新：Object Query 是可学习的"提问向量"——每个 Query 专门寻找某种类型的物体',
        })

    steps.append({
        'id': 'summary', 'name': '检测摘要', 'image': box_vis,
        'explanation': f'检测到: {summary or "无物体"}。DETR 与 YOLO/Faster R-CNN 的区别：没有 Anchor 设计、没有 NMS 后处理——完全端到端',
    })

    return {
        'steps': steps,
        'metrics': {
            'model': 'DETR (ResNet-50)',
            'num_detections': len(detections),
            'num_queries': result['num_queries'],
            'threshold': th,
            'unique_labels': len(label_counts),
        },
        'detections': detections,
    }


def _overlay_heatmap_on_image(img, heatmap_2d):
    from PIL import Image
    h, w = img.shape[:2]
    hm = np.asarray(heatmap_2d, dtype=np.float64)
    hm = np.clip(hm, 0, 1)
    hm_img = np.array(Image.fromarray((hm*255).astype(np.uint8)).resize((w, h), Image.LANCZOS))
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    hm_f = hm_img.astype(np.float32) / 255.0
    overlay[:,:,0] = (hm_f*255).astype(np.uint8)
    overlay[:,:,1] = (hm_f*80).astype(np.uint8)
    return np.clip(img.astype(np.float32)*0.45 + overlay.astype(np.float32)*0.55, 0, 255).astype(np.uint8)
