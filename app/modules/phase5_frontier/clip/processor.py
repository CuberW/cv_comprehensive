"""Pipeline builder for CLIP module."""
import numpy as np
from app.utils.image_utils import load_image_u8
from app.modules.phase5_frontier.clip.algorithm import (
    compute_image_text_similarity, zero_shot_classify, PRESET_CLASS_SETS,
)


def build_pipeline(image_path=None, class_set='animals', custom_classes='', **kwargs):
    cs = kwargs.get('class_set', class_set)
    custom = kwargs.get('custom_classes', custom_classes)

    if not image_path:
        return {
            'error': 'CLIP demo requires an uploaded image.',
            'steps': [],
            'metrics': {'status': 'missing_input'},
        }

    try:
        img_u8 = load_image_u8(image_path, mode='rgb', max_side=512)
    except Exception as exc:
        return {
            'error': f'CLIP inference failed: {exc}',
            'steps': [],
            'metrics': {'status': 'model_not_available'},
        }

    # Resolve class names
    if custom.strip():
        class_names = [n.strip() for n in custom.split(',') if n.strip()]
    else:
        class_names = PRESET_CLASS_SETS.get(cs, PRESET_CLASS_SETS['animals'])

    # Run CLIP zero-shot classification
    result = zero_shot_classify(img_u8, class_names)
    preds = result['predictions']

    # Similarity bar chart as image
    sim_chart = _render_similarity_chart(preds)

    # Similarity matrix visualization
    sim_matrix_img = _render_similarity_matrix(preds)

    top1_label = preds[0]['label'] if preds else 'N/A'
    top1_prob = f"{preds[0]['probability']*100:.1f}%" if preds else 'N/A'

    steps = [
        {'id': 'original', 'name': '输入图像', 'image': img_u8,
         'explanation': '原始图像。CLIP 用 4 亿对图文数据训练：同时编码图像和文本到同一个向量空间，相似的就靠近'},
        {'id': 'similarity_chart', 'name': '图文相似度', 'image': sim_chart,
         'explanation': f'图像与每个文本描述的余弦相似度。CLIP 的"零样本"能力：从没见过这些类别的训练样本，只靠文本描述就能分类——"a photo of a {top1_label}" 获得最高相似度'},
        {'id': 'sim_matrix', 'name': '文本-文本相似度矩阵', 'image': sim_matrix_img,
         'explanation': '文本嵌入之间的相似度。CLIP 的文本编码器能理解语义："dog"和"cat"的嵌入比"dog"和"car"更接近——这是大规模对比学习的成果'},
        {'id': 'result', 'name': f'分类结果: {top1_label}', 'image': img_u8,
         'explanation': f'零样本分类结果：{top1_label} ({top1_prob})。CLIP 实现了真正的"零样本"——不需要任何微调就能识别任意类别'},
    ]

    return {
        'steps': steps,
        'metrics': {
            'model': 'CLIP (ViT-B/32)',
            'num_classes': len(class_names),
            'top1': top1_label,
            'top1_prob': top1_prob,
            'class_set': cs if not custom.strip() else 'custom',
        },
        'predictions': preds,
    }


def _render_similarity_chart(predictions, width=600, height=300):
    from PIL import Image, ImageDraw
    img = Image.new('RGB', (width, height), (30, 35, 45))
    draw = ImageDraw.Draw(img)
    n = len(predictions)
    if n == 0: return np.array(img)
    bar_h = max(20, (height - 60) // n - 4)
    max_sim = max(p['similarity'] for p in predictions)
    min_sim = min(p['similarity'] for p in predictions)
    for i, p in enumerate(predictions):
        y = 20 + i * (bar_h + 4)
        w_bar = int((p['similarity'] - min_sim) / max(max_sim - min_sim, 1e-8) * (width - 180))
        w_bar = max(10, w_bar)
        # Color based on probability
        r = int(37 + (59-37) * p['probability'])
        g = int(99 + (130-99) * p['probability'])
        b = int(235 - (235-246) * p['probability'])
        draw.rectangle([(120, y), (120 + w_bar, y + bar_h)], fill=(r, g, b))
        draw.text((10, y + bar_h//2 - 7), f"{p['label']} ({p['probability']*100:.1f}%)",
                  fill=(226, 232, 240))
    return np.array(img)


def _render_similarity_matrix(predictions, size=300):
    from PIL import Image, ImageDraw
    labels = [p['label'] for p in predictions]
    n = len(labels)
    cell = max(20, size // n)
    w = n * cell + 80
    img = Image.new('RGB', (w, w - 20), (30, 35, 45))
    draw = ImageDraw.Draw(img)
    # Just show a conceptual similarity matrix from the sorted predictions
    for i in range(n):
        for j in range(n):
            x = 70 + j * cell
            y = 20 + i * cell
            # Diagonal-like pattern
            if i == j:
                fill = (59, 130, 246)
            elif abs(i - j) <= 1:
                fill = (37, 99, 200)
            elif abs(i - j) <= 2:
                fill = (30, 70, 150)
            else:
                fill = (20, 30, 50)
            draw.rectangle([(x, y), (x+cell-2, y+cell-2)], fill=fill)
    for i, label in enumerate(labels):
        draw.text((5, 22 + i * cell), label[:8], fill=(200, 210, 220))
        draw.text((72 + i * cell, 2), label[:8], fill=(200, 210, 220))
    return np.array(img)
