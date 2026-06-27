"""Pipeline builder for real CLIP image-text inference."""
from __future__ import annotations

import numpy as np

from app.modules.phase5_frontier.clip.algorithm import PRESET_CLASS_SETS, zero_shot_classify
from app.utils.image_utils import load_image_u8


def build_pipeline(image_path=None, class_set='animals', custom_classes='', **kwargs):
    cs = kwargs.get('class_set', class_set)
    custom = kwargs.get('custom_classes', custom_classes)

    if not image_path:
        return {
            'error': 'CLIP needs an input image so the real pretrained model can run.',
            'steps': [],
            'metrics': {'status': 'missing_input'},
        }

    try:
        img_u8 = load_image_u8(image_path, mode='rgb', max_side=512)
        if custom.strip():
            class_names = [n.strip() for n in custom.split(',') if n.strip()]
        else:
            class_names = PRESET_CLASS_SETS.get(cs, PRESET_CLASS_SETS['animals'])
        result = zero_shot_classify(img_u8, class_names)
    except Exception as exc:
        return {
            'error': f'CLIP inference failed: {exc}',
            'steps': [],
            'metrics': {'status': 'model_not_available', 'error_type': type(exc).__name__},
        }

    preds = result['predictions']
    prompts = [f'a photo of a {name}' for name in class_names]
    prompt_img = _render_prompt_list(prompts)
    embedding_img = _render_embedding_summary(result.get('image_embedding'), result.get('text_embeddings'), class_names)
    sim_chart = _render_similarity_chart(preds)
    sim_matrix_img = _render_similarity_matrix(preds, result.get('text_similarity'))
    top1_label = preds[0]['label'] if preds else 'N/A'
    top1_prob = f"{preds[0]['probability'] * 100:.1f}%" if preds else 'N/A'

    return {
        'steps': [
            {
                'id': 'input',
                'name': '输入图像',
                'image': img_u8,
                'formula': 'v_I = ImageEncoder(I)',
                'explanation': 'CLIP 用图像编码器把整张图片变成单位向量。后续所有判断都来自这个图像向量与文本向量的真实余弦相似度。',
                'data': {'height': int(img_u8.shape[0]), 'width': int(img_u8.shape[1])},
            },
            {
                'id': 'text_prompts',
                'name': '候选文本提示',
                'image': prompt_img,
                'formula': 'T_i = "a photo of a {class_i}"',
                'explanation': '零样本分类不是重新训练分类头，而是把类别名写成自然语言提示，再交给 CLIP 的文本编码器。',
                'data': {'prompts': prompts},
            },
            {
                'id': 'embeddings',
                'name': '图像/文本 embedding',
                'image': embedding_img,
                'formula': 'v = f(x) / ||f(x)||',
                'explanation': '图像向量和文本向量都被归一化到同一个语义空间。图中不是手画示意，而是后端从真实 CLIP embedding 中抽样出来的数值条纹。',
            },
            {
                'id': 'similarity_chart',
                'name': '图文相似度',
                'image': sim_chart,
                'formula': 'sim(I,T_i)=<v_I,v_T_i>',
                'explanation': f'每个文本提示都与图像向量做余弦相似度。当前候选集中，"{top1_label}" 的相似度最高。',
                'data': {'predictions': preds},
            },
            {
                'id': 'text_similarity',
                'name': '文本 embedding 相似度矩阵',
                'image': sim_matrix_img,
                'formula': 'S_ij=<v_T_i,v_T_j>',
                'explanation': '这张矩阵说明候选文本之间本身也有语义距离，例如 car 和 bicycle 可能比 car 和 bird 更接近。',
            },
            {
                'id': 'softmax_result',
                'name': f'零样本分类结果: {top1_label}',
                'image': sim_chart,
                'formula': 'p_i = softmax(100 * sim(I,T_i))',
                'explanation': f'最终概率是在当前候选文本集合内归一化得到的，不代表开放世界绝对概率。Top-1 为 {top1_label}，概率 {top1_prob}。',
            },
        ],
        'metrics': {
            'status': 'pretrained_model',
            'backend': 'transformers',
            'model': 'openai/clip-vit-base-patch32',
            'num_classes': len(class_names),
            'top1': top1_label,
            'top1_prob': top1_prob,
            'class_set': cs if not custom.strip() else 'custom',
        },
        'predictions': preds,
    }


def _render_prompt_list(prompts, width=640, height=260):
    from PIL import Image, ImageDraw

    img = Image.new('RGB', (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(img)
    draw.text((18, 12), 'Text prompts sent into CLIP text encoder', fill=(15, 23, 42))
    for i, prompt in enumerate(prompts[:9]):
        y = 48 + i * 22
        draw.rectangle((18, y - 3, width - 18, y + 17), fill=(241, 245, 249), outline=(203, 213, 225))
        draw.text((28, y), f'{i + 1}. {prompt}', fill=(51, 65, 85))
    return np.array(img)


def _render_embedding_summary(image_embedding, text_embeddings, labels, width=640, height=300):
    from PIL import Image, ImageDraw

    img = Image.new('RGB', (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(img)
    draw.text((18, 12), 'Sampled normalized embedding values', fill=(15, 23, 42))
    image_embedding = np.asarray(image_embedding or [], dtype=np.float64)
    text_embeddings = np.asarray(text_embeddings or [], dtype=np.float64)
    if image_embedding.size == 0 or text_embeddings.size == 0:
        draw.text((18, 60), 'No embedding data returned.', fill=(100, 116, 139))
        return np.array(img)

    sample_idx = np.linspace(0, image_embedding.size - 1, 48).astype(int)
    rows = [('image', image_embedding[sample_idx])]
    for label, emb in zip(labels[:5], text_embeddings[:5]):
        rows.append((label[:12], emb[sample_idx]))

    x0, y0, cell_w, cell_h = 100, 48, 9, 22
    for r, (label, values) in enumerate(rows):
        y = y0 + r * (cell_h + 7)
        draw.text((18, y + 3), label, fill=(51, 65, 85))
        vmax = max(float(np.max(np.abs(values))), 1e-8)
        for c, v in enumerate(values):
            t = float(v / vmax)
            if t >= 0:
                fill = (37, int(110 + 100 * t), 235)
            else:
                fill = (225, int(120 + 80 * (1 + t)), 72)
            draw.rectangle((x0 + c * cell_w, y, x0 + c * cell_w + cell_w - 1, y + cell_h), fill=fill)
    return np.array(img)


def _render_similarity_chart(predictions, width=640, height=320):
    from PIL import Image, ImageDraw

    img = Image.new('RGB', (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(img)
    draw.text((18, 12), 'Image-text cosine similarity and normalized probability', fill=(15, 23, 42))
    if not predictions:
        return np.array(img)
    max_sim = max(p['similarity'] for p in predictions)
    min_sim = min(p['similarity'] for p in predictions)
    bar_h = max(20, (height - 72) // len(predictions) - 5)
    for i, p in enumerate(predictions):
        y = 48 + i * (bar_h + 5)
        w_bar = int((p['similarity'] - min_sim) / max(max_sim - min_sim, 1e-8) * (width - 240))
        w_bar = max(8, w_bar)
        intensity = float(p['probability'])
        fill = (37, int(99 + 95 * intensity), 235)
        draw.text((18, y + 3), f"{p['label'][:14]} {p['probability']*100:.1f}%", fill=(51, 65, 85))
        draw.rectangle((160, y, 160 + w_bar, y + bar_h), fill=fill)
        draw.text((170 + w_bar, y + 3), f"sim {p['similarity']:.3f}", fill=(71, 85, 105))
    return np.array(img)


def _render_similarity_matrix(predictions, matrix=None, size=340):
    from PIL import Image, ImageDraw

    labels = [p['label'] for p in predictions]
    n = len(labels)
    if n == 0:
        return np.zeros((size, size, 3), dtype=np.uint8) + 248
    matrix = np.eye(n, dtype=np.float64) if matrix is None else np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (n, n):
        matrix = np.eye(n, dtype=np.float64)
    mn, mx = float(matrix.min()), float(matrix.max())
    norm = (matrix - mn) / max(mx - mn, 1e-8)
    cell = max(24, size // n)
    width = n * cell + 96
    height = n * cell + 52
    img = Image.new('RGB', (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(img)
    draw.text((12, 8), 'Text-text similarity', fill=(15, 23, 42))
    for i in range(n):
        for j in range(n):
            x = 86 + j * cell
            y = 40 + i * cell
            v = float(norm[i, j])
            fill = (int(30 + 210 * v), int(80 + 95 * v), int(220 - 130 * v))
            draw.rectangle((x, y, x + cell - 2, y + cell - 2), fill=fill)
    for i, label in enumerate(labels):
        draw.text((8, 44 + i * cell), label[:10], fill=(51, 65, 85))
        draw.text((88 + i * cell, 25), label[:5], fill=(51, 65, 85))
    return np.array(img)
