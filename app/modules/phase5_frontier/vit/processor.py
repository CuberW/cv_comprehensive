"""Pipeline builder for ViT module."""
import numpy as np
from app.utils.image_utils import load_image_u8
from app.modules.phase5_frontier.vit.algorithm import (
    extract_attention_maps, get_patch_grid,
    get_position_embedding_heatmap, attention_to_heatmap,
)


def build_pipeline(image_path=None, **kwargs):
    if not image_path:
        return {
            'error': 'ViT demo requires an uploaded image.',
            'steps': [],
            'metrics': {'status': 'missing_input'},
        }

    try:
        img_u8 = load_image_u8(image_path, mode='rgb', max_side=512)
        result = extract_attention_maps(img_u8)
    except Exception as exc:
        return {
            'error': f'ViT inference failed: {exc}',
            'steps': [],
            'metrics': {'status': 'model_not_available'},
        }

    patch_grid = get_patch_grid(img_u8)
    pos_sim = get_position_embedding_heatmap()
    pos_heatmap = _matrix_to_heatmap_image(pos_sim)
    num_layers = result['num_layers']
    grid_size = int(np.sqrt(result['num_patches']))
    shallow_attn = attention_to_heatmap(result['attentions'][0], head_idx=0, patch_grid_size=grid_size)
    deep_attn = attention_to_heatmap(result['attentions'][-1], head_idx=0, patch_grid_size=grid_size)
    shallow_hm = _heatmap_to_image(shallow_attn)
    deep_hm = _heatmap_to_image(deep_attn)
    attn_overlay = _overlay_attention_on_image(img_u8, deep_attn)
    preds = result['predictions']
    pred_text = '\n'.join([f"{p['label']}: {p['probability']*100:.1f}%" for p in preds[:5]])

    steps = [
        {'id': 'original', 'name': '输入图像', 'image': img_u8,
         'formula': 'I in R^{H x W x 3}',
         'explanation': '原始图像。ViT 将图像分割为 16×16 的 Patch，每个 Patch 像一个"视觉单词"送入 Transformer'},
        {'id': 'patches', 'name': 'Patch 切分 (16×16)', 'image': patch_grid,
         'formula': 'x_p^i = flatten(patch_i)',
         'explanation': f'图像被切分为 {grid_size}×{grid_size} = {result["num_patches"]} 个不重叠的 Patch。每个 Patch 通过线性投影变成一个向量(Token)，加上位置编码后送入 Transformer'},
        {'id': 'pos_embed', 'name': '位置编码相似度', 'image': pos_heatmap,
         'formula': 'z_i = E x_p^i + e_pos^i',
         'explanation': '位置编码之间的余弦相似度矩阵。相近位置的编码更相似——这帮助 Transformer 理解 Patch 之间的空间关系，因为自注意力本身是位置无关的'},
        {'id': 'shallow_attn', 'name': f'浅层注意力 (Layer 1)', 'image': shallow_hm,
         'formula': 'Attention(Q,K,V)=softmax(QK^T/sqrt(d))V',
         'explanation': f'第 1 层自注意力图 ({result["num_heads"]} 个头中的第 1 个)。浅层倾向于关注局部邻域——类似于 CNN 的浅层卷积核'},
        {'id': 'deep_attn', 'name': f'深层注意力 (Layer {num_layers})', 'image': deep_hm,
         'formula': 'A_l = softmax(Q_l K_l^T / sqrt(d))',
         'explanation': f'第 {num_layers} 层注意力图。深层逐渐关注语义相关的远距离区域——这是 ViT 替代 CNN 的关键能力：长程依赖建模'},
        {'id': 'overlay', 'name': '注意力叠加原图', 'image': attn_overlay,
         'formula': 'p(y|I)=softmax(head(CLS))',
         'explanation': '深层注意力热度叠加在原图上。亮区 = 模型最关注的区域。CLS Token 通过这些注意力聚合全局信息来做最终分类'},
    ]

    return {
        'steps': steps,
        'metrics': {
            'status': 'pretrained_model',
            'backend': 'transformers',
            'model': 'ViT-B/16',
            'layers': num_layers,
            'heads': result['num_heads'],
            'patches': result['num_patches'],
            'top1': preds[0]['label'] if preds else 'N/A',
            'top1_prob': f"{preds[0]['probability']*100:.1f}%" if preds else 'N/A',
        },
        'predictions': preds,
        'pred_text': pred_text,
    }


def _matrix_to_heatmap_image(matrix):
    """Convert a 2D similarity matrix to a color heatmap image."""
    m = np.asarray(matrix, dtype=np.float64)
    m = np.clip(m, 0, 1)
    h, w = m.shape
    # Upsample for visibility
    scale = max(1, 400 // max(h, w))
    big = np.kron(m, np.ones((scale, scale)))
    # RGB heatmap: blue=cold, red=hot
    r = (big * 255).astype(np.uint8)
    g = ((1 - np.abs(big - 0.5) * 2) * 200).astype(np.uint8)
    b = ((1 - big) * 255).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def _heatmap_to_image(heatmap_2d):
    """Convert a 2D attention heatmap to a color image."""
    hm = np.asarray(heatmap_2d, dtype=np.float64)
    hm = np.clip(hm, 0, 1)
    h, w = hm.shape
    scale = max(1, 400 // max(h, w))
    big = np.kron(hm, np.ones((scale, scale)))
    r = (big * 255).astype(np.uint8)
    g = ((1 - np.abs(big - 0.5) * 2) * 200).astype(np.uint8)
    b = ((1 - big) * 255).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def _overlay_attention_on_image(img, heatmap_2d):
    """Overlay attention heatmap on the original image."""
    from PIL import Image
    h, w = img.shape[:2]
    hm = np.asarray(heatmap_2d, dtype=np.float64)
    hm = np.clip(hm, 0, 1)
    # Upsample heatmap to image size
    hm_img = np.array(Image.fromarray((hm * 255).astype(np.uint8)).resize((w, h), Image.LANCZOS))
    # Red-yellow heat overlay
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    hm_f = hm_img.astype(np.float32) / 255.0
    overlay[:,:,0] = (hm_f * 255).astype(np.uint8)
    overlay[:,:,1] = (hm_f * 100).astype(np.uint8)
    alpha = 0.5
    return np.clip(img.astype(np.float32)*(1-alpha) + overlay.astype(np.float32)*alpha, 0, 255).astype(np.uint8)
