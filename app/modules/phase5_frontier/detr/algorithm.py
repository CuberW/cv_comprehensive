"""DETR (DEtection TRansformer) algorithm — real inference with pretrained model.

Uses HuggingFace transformers DetrForObjectDetection for end-to-end
object detection with Transformer encoder-decoder and object queries.
"""
import numpy as np
import torch
from PIL import Image


_MODEL = None
_PROCESSOR = None
_MODEL_NAME = "facebook/detr-resnet-50"


def _get_model():
    global _MODEL, _PROCESSOR
    if _MODEL is None:
        from transformers import DetrForObjectDetection, DetrImageProcessor
        _PROCESSOR = DetrImageProcessor.from_pretrained(_MODEL_NAME)
        _MODEL = DetrForObjectDetection.from_pretrained(
            _MODEL_NAME, output_attentions=True)
        _MODEL.eval()
    return _MODEL, _PROCESSOR


def load_image_as_array(image_path_or_array):
    if isinstance(image_path_or_array, str):
        img = Image.open(image_path_or_array).convert('RGB')
    elif isinstance(image_path_or_array, np.ndarray):
        img = Image.fromarray(image_path_or_array.astype(np.uint8))
    else:
        img = image_path_or_array
    return img


def detect_objects(image, threshold=0.5):
    """
    Run DETR object detection.

    Returns:
        detections with boxes, labels, scores
        decoder cross-attention maps
    """
    model, processor = _get_model()
    img = load_image_as_array(image)
    inputs = processor(images=img, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # Process detections
    target_sizes = torch.tensor([img.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=threshold)[0]

    detections = []
    for score, label, box in zip(results['scores'], results['labels'], results['boxes']):
        detections.append({
            'box': [round(float(v), 1) for v in box.tolist()],
            'label': model.config.id2label[int(label)],
            'class_id': int(label),
            'score': round(float(score), 3),
        })

    # Extract cross-attention from decoder (last layer, last decoder layer)
    # Shape: (batch, num_heads, num_queries, encoder_seq_len)
    cross_attentions = None
    if hasattr(outputs, 'cross_attentions') and outputs.cross_attentions:
        cross_attn = outputs.cross_attentions[-1][0].cpu().numpy()  # (heads, queries, seq_len)

        # Get feature map spatial size (encoder output after CNN backbone)
        encoder_seq_len = cross_attn.shape[-1]
        feat_size = int(np.sqrt(encoder_seq_len))

        cross_attentions = {
            'maps': cross_attn,
            'num_heads': cross_attn.shape[0],
            'num_queries': cross_attn.shape[1],
            'feature_size': feat_size,
        }

    return {
        'detections': detections,
        'cross_attentions': cross_attentions,
        'num_queries': 100,
    }


def visualize_query_attention(cross_attn, query_idx=0, head_idx=0):
    """Extract a single query's cross-attention heatmap."""
    if cross_attn is None:
        return None
    # cross_attn['maps']: (heads, queries, seq_len)
    attn = cross_attn['maps'][head_idx, query_idx]  # (seq_len,)
    feat_size = cross_attn['feature_size']
    heatmap = attn.reshape(feat_size, feat_size)
    hm_min, hm_max = heatmap.min(), heatmap.max()
    if hm_max - hm_min > 1e-8:
        heatmap = (heatmap - hm_min) / (hm_max - hm_min)
    return heatmap
