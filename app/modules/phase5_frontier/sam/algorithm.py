"""SAM (Segment Anything Model) algorithm — real inference with pretrained model.

Uses Meta's segment-anything package for prompt-based segmentation.
"""
import numpy as np
import torch
from PIL import Image


_MODEL = None
_DEVICE = None


def _get_model():
    global _MODEL, _DEVICE
    if _MODEL is None:
        _DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        from segment_anything import sam_model_registry, SamPredictor
        # Use ViT-B SAM model
        import os
        checkpoint = os.environ.get('SAM_CHECKPOINT',
                                     'models/sam_vit_b_01ec64.pth')
        if not os.path.exists(checkpoint):
            # Try alternate locations
            alt_paths = [
                'E:/SAM/sam_vit_b_01ec64.pth',
                'sam_vit_b_01ec64.pth',
            ]
            for p in alt_paths:
                if os.path.exists(p):
                    checkpoint = p
                    break
        sam = sam_model_registry['vit_b'](checkpoint=checkpoint)
        sam.to(device=_DEVICE)
        sam.eval()
        _MODEL = SamPredictor(sam)
    return _MODEL, _DEVICE


def load_image(image_path_or_array):
    if isinstance(image_path_or_array, str):
        img = Image.open(image_path_or_array).convert('RGB')
    elif isinstance(image_path_or_array, np.ndarray):
        img = Image.fromarray(image_path_or_array.astype(np.uint8))
    else:
        img = image_path_or_array
    return np.array(img)


def encode_image(image):
    """Encode image with SAM's image encoder (run once per image)."""
    predictor, device = _get_model()
    img_arr = load_image(image)
    predictor.set_image(img_arr)
    return {
        'image_shape': img_arr.shape,
        'original_size': predictor.original_size,
        'input_size': predictor.input_size,
    }


def predict_from_points(points, labels):
    """
    Predict mask from point prompts.
    points: list of (x, y) in original image coordinates
    labels: list of 1 (foreground) or 0 (background)
    """
    predictor, _ = _get_model()
    input_points = np.array(points)
    input_labels = np.array(labels)

    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True,
    )
    # masks: (3, H, W) - 3 candidate masks
    # scores: (3,) - IoU predictions
    return {
        'masks': [mask.astype(np.uint8) * 255 for mask in masks],
        'scores': [float(s) for s in scores],
        'best_idx': int(np.argmax(scores)),
    }


def predict_from_box(box):
    """
    Predict mask from bounding box prompt.
    box: (x1, y1, x2, y2) in original image coordinates
    """
    predictor, _ = _get_model()
    input_box = np.array([box])  # (1, 4)

    masks, scores, logits = predictor.predict(
        box=input_box,
        multimask_output=False,
    )
    return {
        'masks': [mask.astype(np.uint8) * 255 for mask in masks],
        'scores': [float(s) for s in scores],
        'best_idx': 0,
    }
