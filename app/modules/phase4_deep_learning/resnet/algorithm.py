"""ResNet-50 inference + Grad-CAM heatmap visualization."""
import numpy as np
import os
from app.utils.image_utils import load_image_u8

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
_MODEL = None
_DEVICE = None

def _get_model():
    global _MODEL, _DEVICE
    if _MODEL is None:
        import torch
        from torchvision.models import resnet50, ResNet50_Weights
        _DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _MODEL = resnet50(weights=ResNet50_Weights.DEFAULT)
        _MODEL.to(_DEVICE)
        _MODEL.eval()
    return _MODEL, _DEVICE


def build_pipeline(image_path=None, **kwargs):
    try:
        import torch
        from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
        from torchvision.models import ResNet50_Weights
    except ImportError:
        return _error('PyTorch/torchvision 未安装', np.zeros((100,400,3),dtype=np.uint8)+240)

    if not image_path:
        import os as _os
        demo = _os.path.join(_PROJECT_ROOT, 'bus.jpg')
        image_path = demo if _os.path.exists(demo) else None
    # Full-size for display; model gets 224×224 separately
    display_img = load_image_u8(image_path, mode='rgb') if image_path else (np.ones((224,224,3),dtype=np.uint8)*128)
    img_u8 = load_image_u8(image_path, mode='rgb', max_side=256) if image_path else display_img

    try:
        model, device = _get_model()
    except Exception as e:
        return _error(f'模型加载失败: {e}', img_u8)

    # --- Preprocess ---
    preprocess = Compose([Resize(256), CenterCrop(224), ToTensor(),
                          Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
    from PIL import Image
    pil_img = Image.fromarray(img_u8)  # 224×224 for model
    input_tensor = preprocess(pil_img).unsqueeze(0).to(device)

    # --- Classification ---
    model.zero_grad()
    input_tensor.requires_grad_(True)
    logits = model(input_tensor)
    probs = torch.softmax(logits, dim=1)[0]
    top5_prob, top5_idx = torch.topk(probs, 5)

    # --- Load class names ---
    classes_path = os.path.join(_PROJECT_ROOT, 'static', 'data', 'imagenet_classes.txt')
    try:
        with open(classes_path, encoding='utf-8') as f:
            classes = [l.strip() for l in f.readlines()]
    except FileNotFoundError:
        # Fallback: use torchvision metadata
        classes = ResNet50_Weights.DEFAULT.meta.get('categories', [str(i) for i in range(1000)])

    top5 = [(classes[idx], float(p)) for idx, p in zip(top5_idx.tolist(), top5_prob.tolist())]

    # --- Grad-CAM ---
    # Hook the last conv layer (layer4)
    target_layer = model.layer4[-1].conv3  # last conv in ResNet-50
    activations = {}
    gradients = {}

    def forward_hook(module, inp, out):
        activations['value'] = out

    def backward_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0]

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    # Forward again to get activations (need fresh forward for hook to fire)
    _ = model(input_tensor)
    # Backward on the top-1 class
    top_class = top5_idx[0]
    model.zero_grad()
    logits = model(input_tensor)
    score = logits[0, top_class]
    score.backward()

    fh.remove(); bh.remove()

    # Compute Grad-CAM heatmap
    act = activations['value'].detach()[0]       # (C, H, W)
    grad = gradients['value'].detach()[0]         # (C, H, W)
    weights = grad.mean(dim=(1,2), keepdim=True)  # (C, 1, 1)
    cam = (weights * act).sum(dim=0)               # (H, W)
    cam = torch.relu(cam)
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    cam = cam.cpu().numpy()

    # Resize CAM to match input image
    from PIL import Image as PILImage
    # Resize CAM to match display image size (not 224×224)
    cam_img = PILImage.fromarray((cam * 255).astype(np.uint8)).resize(
        (display_img.shape[1], display_img.shape[0]), PILImage.BILINEAR)
    cam_arr = np.array(cam_img, dtype=np.float32) / 255.0

    # Heatmap overlay on full-size display image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.cm as cm
    heatmap = cm.jet(cam_arr)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    overlay = (display_img.astype(np.float32) * 0.5 + heatmap.astype(np.float32) * 0.5).clip(0, 255).astype(np.uint8)

    # --- Result label overlay ---
    from PIL import ImageDraw
    result_img = PILImage.fromarray(display_img.copy())
    draw = ImageDraw.Draw(result_img)
    for i, (name, prob) in enumerate(top5):
        draw.text((8, 8 + i*20), f'{i+1}. {name[:30]}: {prob:.3f}', fill=(34, 197, 94))

    return {
        'steps': [
            {'id': 'input', 'name': '输入图像', 'image': display_img,
             'explanation': '上传的原图。模型实际处理的是缩放+裁剪后的224×224版本。'},
            {'id': 'predictions', 'name': f'Top-5 预测 (Top-1: {top5[0][0]})', 'image': np.array(result_img),
             'explanation': f'ImageNet-1K 1000类分类。Top-1: {top5[0][0]} ({top5[0][1]:.3f})。'},
            {'id': 'gradcam', 'name': 'Grad-CAM 热力图', 'image': overlay,
             'explanation': f'模型判断"{top5[0][0]}"时关注的区域（红=高关注，蓝=低关注），已放大到原图尺寸。'},
            {'id': 'heatmap', 'name': '原始热力值', 'image': heatmap,
             'explanation': 'CAM 激活值双线性插值到原图尺寸。'},
        ],
        'metrics': {
            'status': 'pretrained_model', 'model': 'resnet50', 'backend': 'torchvision',
            'device': str(device), 'top1': top5[0][0], 'top1_conf': round(top5[0][1], 4),
            'top5': ', '.join(f'{n}({p:.2f})' for n, p in top5),
        }
    }


def _error(message, img):
    return {
        'steps': [{'id': 'error', 'name': '错误', 'image': img, 'explanation': message}],
        'metrics': {'status': 'error', 'error': message}
    }
