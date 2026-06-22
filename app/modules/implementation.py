"""Implementation truth table for frontend/API reporting.

This file is intentionally boring: it says what each registered module really
runs today, so the UI does not imply that an educational visualization is a
pretrained model.
"""

DEFAULT_IMPLEMENTATION = {
    'status': '真实 NumPy 算法',
    'category': 'numpy_algorithm',
    'backend': 'NumPy',
    'local_inference': True,
    'real_model': False,
    'requires_upload': True,
    'model': '',
    'note': '项目内手写算法实现，返回真实中间结果。',
}

REQUIRES_EXTERNAL_WEIGHTS = {
    'vit', 'detr', 'clip', 'sam', 'stable_diffusion',
    'swin', 'dino', 'mae', 'dino_det', 'grdino', 'mask2former', 'sam2',
    'blip2', 'controlnet', 'dit', 'flux', 'stylegan', 'dust3r',
    'orbslam3', 'mediapipe', 'vitpose',
}

OFFLINE_TEACHING = set()  # 已废弃——所有模块必须有真实实现

EXTERNAL_WEIGHT_META = {
    'status': 'requires external weights',
    'category': 'requires_external_weights',
    'backend': 'external pretrained model',
    'local_inference': False,
    'real_model': True,
    'requires_upload': False,
    'model': '',
    'note': 'This algorithm requires local pretrained weights or a remote model. It is intentionally disabled in offline mode.',
}

OFFLINE_TEACHING_META = {
    'status': 'offline teaching demo',
    'category': 'teaching_simulation',
    'backend': 'NumPy/PIL',
    'local_inference': True,
    'real_model': False,
    'requires_upload': False,
    'model': '',
    'note': 'Offline teaching visualization; not a pretrained-model inference result.',
}

IMPLEMENTATION_META = {
    'gan': {
        'status': '真实 NumPy 算法',
        'category': 'numpy_algorithm',
        'backend': 'NumPy DCGAN',
        'local_inference': True,
        'real_model': False,
        'requires_upload': False,
        'model': 'DCGAN random-weight forward pass',
        'note': '四层转置卷积DCGAN生成器前向传播。权重随机初始化，展示真实网络架构。',
    },
    'diffusion': {
        'status': '真实预训练模型',
        'category': 'pretrained_model',
        'backend': 'PyTorch + diffusers',
        'local_inference': True,
        'real_model': True,
        'requires_upload': False,
        'model': 'runwayml/stable-diffusion-v1-5',
        'note': '通过 diffusers 加载 Stable Diffusion v1.5，不再使用前向加噪模拟。',
    },
    'detection': {
        'status': '真实预训练模型',
        'category': 'pretrained_model',
        'backend': 'torchvision',
        'local_inference': True,
        'real_model': True,
        'requires_upload': True,
        'model': 'fasterrcnn_resnet50_fpn',
        'note': '通过 torchvision 加载 Faster R-CNN COCO 权重；torch 不可用时自动降级到本地教学检测。',
    },
    'semantic': {
        'status': '真实预训练模型',
        'category': 'pretrained_model',
        'backend': 'torchvision',
        'local_inference': True,
        'real_model': True,
        'requires_upload': True,
        'model': 'fcn_resnet50',
        'note': '通过 torchvision 加载 FCN ResNet50 COCO 权重；torch 不可用时自动降级到本地教学分割。',
    },
    'instance': {
        'status': '真实预训练模型',
        'category': 'pretrained_model',
        'backend': 'torchvision',
        'local_inference': True,
        'real_model': True,
        'requires_upload': True,
        'model': 'maskrcnn_resnet50_fpn',
        'note': '通过 torchvision 加载 Mask R-CNN COCO 权重；torch 不可用时自动降级到本地教学分割。',
    },


    # Actual model-loading pages.
    'vit': {
        'status': '真实预训练模型',
        'category': 'pretrained_model',
        'backend': 'PyTorch + transformers',
        'local_inference': True,
        'real_model': True,
        'requires_upload': True,
        'model': 'google/vit-base-patch16-224',
        'note': '通过 HuggingFace 加载 ViT 分类模型并提取真实注意力。',
    },
    'detr': {
        'status': '真实预训练模型',
        'category': 'pretrained_model',
        'backend': 'PyTorch + transformers',
        'local_inference': True,
        'real_model': True,
        'requires_upload': True,
        'model': 'facebook/detr-resnet-50',
        'note': '通过 HuggingFace 加载 DETR 并运行真实目标检测。',
    },
    'clip': {
        'status': '真实预训练模型',
        'category': 'pretrained_model',
        'backend': 'PyTorch + transformers',
        'local_inference': True,
        'real_model': True,
        'requires_upload': True,
        'model': 'openai/clip-vit-base-patch32',
        'note': '通过 HuggingFace 加载 CLIP 并运行零样本图文相似度。',
    },
    'sam': {
        'status': '真实预训练模型',
        'category': 'pretrained_model',
        'backend': 'PyTorch + segment-anything',
        'local_inference': True,
        'real_model': True,
        'requires_upload': True,
        'model': 'SAM ViT-B',
        'weights': 'models/sam_vit_b_01ec64.pth 或 SAM_CHECKPOINT',
        'note': '需要本地 SAM checkpoint；缺失时 API 会明确报告 model_not_available。',
    },
    'stable_diffusion': {
        'status': '真实预训练模型',
        'category': 'pretrained_model',
        'backend': 'PyTorch + diffusers',
        'local_inference': True,
        'real_model': True,
        'requires_upload': False,
        'model': 'runwayml/stable-diffusion-v1-5',
        'note': '通过 diffusers 加载 Stable Diffusion v1.5 生成图像。',
    },
    'nerf': {
        'status': '真实 NumPy 算法',
        'category': 'numpy_algorithm',
        'backend': 'PyTorch + NumPy',
        'local_inference': True,
        'real_model': False,
        'requires_upload': False,
        'model': 'TinyNeRF Ray Casting',
        'note': '真实射线追踪 + 体渲染。TinyNeRF MLP前向传播预测RGB和密度。',
    },
    'lenet': {
        'status': '真实 NumPy 网络',
        'category': 'numpy_model',
        'backend': 'NumPy',
        'local_inference': True,
        'real_model': True,
        'requires_upload': True,
        'model': 'LeNet-5 + lenet_weights.json',
        'note': '加载本地 LeNet 权重后运行真实前向传播；权重缺失时会返回错误。',
    },
    'conv_training': {
        'status': '真实 NumPy 算法',
        'category': 'numpy_algorithm',
        'backend': 'NumPy',
        'local_inference': True,
        'real_model': False,
        'requires_upload': False,
        'model': 'Kernel gradient descent training',
        'note': '真实梯度下降训练卷积核。损失曲线和参数更新来自实际优化过程。',
    },
}


def get_implementation_meta(module_id):
    """Return public implementation metadata for a module id."""
    meta = dict(DEFAULT_IMPLEMENTATION)
    if module_id in REQUIRES_EXTERNAL_WEIGHTS:
        meta.update(EXTERNAL_WEIGHT_META)
    if module_id in OFFLINE_TEACHING:
        meta.update(OFFLINE_TEACHING_META)
    meta.update(IMPLEMENTATION_META.get(module_id, {}))
    if module_id in REQUIRES_EXTERNAL_WEIGHTS:
        # Keep plan-selected offline behavior even if older per-module metadata
        # says a pretrained model exists locally.
        meta.update(EXTERNAL_WEIGHT_META)
    if module_id in OFFLINE_TEACHING:
        # Keep local teaching demos runnable even if older metadata marked them
        # as placeholders.
        meta.update(OFFLINE_TEACHING_META)
    if module_id == 'sam' and module_id not in REQUIRES_EXTERNAL_WEIGHTS:
        import os
        checkpoint = os.environ.get('SAM_CHECKPOINT', 'models/sam_vit_b_01ec64.pth')
        candidates = [checkpoint, 'E:/SAM/sam_vit_b_01ec64.pth', 'sam_vit_b_01ec64.pth']
        if not any(os.path.exists(p) for p in candidates):
            meta.update({
                'status': 'SAM checkpoint not configured',
                'category': 'model_not_available',
                'local_inference': False,
                'note': 'Set SAM_CHECKPOINT or place sam_vit_b_01ec64.pth under models/ before this module is counted as runnable.',
            })

    # Remote runners are intentionally ignored for external-weight modules in
    # offline mode so UI/API behavior stays deterministic and never depends on
    # tokens or network availability.
    if module_id not in REQUIRES_EXTERNAL_WEIGHTS and module_id not in {'detection', 'semantic', 'instance'}:
        try:
            from app.runners import get_remote_runner
            runner = get_remote_runner(module_id)
            if runner is not None:
                meta.update(runner.get_metadata())
        except Exception:
            pass

    return meta


def requires_upload(module_id):
    """Return whether the demo endpoint needs an uploaded image."""
    return bool(get_implementation_meta(module_id).get('requires_upload', True))
