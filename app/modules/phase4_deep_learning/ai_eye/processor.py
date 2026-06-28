"""Unified AI Eye pretrained vision pipeline.

The model forward pass is torchvision; image processing, overlays, charts and
result packaging stay in local NumPy/Pillow code.  No OpenCV is used.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from app.modules.offline_teaching import _load_or_fixture
from app.utils.image_utils import ensure_gray, load_image_u8, to_base64


_MODEL_CACHE: dict[str, tuple[Any, Any, Any]] = {}
_TORCH_AVAILABLE: bool | None = None


@dataclass(frozen=True)
class ModelSpec:
    id: str
    task: str
    name: str
    family: str
    builder_module: str
    builder_name: str
    weights_name: str
    default: bool = False
    size_mb: int | None = None
    source: str = 'torchvision official pretrained weights'
    license: str = 'See torchvision model documentation and source dataset terms.'
    description: str = ''
    strengths: str = ''
    limitations: str = ''


AI_EYE_IMPLEMENTATION = {
    'status': '真实预训练模型',
    'category': 'pretrained_model',
    'backend': 'torchvision',
    'local_inference': True,
    'real_model': True,
    'requires_upload': True,
    'model': 'torchvision detection / segmentation models',
    'note': 'AI之眼主结果来自 torchvision 官方预训练权重；权重缺失时返回下载建议，不用本地启发式结果冒充成功。',
}


MODEL_SPECS: dict[str, ModelSpec] = {
    'fasterrcnn_resnet50_fpn': ModelSpec(
        id='fasterrcnn_resnet50_fpn',
        task='detection',
        name='Faster R-CNN ResNet50-FPN',
        family='two_stage_detector',
        builder_module='torchvision.models.detection',
        builder_name='fasterrcnn_resnet50_fpn',
        weights_name='FasterRCNN_ResNet50_FPN_Weights',
        default=True,
        size_mb=160,
        description='两阶段检测器：先生成候选区域，再分类并回归边界框。',
        strengths='定位稳、教学结构清楚，适合解释 RPN、RoI 和分类回归。',
        limitations='CPU 上较慢，只输出矩形框，不能给出精细像素边界。',
    ),
    'fasterrcnn_mobilenet_v3_large_fpn': ModelSpec(
        id='fasterrcnn_mobilenet_v3_large_fpn',
        task='detection',
        name='Faster R-CNN MobileNetV3-FPN',
        family='light_two_stage_detector',
        builder_module='torchvision.models.detection',
        builder_name='fasterrcnn_mobilenet_v3_large_fpn',
        weights_name='FasterRCNN_MobileNet_V3_Large_FPN_Weights',
        size_mb=75,
        description='轻量骨干网络版本 Faster R-CNN，保留两阶段检测流程。',
        strengths='比 ResNet50 版本更轻，适合普通电脑演示。',
        limitations='精度通常低于更重的 ResNet/FPN 检测器。',
    ),
    'retinanet_resnet50_fpn': ModelSpec(
        id='retinanet_resnet50_fpn',
        task='detection',
        name='RetinaNet ResNet50-FPN',
        family='one_stage_detector',
        builder_module='torchvision.models.detection',
        builder_name='retinanet_resnet50_fpn',
        weights_name='RetinaNet_ResNet50_FPN_Weights',
        size_mb=130,
        description='一阶段检测器，用 Focal Loss 缓解前景/背景样本极不均衡。',
        strengths='结构比两阶段检测更直接，适合讲密集预测和 Focal Loss。',
        limitations='仍需 NMS；小目标和密集场景效果依赖训练数据。',
    ),
    'fcos_resnet50_fpn': ModelSpec(
        id='fcos_resnet50_fpn',
        task='detection',
        name='FCOS ResNet50-FPN',
        family='anchor_free_detector',
        builder_module='torchvision.models.detection',
        builder_name='fcos_resnet50_fpn',
        weights_name='FCOS_ResNet50_FPN_Weights',
        size_mb=125,
        description='无 anchor 一阶段检测器，直接预测像素点到框四边距离。',
        strengths='适合解释 anchor-free 检测思想。',
        limitations='框质量依然需要分数过滤与 NMS 后处理。',
    ),
    'fcn_resnet50': ModelSpec(
        id='fcn_resnet50',
        task='semantic',
        name='FCN ResNet50',
        family='fully_convolutional_segmentation',
        builder_module='torchvision.models.segmentation',
        builder_name='fcn_resnet50',
        weights_name='FCN_ResNet50_Weights',
        size_mb=135,
        description='全卷积语义分割经典结构，把分类网络改成像素级分类网络。',
        strengths='概念朴素，适合讲 dense prediction 和逐像素 softmax。',
        limitations='边界细节和多尺度上下文不如更新结构。',
    ),
    'deeplabv3_resnet50': ModelSpec(
        id='deeplabv3_resnet50',
        task='semantic',
        name='DeepLabV3 ResNet50',
        family='atrous_context_segmentation',
        builder_module='torchvision.models.segmentation',
        builder_name='deeplabv3_resnet50',
        weights_name='DeepLabV3_ResNet50_Weights',
        default=True,
        size_mb=160,
        description='使用空洞卷积和 ASPP 多尺度上下文的主流语义分割模型。',
        strengths='上下文建模更强，适合讲空洞卷积和多尺度感受野。',
        limitations='只区分类别，不区分同类中的不同个体。',
    ),
    'deeplabv3_mobilenet_v3_large': ModelSpec(
        id='deeplabv3_mobilenet_v3_large',
        task='semantic',
        name='DeepLabV3 MobileNetV3',
        family='light_atrous_segmentation',
        builder_module='torchvision.models.segmentation',
        builder_name='deeplabv3_mobilenet_v3_large',
        weights_name='DeepLabV3_MobileNet_V3_Large_Weights',
        size_mb=45,
        description='轻量版 DeepLabV3，用 MobileNetV3 做特征骨干。',
        strengths='权重小、CPU 演示更友好。',
        limitations='类别边界和困难场景通常不如更重模型。',
    ),
    'lraspp_mobilenet_v3_large': ModelSpec(
        id='lraspp_mobilenet_v3_large',
        task='semantic',
        name='LR-ASPP MobileNetV3',
        family='mobile_segmentation',
        builder_module='torchvision.models.segmentation',
        builder_name='lraspp_mobilenet_v3_large',
        weights_name='LRASPP_MobileNet_V3_Large_Weights',
        size_mb=13,
        description='移动端友好的轻量语义分割模型。',
        strengths='非常轻，适合低配置设备体验语义分割。',
        limitations='表达能力有限，复杂场景下类别图更粗糙。',
    ),
    'maskrcnn_resnet50_fpn': ModelSpec(
        id='maskrcnn_resnet50_fpn',
        task='instance',
        name='Mask R-CNN ResNet50-FPN',
        family='two_stage_instance_segmentation',
        builder_module='torchvision.models.detection',
        builder_name='maskrcnn_resnet50_fpn',
        weights_name='MaskRCNN_ResNet50_FPN_Weights',
        default=True,
        size_mb=170,
        description='在 Faster R-CNN 上增加 mask 分支，输出每个实例的独立掩码。',
        strengths='最适合讲“检测框 + 每个框内 mask”的实例分割范式。',
        limitations='CPU 推理较慢，mask 分辨率和类别由预训练数据决定。',
    ),
    'maskrcnn_resnet50_fpn_v2': ModelSpec(
        id='maskrcnn_resnet50_fpn_v2',
        task='instance',
        name='Mask R-CNN ResNet50-FPN V2',
        family='two_stage_instance_segmentation',
        builder_module='torchvision.models.detection',
        builder_name='maskrcnn_resnet50_fpn_v2',
        weights_name='MaskRCNN_ResNet50_FPN_V2_Weights',
        size_mb=175,
        description='改进训练配方的 Mask R-CNN V2 权重。',
        strengths='通常比 V1 权重更稳，可作为实例分割增强选项。',
        limitations='依旧是较重模型，不适合无 GPU 的批量高频调用。',
    ),
}


TASK_DEFAULTS = {
    'detection': 'fasterrcnn_resnet50_fpn',
    'semantic': 'deeplabv3_resnet50',
    'instance': 'maskrcnn_resnet50_fpn',
}


TASK_LABELS = {
    'detection': '目标检测',
    'semantic': '语义分割',
    'instance': '实例分割',
}


def list_models() -> dict[str, Any]:
    """Return model manifest with local cache status."""
    cache_dir = _torch_cache_dir()
    models = {}
    for model_id, spec in MODEL_SPECS.items():
        weights = _weights_enum(spec, allow_import_error=True)
        filename = _weight_filename(weights)
        cached = bool(filename and os.path.exists(os.path.join(cache_dir, filename)))
        models[model_id] = {
            'id': spec.id,
            'task': spec.task,
            'name': spec.name,
            'family': spec.family,
            'default': spec.default,
            'size_mb': spec.size_mb,
            'source': spec.source,
            'license': spec.license,
            'description': spec.description,
            'strengths': spec.strengths,
            'limitations': spec.limitations,
            'weights': spec.weights_name + '.DEFAULT',
            'cache_dir': cache_dir,
            'cache_file': filename,
            'cached': cached,
            'download_command': f'python prepare_ai_eye_assets.py --model {spec.id}',
        }
    return {
        'implementation': AI_EYE_IMPLEMENTATION,
        'cache_dir': cache_dir,
        'defaults': dict(TASK_DEFAULTS),
        'tasks': {
            task: [m.id for m in MODEL_SPECS.values() if m.task == task]
            for task in TASK_DEFAULTS
        },
        'models': models,
        'notes': [
            '权重使用 torchvision 官方下载机制，默认存入 torch hub checkpoints 缓存。',
            '仓库不提交大权重；公开平台请运行 prepare_ai_eye_assets.py 或按页面提示下载。',
        ],
    }


def build_pipeline(
    image_path=None,
    task='all',
    model=None,
    score_threshold=0.5,
    mask_threshold=0.5,
    **kwargs,
):
    """Run one or all AI Eye pretrained tasks."""
    task = str(kwargs.get('task', task or 'all')).strip().lower()
    if task not in {'all', 'detection', 'semantic', 'instance'}:
        return _error_result(f'Unknown AI Eye task: {task}', 'bad_task', task=task)

    img_u8 = _load_ai_eye_image(image_path)
    score_th = float(kwargs.get('score_threshold', kwargs.get('threshold', score_threshold)))
    mask_th = float(kwargs.get('mask_threshold', mask_threshold))
    selected_model = kwargs.get('model', model)
    tasks = ['detection', 'semantic', 'instance'] if task == 'all' else [task]

    algorithms = {}
    flat_steps = [{'id': 'shared_input', 'name': '统一输入图像', 'image': img_u8,
                   'formula': 'I in R^{H x W x 3}',
                   'explanation': '同一张输入图像会被送入目标检测、语义分割和实例分割模型，方便比较三种视觉任务的输出差异。',
                   'data': {'height': int(img_u8.shape[0]), 'width': int(img_u8.shape[1]), 'channels': 3}}]
    outputs = {}
    errors = {}
    started = time.perf_counter()

    for current_task in tasks:
        model_id = _resolve_model(current_task, selected_model)
        try:
            result = _run_task(current_task, model_id, img_u8, score_th, mask_th)
        except Exception as exc:  # keep endpoint JSON, never hide failure behind fallback
            failure = _task_failure(current_task, model_id, exc)
            algorithms[current_task] = failure
            errors[current_task] = failure['error']
            continue
        algorithms[current_task] = _public_algorithm_summary(result)
        outputs[current_task] = result.get('output', {})
        flat_steps.extend(_prefix_steps(current_task, result.get('steps', [])))

    status = 'pretrained_model' if not errors else ('partial_error' if outputs else 'model_not_available')
    metrics = {
        'status': status,
        'backend': 'torchvision',
        'tasks_requested': ','.join(tasks),
        'tasks_succeeded': len(outputs),
        'tasks_failed': len(errors),
        'elapsed_ms': round((time.perf_counter() - started) * 1000, 1),
        'score_threshold': score_th,
        'mask_threshold': mask_th,
    }
    metrics.update({
        f'{task_id}_model': algorithms.get(task_id, {}).get('model_id', _resolve_model(task_id, selected_model))
        for task_id in tasks
    })
    if errors:
        metrics['errors'] = errors

    response = {
        'module_id': 'ai_eye',
        'family_module_id': 'ai_eye',
        'steps': flat_steps,
        'algorithms': algorithms,
        'outputs': outputs,
        'models': list_models(),
        'metrics': metrics,
    }
    if errors and not outputs:
        response['error'] = 'AI之眼预训练模型不可用：' + '；'.join(errors.values())
    return response


def build_detection_pipeline(image_path=None, **kwargs):
    kwargs['task'] = 'detection'
    kwargs.setdefault('model', kwargs.get('model') or TASK_DEFAULTS['detection'])
    return build_pipeline(image_path=image_path, **kwargs)


def build_semantic_pipeline(image_path=None, **kwargs):
    kwargs['task'] = 'semantic'
    kwargs.setdefault('model', kwargs.get('model') or TASK_DEFAULTS['semantic'])
    return build_pipeline(image_path=image_path, **kwargs)


def build_instance_pipeline(image_path=None, **kwargs):
    kwargs['task'] = 'instance'
    kwargs.setdefault('model', kwargs.get('model') or TASK_DEFAULTS['instance'])
    return build_pipeline(image_path=image_path, **kwargs)


def download_model(model_id: str) -> dict[str, Any]:
    """Load a model once so torchvision downloads and validates weights."""
    if model_id not in MODEL_SPECS:
        raise ValueError(f'Unknown AI Eye model: {model_id}')
    spec = MODEL_SPECS[model_id]
    model, weights, device = _get_model(spec, allow_download=True)
    return {
        'model': model_id,
        'weights': str(weights),
        'device': str(device),
        'cache_dir': _torch_cache_dir(),
        'parameters': sum(int(p.numel()) for p in model.parameters()),
    }


def _run_task(task: str, model_id: str, img_u8: np.ndarray, score_th: float, mask_th: float) -> dict[str, Any]:
    spec = MODEL_SPECS[model_id]
    model, weights, device = _get_model(spec)
    preprocess_info = _preprocess_summary(img_u8, weights)
    started = time.perf_counter()
    if task == 'detection':
        payload = _run_detection(model, weights, device, img_u8, score_th, spec)
    elif task == 'semantic':
        payload = _run_semantic(model, weights, device, img_u8, spec)
    elif task == 'instance':
        payload = _run_instance(model, weights, device, img_u8, score_th, mask_th, spec)
    else:
        raise ValueError(f'Unknown task: {task}')
    payload['steps'].insert(0, {
        'id': 'preprocess',
        'name': '模型预处理',
        'image': _preprocess_card(img_u8, preprocess_info, spec),
        'formula': 'x = normalize(resize(I), mean, std)',
        'explanation': '上传图像被转换为模型需要的张量。不同权重会携带自己的 resize、归一化和类别表配置。',
        'data': preprocess_info,
    })
    payload['steps'].insert(0, {
        'id': 'input',
        'name': '输入图像',
        'image': img_u8,
        'formula': 'I(x,y)=[R,G,B]',
        'explanation': f'{TASK_LABELS[task]} 使用同一张 RGB 输入图像，后续步骤全部来自该图像的真实模型推理。',
        'data': {'height': int(img_u8.shape[0]), 'width': int(img_u8.shape[1])},
    })
    payload['elapsed_ms'] = round((time.perf_counter() - started) * 1000, 1)
    payload.update({
        'task': task,
        'task_label': TASK_LABELS[task],
        'model_id': model_id,
        'model_name': spec.name,
        'family': spec.family,
        'implementation': AI_EYE_IMPLEMENTATION,
    })
    return payload


def _run_detection(model, weights, device, img_u8, score_th, spec):
    import torch
    from torchvision.transforms.functional import to_tensor

    tensor = to_tensor(img_u8).to(device)
    with torch.no_grad():
        output = model([tensor])[0]

    scores = output['scores'].detach().cpu().numpy()
    boxes = output['boxes'].detach().cpu().numpy()
    labels = output['labels'].detach().cpu().numpy()
    categories = _categories(weights)
    raw_count = int(len(scores))
    keep = np.where(scores >= score_th)[0][:40]
    detections = []
    for idx in keep:
        label_idx = int(labels[idx])
        detections.append({
            'box': [round(float(v), 1) for v in boxes[idx].tolist()],
            'label_id': label_idx,
            'label': categories[label_idx] if label_idx < len(categories) else str(label_idx),
            'score': round(float(scores[idx]), 4),
        })
    detections = _attach_detection_focus_images(img_u8, detections)

    objectness = _objectness_heatmap(img_u8, boxes, scores)
    proposal_vis = _draw_boxes(img_u8, _top_raw_detections(boxes, labels, scores, categories, limit=18), show_scores=False)
    result_vis = _draw_boxes(img_u8, detections)
    score_chart = _score_chart(detections, 'Detection confidence')
    return {
        'steps': [
            {'id': 'backbone_fpn', 'name': 'Backbone + FPN 特征金字塔', 'image': _feature_pyramid_card(img_u8, spec),
             'formula': 'P_l = FPN(C_l), l in {2,3,4,5}',
             'explanation': 'FPN 将不同尺度的卷积特征整理成金字塔，让大物体和小物体都能被检测头看到。',
             'data': {'family': spec.family}},
            {'id': 'raw_predictions', 'name': '候选框 / 原始预测', 'image': proposal_vis,
             'formula': 'B_raw = head(P_l)',
             'explanation': '检测头在特征图上产生大量候选框或密集预测；这里展示分数最高的一批原始框。',
             'data': {'raw_predictions': raw_count, 'shown': min(raw_count, 18)}},
            {'id': 'objectness', 'name': '目标性热力图', 'image': objectness,
             'formula': 'H(x,y)=sum_i score_i * 1[(x,y) in box_i]',
             'explanation': '把模型预测框的置信度投影回图像平面，越亮代表越多高分候选认为这里有目标。',
             'data': {'max_score': round(float(scores.max()), 4) if raw_count else 0}},
            {'id': 'score_filter', 'name': '置信度过滤', 'image': score_chart,
             'formula': 'keep_i = 1[score_i >= tau]',
             'explanation': f'只保留置信度不低于 {score_th:.2f} 的预测，降低背景框和低可信框的干扰。',
             'data': {'threshold': score_th, 'kept': len(detections), 'raw': raw_count}},
            {'id': 'detections', 'name': '最终检测框', 'image': result_vis,
             'formula': 'D={(box_i, class_i, score_i)}',
             'explanation': '最终输出是物体级结构：类别、置信度和边界框。它回答“是什么、在哪里”。',
             'data': {'detections': detections}},
        ],
        'metrics': {
            'status': 'pretrained_model',
            'detections': len(detections),
            'raw_predictions': raw_count,
            'threshold': score_th,
        },
        'output': {'detections': detections},
        'detections': detections,
    }


def _run_semantic(model, weights, device, img_u8, spec):
    import torch

    preprocess = weights.transforms()
    tensor = preprocess(_pil_image(img_u8)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)['out'][0]
        probs = torch.softmax(logits, dim=0)
        seg = torch.argmax(probs, dim=0).detach().cpu().numpy().astype(np.int32)
        conf = torch.max(probs, dim=0).values.detach().cpu().numpy()

    seg_resized = _resize_nearest(seg, img_u8.shape[:2])
    conf_resized = _resize_float(conf, img_u8.shape[:2])
    categories = _categories(weights)
    color = _colorize(seg_resized)
    overlay = _overlay(img_u8, color, alpha=0.46)
    conf_vis = _confidence_heatmap(conf_resized)
    labels = _label_summary(seg_resized, categories)
    labels = _attach_semantic_focus_images(img_u8, seg_resized, labels)
    logits_chart = _semantic_chart(labels)
    boundary = _semantic_boundaries(img_u8, seg_resized)
    return {
        'steps': [
            {'id': 'encoder_context', 'name': '编码器与上下文特征', 'image': _semantic_context_card(img_u8, spec),
             'formula': 'F = Encoder(I), logits = Head(F)',
             'explanation': '语义分割把整张图转成密集特征，再用分割头为每个像素预测类别分数。',
             'data': {'family': spec.family, 'logit_shape': list(logits.shape)}},
            {'id': 'logits_summary', 'name': '类别 logits 摘要', 'image': logits_chart,
             'formula': 'p_c(x,y)=softmax(z_c(x,y))',
             'explanation': '每个像素都有一组类别分数；图中汇总面积最大的类别，展示模型认为画面主要由哪些语义组成。',
             'data': {'labels': labels}},
            {'id': 'label_map', 'name': '像素 argmax 标签图', 'image': color,
             'formula': 'label(x,y)=argmax_c p_c(x,y)',
             'explanation': '对每个像素取概率最大的类别，得到语义标签图。同类区域会被合并成同一种颜色。',
             'data': {'classes_present': len(labels)}},
            {'id': 'confidence', 'name': '像素置信度热力图', 'image': conf_vis,
             'formula': 'conf(x,y)=max_c p_c(x,y)',
             'explanation': '越亮的位置代表模型对该像素类别越确定；边界或困难区域通常置信度较低。',
             'data': {'mean_confidence': round(float(conf_resized.mean()), 4)}},
            {'id': 'boundaries', 'name': '语义边界叠加', 'image': boundary,
             'formula': 'edge(label)=1[label(x,y)!=label(neighbor)]',
             'explanation': '标签变化的位置形成语义边界，用来观察分割区域是否贴合图像结构。',
             'data': {'boundary_pixels': int(_label_edges(seg_resized).sum())}},
            {'id': 'overlay', 'name': '语义分割叠加结果', 'image': overlay,
             'formula': 'Y = (1-alpha)I + alpha Color(label)',
             'explanation': '最终语义分割结果覆盖整张图，回答“每个像素属于什么类别”。',
             'data': {'labels': labels}},
        ],
        'metrics': {
            'status': 'pretrained_model',
            'classes_present': len(labels),
            'top_label': labels[0]['label'] if labels else '',
            'mean_confidence': round(float(conf_resized.mean()), 4),
        },
        'output': {'labels': labels},
        'labels': labels,
    }


def _run_instance(model, weights, device, img_u8, score_th, mask_th, spec):
    import torch
    from torchvision.transforms.functional import to_tensor

    tensor = to_tensor(img_u8).to(device)
    with torch.no_grad():
        output = model([tensor])[0]

    scores = output['scores'].detach().cpu().numpy()
    boxes = output['boxes'].detach().cpu().numpy()
    labels = output['labels'].detach().cpu().numpy()
    masks = output.get('masks')
    if masks is None:
        raise RuntimeError('Selected model did not return masks.')
    masks_np = masks.detach().cpu().numpy()[:, 0]
    categories = _categories(weights)
    keep = np.where(scores >= score_th)[0][:25]
    instances = []
    for idx in keep:
        label_idx = int(labels[idx])
        mask_prob = masks_np[idx]
        mask = mask_prob >= mask_th
        instances.append({
            'box': [round(float(v), 1) for v in boxes[idx].tolist()],
            'label_id': label_idx,
            'label': categories[label_idx] if label_idx < len(categories) else str(label_idx),
            'score': round(float(scores[idx]), 4),
            'area': int(mask.sum()),
            'mask': mask,
            'mask_mean_probability': round(float(mask_prob.mean()), 4),
        })

    box_vis = _draw_boxes(img_u8, instances)
    mask_prob_vis = _mask_probability_mosaic(masks_np[keep], instances, img_u8.shape[:2])
    mask_vis = _overlay_masks(img_u8, instances)
    strip = _instance_strip(img_u8, instances)
    public_instances = _public_instances_with_focus(img_u8, instances)
    return {
        'steps': [
            {'id': 'detector_stage', 'name': '检测分支：候选实例框', 'image': box_vis,
             'formula': 'RoI_i = proposal_i(P_l)',
             'explanation': 'Mask R-CNN 先像目标检测一样找到可能存在实例的位置，每个框后续都会进入 mask 分支。',
             'data': {'instances': public_instances}},
            {'id': 'mask_logits', 'name': 'Mask 概率图', 'image': mask_prob_vis,
             'formula': 'P_i(x,y)=sigmoid(mask_head(RoI_i))',
             'explanation': 'mask 分支为每个候选实例预测前景概率图；亮的地方更可能属于该实例。',
             'data': {'mask_threshold': mask_th}},
            {'id': 'mask_threshold', 'name': 'Mask 阈值化', 'image': _mask_binary_mosaic(instances, img_u8.shape[:2]),
             'formula': 'M_i(x,y)=1[P_i(x,y)>=tau_m]',
             'explanation': f'概率图经过 {mask_th:.2f} 阈值变成二值实例掩码，每个实例拥有自己的独立 mask。',
             'data': {'mask_threshold': mask_th, 'areas': [inst['area'] for inst in instances]}},
            {'id': 'masks', 'name': '实例分割叠加结果', 'image': mask_vis,
             'formula': 'Y_i = I overlaid with M_i',
             'explanation': '不同颜色代表不同实例。即使两个目标类别相同，也会被分配不同的实例编号。',
             'data': {'instances': public_instances}},
            {'id': 'instance_crops', 'name': '实例裁剪与个体检查', 'image': strip,
             'formula': 'crop_i = I[box_i] * M_i',
             'explanation': '把每个实例单独裁出，方便观察实例分割是否把独立物体分清楚。',
             'data': {'count': len(public_instances)}},
        ],
        'metrics': {
            'status': 'pretrained_model',
            'instances': len(instances),
            'threshold': score_th,
            'mask_threshold': mask_th,
        },
        'output': {'instances': public_instances},
        'instances': public_instances,
    }


def _allow_model_download():
    return os.environ.get('CV_ALLOW_MODEL_DOWNLOAD', '').strip().lower() in {'1', 'true', 'yes', 'on'}


def _get_model(spec: ModelSpec, allow_download=False):
    global _TORCH_AVAILABLE
    if spec.id in _MODEL_CACHE:
        return _MODEL_CACHE[spec.id]
    try:
        import importlib
        import torch
    except ImportError as exc:
        _TORCH_AVAILABLE = False
        raise RuntimeError('PyTorch/torchvision 未安装，无法运行 AI之眼预训练模型。') from exc

    _TORCH_AVAILABLE = True
    module = importlib.import_module(spec.builder_module)
    builder = getattr(module, spec.builder_name)
    weights_cls = getattr(module, spec.weights_name)
    weights = weights_cls.DEFAULT
    if not (allow_download or _allow_model_download()) and not _is_weight_cached(weights):
        raise RuntimeError(_missing_weight_message(spec))
    try:
        model = builder(weights=weights)
    except Exception as exc:
        raise RuntimeError(_download_error_message(spec, exc)) from exc
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    _MODEL_CACHE[spec.id] = (model, weights, device)
    return _MODEL_CACHE[spec.id]


def _weights_enum(spec: ModelSpec, allow_import_error=False):
    try:
        import importlib
        module = importlib.import_module(spec.builder_module)
        return getattr(module, spec.weights_name).DEFAULT
    except Exception:
        if allow_import_error:
            return None
        raise


def _weight_filename(weights) -> str:
    if weights is None:
        return ''
    url = getattr(weights, 'url', '') or ''
    return url.rsplit('/', 1)[-1] if url else ''


def _is_weight_cached(weights) -> bool:
    filename = _weight_filename(weights)
    return bool(filename and os.path.exists(os.path.join(_torch_cache_dir(), filename)))


def _torch_cache_dir() -> str:
    try:
        import torch
        return os.path.join(torch.hub.get_dir(), 'checkpoints')
    except Exception:
        return os.environ.get('TORCH_HOME', os.path.expanduser('~/.cache/torch')) + '/hub/checkpoints'


def _download_error_message(spec: ModelSpec, exc: Exception) -> str:
    return (
        f'{spec.name} 权重不可用或下载失败：{type(exc).__name__}: {exc}. '
        f'请运行 python prepare_ai_eye_assets.py --model {spec.id}，'
        f'或确认网络可访问 torchvision 官方权重；缓存目录：{_torch_cache_dir()}'
    )


def _missing_weight_message(spec: ModelSpec) -> str:
    return (
        f'{spec.name} 权重尚未缓存到本机。为避免网页请求卡住，/api/demo/ai_eye 默认不会自动下载大权重。'
        f'请先运行 python prepare_ai_eye_assets.py --model {spec.id}，'
        f'或设置 CV_ALLOW_MODEL_DOWNLOAD=1 后显式预热；缓存目录：{_torch_cache_dir()}'
    )


def _load_ai_eye_image(image_path):
    if image_path:
        return load_image_u8(image_path, mode='rgb', max_side=512)
    return _load_or_fixture(image_path=image_path)


def _resolve_model(task: str, requested: str | None):
    if requested and requested in MODEL_SPECS and MODEL_SPECS[requested].task == task:
        return requested
    return TASK_DEFAULTS[task]


def _task_failure(task: str, model_id: str, exc: Exception) -> dict[str, Any]:
    return {
        'task': task,
        'task_label': TASK_LABELS.get(task, task),
        'model_id': model_id,
        'model_name': MODEL_SPECS.get(model_id, ModelSpec(model_id, task, model_id, '', '', '', '')).name,
        'steps': [],
        'metrics': {'status': 'model_not_available', 'error_type': type(exc).__name__},
        'error': str(exc),
        'download_command': f'python prepare_ai_eye_assets.py --model {model_id}',
        'cache_dir': _torch_cache_dir(),
    }


def _public_algorithm_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Keep nested algorithm metadata JSON-safe; images live in top-level steps."""
    return {
        'task': result.get('task'),
        'task_label': result.get('task_label'),
        'model_id': result.get('model_id'),
        'model_name': result.get('model_name'),
        'family': result.get('family'),
        'elapsed_ms': result.get('elapsed_ms'),
        'metrics': result.get('metrics', {}),
        'step_ids': [step.get('id') for step in result.get('steps', [])],
    }


def _error_result(message, status, **extra):
    metrics = {'status': status, 'error': message}
    metrics.update(extra)
    return {
        'module_id': 'ai_eye',
        'steps': [],
        'metrics': metrics,
        'models': list_models(),
        'outputs': {},
        'algorithms': {},
        'error': message,
    }


def _prefix_steps(task: str, steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for step in steps:
        item = dict(step)
        item['id'] = f'{task}_{item.get("id", "step")}'
        item['name'] = f'{TASK_LABELS.get(task, task)}：{item.get("name", item["id"])}'
        out.append(item)
    return out


def _preprocess_summary(img, weights) -> dict[str, Any]:
    meta = getattr(weights, 'meta', {}) or {}
    transform = weights.transforms()
    return {
        'input_shape': [int(img.shape[0]), int(img.shape[1]), 3],
        'weights': str(weights),
        'categories': len(meta.get('categories', [])),
        'transform': transform.__class__.__name__,
    }


def _categories(weights) -> list[str]:
    return list((getattr(weights, 'meta', {}) or {}).get('categories', []))


def _pil_image(arr):
    from PIL import Image
    return Image.fromarray(np.asarray(arr, dtype=np.uint8))


def _resize_nearest(arr, shape_hw):
    from PIL import Image
    h, w = shape_hw
    return np.array(Image.fromarray(arr.astype(np.int32), mode='I').resize((w, h), Image.NEAREST), dtype=np.int32)


def _resize_float(arr, shape_hw):
    from PIL import Image
    h, w = shape_hw
    return np.array(Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0


def _colors():
    return np.array([
        [239, 68, 68], [34, 197, 94], [59, 130, 246], [245, 158, 11],
        [168, 85, 247], [20, 184, 166], [236, 72, 153], [250, 204, 21],
        [14, 165, 233], [132, 204, 22],
    ], dtype=np.uint8)


def _draw_boxes(img, detections, show_scores=True):
    from PIL import Image, ImageDraw, ImageFont
    canvas = Image.fromarray(img.copy())
    draw = ImageDraw.Draw(canvas)
    colors = _colors()
    w, h = canvas.size
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = [int(round(v)) for v in det['box']]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        col = tuple(int(v) for v in colors[i % len(colors)])
        for off in range(3):
            draw.rectangle((x1 - off, y1 - off, x2 + off, y2 + off), outline=col)
        if show_scores:
            text = det.get('label', 'object')
            if det.get('score') is not None:
                text += f" {float(det['score']):.2f}"
            tw = min(max(70, 7 * len(text) + 12), max(80, x2 - x1))
            y_text = max(0, y1 - 22)
            draw.rectangle((x1, y_text, x1 + tw, y_text + 20), fill=col)
            draw.text((x1 + 5, y_text + 3), text[:34], fill=(255, 255, 255))
    return np.array(canvas)


def _attach_detection_focus_images(img, detections, limit=16):
    out = []
    for idx, det in enumerate(detections):
        item = dict(det)
        if idx < limit:
            item['focus_image_base64'] = to_base64(_draw_boxes(img, [det]))
            item['focus_note'] = '该图只高亮当前检测框，便于把类别、置信度和图像位置对应起来。'
        out.append(item)
    return out


def _top_raw_detections(boxes, labels, scores, categories, limit=18):
    order = np.argsort(-scores)[:limit]
    rows = []
    for idx in order:
        label_idx = int(labels[idx])
        rows.append({
            'box': [float(v) for v in boxes[idx].tolist()],
            'label': categories[label_idx] if label_idx < len(categories) else str(label_idx),
            'score': float(scores[idx]),
        })
    return rows


def _objectness_heatmap(img, boxes, scores):
    h, w = img.shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)
    for box, score in zip(boxes[:80], scores[:80]):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            heat[y1:y2, x1:x2] += float(score)
    if heat.max() > 0:
        heat /= float(heat.max())
    color = _heatmap(heat)
    return _overlay(img, color, alpha=0.58)


def _heatmap(values, color=(239, 68, 68)):
    v = np.clip(np.asarray(values, dtype=np.float32), 0, 1)
    r = (v * color[0] + (1 - v) * 15).astype(np.uint8)
    g = (v * color[1] + (1 - v) * 23).astype(np.uint8)
    b = (v * color[2] + (1 - v) * 42).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def _overlay(img, color, alpha=0.45):
    return np.clip(img.astype(np.float32) * (1 - alpha) + color.astype(np.float32) * alpha, 0, 255).astype(np.uint8)


def _score_chart(rows, title, width=720, height=320):
    from PIL import Image, ImageDraw
    canvas = Image.new('RGB', (width, height), (15, 23, 42))
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 14), title, fill=(226, 232, 240))
    if not rows:
        draw.text((24, height // 2), 'No predictions above threshold', fill=(148, 163, 184))
        return np.array(canvas)
    shown = rows[:10]
    bar_h = max(16, (height - 62) // len(shown) - 7)
    for i, row in enumerate(shown):
        y = 48 + i * (bar_h + 7)
        score = float(row.get('score', 0))
        label = str(row.get('label', 'object'))[:22]
        draw.text((18, y + 2), label, fill=(203, 213, 225))
        draw.rectangle((190, y, width - 26, y + bar_h), outline=(51, 65, 85))
        draw.rectangle((190, y, 190 + int((width - 216) * score), y + bar_h), fill=(56, 189, 248))
        draw.text((width - 76, y + 2), f'{score:.2f}', fill=(226, 232, 240))
    return np.array(canvas)


def _semantic_chart(labels, width=720, height=320):
    rows = [{'label': item['label'], 'score': item['ratio']} for item in labels[:10]]
    return _score_chart(rows, 'Semantic pixel area ratio')


def _colorize(seg):
    palette = np.array([
        [15, 23, 42], [239, 68, 68], [34, 197, 94], [59, 130, 246],
        [245, 158, 11], [168, 85, 247], [20, 184, 166], [236, 72, 153],
        [250, 204, 21], [14, 165, 233], [132, 204, 22], [251, 113, 133],
        [99, 102, 241], [45, 212, 191], [251, 146, 60], [190, 242, 100],
        [125, 211, 252], [216, 180, 254], [253, 164, 175], [187, 247, 208],
        [226, 232, 240],
    ], dtype=np.uint8)
    return palette[np.asarray(seg, dtype=np.int32) % len(palette)]


def _confidence_heatmap(conf):
    conf = np.clip(np.asarray(conf, dtype=np.float32), 0, 1)
    return _heatmap(conf, color=(34, 197, 94))


def _label_summary(seg, categories):
    labels, counts = np.unique(seg, return_counts=True)
    total = max(1, int(seg.size))
    top = sorted(zip(labels.tolist(), counts.tolist()), key=lambda x: x[1], reverse=True)[:10]
    return [
        {
            'label_id': int(label),
            'label': categories[int(label)] if int(label) < len(categories) else str(label),
            'pixels': int(count),
            'ratio': round(float(count) / total, 4),
        }
        for label, count in top
    ]


def _attach_semantic_focus_images(img, seg, labels):
    out = []
    for label in labels:
        item = dict(label)
        label_id = int(item['label_id'])
        focus = _semantic_label_focus(img, seg, label_id)
        item['focus_image_base64'] = to_base64(focus)
        item['focus_note'] = '该图只保留当前语义类别的像素区域，灰暗部分是其他类别。'
        out.append(item)
    return out


def _semantic_label_focus(img, seg, label_id):
    mask = np.asarray(seg == label_id, dtype=bool)
    out = (img.astype(np.float32) * 0.24).astype(np.uint8)
    palette = _colors()
    color = palette[int(label_id) % len(palette)].astype(np.float32)
    out[mask] = np.clip(img[mask].astype(np.float32) * 0.48 + color * 0.52, 0, 255)
    edge = _mask_edges(mask)
    out[edge] = [250, 204, 21]
    return out.astype(np.uint8)


def _label_edges(seg):
    edge = np.zeros(seg.shape, dtype=bool)
    edge[1:, :] |= seg[1:, :] != seg[:-1, :]
    edge[:, 1:] |= seg[:, 1:] != seg[:, :-1]
    return edge


def _mask_edges(mask):
    return _label_edges(np.asarray(mask, dtype=np.uint8))


def _semantic_boundaries(img, seg):
    out = img.copy()
    edge = _label_edges(seg)
    out[edge] = [250, 204, 21]
    return out


def _overlay_masks(img, instances):
    out = img.astype(np.float32).copy()
    colors = _colors().astype(np.float32)
    for i, inst in enumerate(instances):
        mask = np.asarray(inst['mask'], dtype=bool)
        col = colors[i % len(colors)]
        out[mask] = out[mask] * 0.45 + col * 0.55
    return np.clip(out, 0, 255).astype(np.uint8)


def _public_instances_with_focus(img, instances, limit=16):
    out = []
    colors = _colors()
    for idx, inst in enumerate(instances):
        item = {k: v for k, v in inst.items() if k != 'mask'}
        if idx < limit:
            mask = np.asarray(inst['mask'], dtype=bool)
            color = tuple(int(v) for v in colors[idx % len(colors)])
            focus = _overlay_mask_focus(img, mask, inst['box'], color=color)
            item['focus_image_base64'] = to_base64(focus)
            item['focus_note'] = '该图只高亮当前实例 mask；同类物体仍按独立个体显示。'
        out.append(item)
    return out


def _overlay_mask_focus(img, mask, box, color=(34, 197, 94)):
    from PIL import Image, ImageDraw
    out = (img.astype(np.float32) * 0.25).astype(np.uint8)
    col = np.array(color, dtype=np.float32)
    out[mask] = np.clip(img[mask].astype(np.float32) * 0.38 + col * 0.62, 0, 255)
    edge = _mask_edges(mask)
    out[edge] = [250, 204, 21]
    canvas = Image.fromarray(out)
    draw = ImageDraw.Draw(canvas)
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
    return np.array(canvas)


def _mask_probability_mosaic(mask_probs, instances, shape_hw):
    h, w = shape_hw
    if len(mask_probs) == 0:
        return np.zeros((180, 360, 3), dtype=np.uint8) + 18
    thumbs = []
    for mask in mask_probs[:8]:
        thumbs.append(_heatmap(np.clip(mask, 0, 1), color=(236, 72, 153)))
    return _tile_images(thumbs, thumb_h=150)


def _mask_binary_mosaic(instances, shape_hw):
    thumbs = []
    colors = _colors()
    for i, inst in enumerate(instances[:8]):
        mask = np.asarray(inst['mask'], dtype=bool)
        canvas = np.zeros((*mask.shape, 3), dtype=np.uint8) + 18
        canvas[mask] = colors[i % len(colors)]
        thumbs.append(canvas)
    if not thumbs:
        return np.zeros((180, 360, 3), dtype=np.uint8) + 18
    return _tile_images(thumbs, thumb_h=150)


def _instance_strip(img, instances):
    thumbs = []
    for inst in instances[:8]:
        x1, y1, x2, y2 = [int(round(v)) for v in inst['box']]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = img[y1:y2, x1:x2].copy()
        mask = np.asarray(inst['mask'][y1:y2, x1:x2], dtype=bool)
        crop[~mask] = (crop[~mask] * 0.25).astype(np.uint8)
        thumbs.append(crop)
    if not thumbs:
        return np.zeros((150, 360, 3), dtype=np.uint8) + 18
    return _tile_images(thumbs, thumb_h=150)


def _tile_images(images, thumb_h=140):
    from PIL import Image
    resized = []
    for img in images:
        pil = Image.fromarray(np.asarray(img, dtype=np.uint8))
        ratio = thumb_h / max(1, pil.height)
        resized.append(np.array(pil.resize((max(1, int(pil.width * ratio)), thumb_h), Image.BILINEAR)))
    gap = 8
    width = sum(im.shape[1] for im in resized) + gap * (len(resized) - 1)
    canvas = np.zeros((thumb_h, max(width, 320), 3), dtype=np.uint8) + 18
    x = 0
    for im in resized:
        canvas[:im.shape[0], x:x + im.shape[1]] = im
        x += im.shape[1] + gap
    return canvas


def _preprocess_card(img, info, spec):
    from PIL import Image, ImageDraw
    canvas = Image.new('RGB', (720, 320), (15, 23, 42))
    draw = ImageDraw.Draw(canvas)
    thumb = Image.fromarray(img).resize((210, 210), Image.BILINEAR)
    canvas.paste(thumb, (24, 72))
    draw.text((24, 28), 'Preprocess', fill=(226, 232, 240))
    lines = [
        spec.name,
        f"input: {info['input_shape'][1]} x {info['input_shape'][0]} x 3",
        f"weights: {info['weights']}",
        f"categories: {info['categories']}",
        'RGB image -> tensor -> normalized batch',
    ]
    y = 74
    for line in lines:
        draw.text((270, y), line[:58], fill=(203, 213, 225))
        y += 34
    return np.array(canvas)


def _feature_pyramid_card(img, spec):
    from PIL import Image, ImageDraw
    canvas = Image.new('RGB', (720, 320), (15, 23, 42))
    draw = ImageDraw.Draw(canvas)
    gray = ensure_gray(img)
    edge = _gradient_vis(gray)
    thumb = Image.fromarray(edge).resize((220, 220), Image.BILINEAR)
    canvas.paste(thumb, (24, 62))
    draw.text((24, 26), 'Backbone feature evidence', fill=(226, 232, 240))
    x = 300
    for i, scale in enumerate([1.0, 0.72, 0.50, 0.34]):
        w = int(230 * scale)
        h = int(130 * scale)
        y = 50 + i * 55
        draw.rectangle((x, y, x + w, y + h), outline=(56, 189, 248), width=2)
        draw.text((x + w + 12, y + 6), f'P{i + 2}', fill=(125, 211, 252))
    draw.text((300, 270), spec.description[:62], fill=(148, 163, 184))
    return np.array(canvas)


def _semantic_context_card(img, spec):
    from PIL import Image, ImageDraw
    canvas = Image.new('RGB', (720, 320), (15, 23, 42))
    draw = ImageDraw.Draw(canvas)
    small = Image.fromarray(img).resize((180, 180), Image.BILINEAR)
    canvas.paste(small, (26, 70))
    for i, radius in enumerate([35, 62, 92]):
        draw.ellipse((330 - radius, 160 - radius, 330 + radius, 160 + radius), outline=[56, 189, 248, 168, 85, 247, 34, 197, 94][i % 3], width=3)
    draw.text((26, 28), 'Dense context', fill=(226, 232, 240))
    draw.text((450, 72), spec.name, fill=(226, 232, 240))
    draw.text((450, 110), '每个像素都有类别概率', fill=(203, 213, 225))
    draw.text((450, 146), '上下文决定区域语义', fill=(203, 213, 225))
    draw.text((450, 182), '输出尺寸再对齐原图', fill=(203, 213, 225))
    return np.array(canvas)


def _gradient_vis(gray):
    g = gray.astype(np.float32)
    gy, gx = np.gradient(g)
    mag = np.sqrt(gx * gx + gy * gy)
    if mag.max() > 0:
        mag = mag / mag.max()
    return _heatmap(mag, color=(56, 189, 248))
