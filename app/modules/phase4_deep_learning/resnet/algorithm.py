
"""ResNet inference via torchvision — real pretrained weights."""
import numpy as np
from app.utils.image_utils import load_image_u8

_MODEL = None

def _get_model():
    global _MODEL
    if _MODEL is None:
        import torch
        from torchvision.models import resnet50, ResNet50_Weights
        _MODEL = resnet50(weights=ResNet50_Weights.DEFAULT)
        _MODEL.eval()
    return _MODEL

def build_pipeline(image_path=None, **kwargs):
    try:
        import torch
        from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
    except ImportError:
        return {'steps': [{'id':'error','name':'PyTorch未安装','image':np.zeros((100,400,3),dtype=np.uint8)+240,
                'explanation':'需要安装PyTorch和torchvision才能运行ResNet真实模型。'}], 'metrics':{'status':'dependency_missing'}}

    if not image_path:
        img_u8 = (np.ones((224,224,3),dtype=np.uint8)*128)
    else:
        img_u8 = load_image_u8(image_path, mode='rgb', max_side=256)

    try:
        model = _get_model()
    except Exception as e:
        return {'steps': [{'id':'error','name':'模型加载失败','image':img_u8,
                'explanation': str(e)[:200]}], 'metrics':{'status':'model_load_failed'}}

    preprocess = Compose([Resize(256), CenterCrop(224), ToTensor(),
                          Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
    from PIL import Image
    input_tensor = preprocess(Image.fromarray(img_u8)).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)[0]
        probs = torch.softmax(logits, dim=0)
        top5_prob, top5_idx = torch.topk(probs, 5)

    with open('static/data/imagenet_classes.txt') as f:
        classes = [l.strip() for l in f.readlines()]

    top5 = [(classes[idx], float(p)) for idx, p in zip(top5_idx.tolist(), top5_prob.tolist())]

    from PIL import ImageDraw
    result_img = Image.fromarray(img_u8)
    draw = ImageDraw.Draw(result_img)
    for i, (name, prob) in enumerate(top5):
        draw.text((10, 10 + i*22), f'{name}: {prob:.3f}', fill=(34,197,94))

    return {'steps': [
        {'id':'input','name':'输入图像','image':img_u8,'explanation':'ResNet-50 输入 (224x224)。'},
        {'id':'features','name':'残差块特征提取','image':np.array(result_img),
         'explanation':'50层残差网络提取层次化特征。'},
        {'id':'predictions','name':'Top-5 预测','image':_bar_chart(top5),
         'explanation':'ImageNet-1K 1000类分类结果。'},
    ], 'metrics': {'status':'pretrained_model','model':'resnet50','backend':'torchvision',
        'top1':top5[0][0],'top1_conf':round(top5[0][1],4)}}

def _bar_chart(items, w=400, h=180):
    from PIL import Image as PILImage, ImageDraw
    img = PILImage.new('RGB',(w,h),(15,23,42))
    d = ImageDraw.Draw(img)
    for i,(name,p) in enumerate(items):
        y=10+i*32; bw=int(p*w*0.7)
        d.rectangle((100,y,100+bw,y+22),fill=(59,130,246))
        d.text((8,y+4),name[:20],fill=(226,232,240))
        d.text((105+bw,y+4),f'{p:.3f}',fill=(148,163,184))
    return np.array(img)
