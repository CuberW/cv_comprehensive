"""
Route hub.
- Root path returns the main shell page
- /api/modules returns module registry info (for frontend navigation)
- Legacy API routes for interactive algorithm pages (compatible with old CV project)
"""
import os
import inspect
import imageio.v3 as iio
from datetime import datetime
from flask import Blueprint, render_template, jsonify, request
from app.modules import MODULE_REGISTRY, get_modules_by_phase
from app.modules.implementation import get_implementation_meta
from app.modules.offline_teaching import (
    BLUEPRINT_MODULES,
    EXTERNAL_WEIGHT_MODULES,
    LOCAL_FRONTIER_ALGORITHM_MODULES,
    LOCAL_TEACHING_MODEL_MODULES,
    OFFLINE_TEACHING_MODULES,
)
from app.utils.image_utils import to_base64, load_image_u8
from app.runners import get_remote_runner

main_bp = Blueprint('main', __name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LEGACY_HISTORY = []
_LEGACY_HISTORY_NEXT_ID = 1
TEACHING_PAGE_IDS = {
    'grayscale', 'noise', 'bilateral', 'convolution', 'smoothing',
    'kmeans', 'watershed', 'grabcut', 'slic', 'hog_svm', 'optical_flow', 'stereo',
    'ai_eye', 'detection', 'semantic', 'instance', 'unet', 'yolo',
    'resnet', 'gan', 'diffusion', 'vit', 'detr', 'clip', 'sam', 'stable_diffusion', 'nerf',
}

# New dedicated pages (override teaching.html fallback)
DEDICATED_PAGES = {
    'smoothing': 'smoothing.html',
    'median': 'smoothing.html',
    'sobel': 'sobel_new.html',
    'histogram': 'histogram_new.html',
    'threshold': 'threshold_new.html',
    'gaussian': 'smoothing.html',
    'bilateral': 'smoothing.html',
}

PREFERRED_STATIC_PAGES = {
    'ai_eye': 'detection_segmentation.html',
    'detection': 'detection_segmentation.html?task=detection',
    'semantic': 'detection_segmentation.html?task=semantic',
    'instance': 'detection_segmentation.html?task=instance',
    'yolo': 'detection_segmentation.html?task=yolo',
    'unet': 'detection_segmentation.html?task=unet',
    'resnet': 'nn_resnet.html',
    'gan': 'nn_gan.html',
    'diffusion': 'diffusion.html',
    'vit': 'vit.html',
    'detr': 'detr.html',
    'clip': 'clip.html',
    'sam': 'sam.html',
    'nerf': 'nerf.html',
    'stable_diffusion': 'stable_diffusion.html',
}

MODULE_ALIASES = {
    'canny': 'edge',
    'harris': 'corner',
    'faster_rcnn': 'detection',
    'fcn': 'semantic',
    'mask_rcnn': 'instance',
    'tpl_match': 'template_match',
    'sd': 'stable_diffusion',
}


def _canonical_module_id(module_id):
    return MODULE_ALIASES.get(module_id, module_id)


def _static_page_exists(page):
    page_name = (page or '').split('?', 1)[0]
    return bool(page_name) and os.path.exists(os.path.join(PROJECT_ROOT, 'static', 'pages', page_name))


def _module_page(module_id, cls):
    if module_id in DEDICATED_PAGES:
        return DEDICATED_PAGES[module_id]
    preferred = PREFERRED_STATIC_PAGES.get(module_id)
    if preferred and _static_page_exists(preferred):
        return preferred
    page = cls.get_page() if cls and hasattr(cls, 'get_page') else None
    if page:
        return page
    if module_id in TEACHING_PAGE_IDS:
        return f'teaching.html?id={module_id}'
    return None


# ================================================================
#  Main shell
# ================================================================

@main_bp.route('/')
def index():
    """Main entry point -- returns the SPA shell."""
    return render_template('index.html')


@main_bp.route('/api/modules')
def api_modules():
    """Return all registered module metadata organized by phase."""
    phases = get_modules_by_phase()
    hidden_module_ids = {'ai_eye', 'nms', 'template_match', 'hough', 'contour'}
    for phase in phases:
        phase['modules'] = [
            mod for mod in phase['modules']
            if mod.get('id') not in hidden_module_ids
        ]
        for mod in phase['modules']:
            cls = MODULE_REGISTRY.get(mod['id'])
            page = _module_page(mod['id'], cls)
            if page:
                mod['page'] = page
            mod['implementation'] = get_implementation_meta(mod['id'])
    phases = [phase for phase in phases if phase['modules']]

    return jsonify({'phases': phases, 'total': len(MODULE_REGISTRY)})


# ================================================================
#  Legacy API routes (compatible with old CV project interactive pages)
#  These routes are called by the interactive HTML pages ported from
#  the old CV project.
# ================================================================

def _save_upload(file):
    """Save uploaded file and return (unique_name, upload_path)."""
    import uuid
    upload_dir = os.path.join(PROJECT_ROOT, 'static', 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    ext = os.path.splitext(file.filename)[1] or '.png'
    unique_name = f"{uuid.uuid4().hex}{ext}"
    upload_path = os.path.join(upload_dir, unique_name)
    file.save(upload_path)
    return unique_name, upload_path


def _save_legacy_result(image, source_name, prefix):
    upload_dir = os.path.join(PROJECT_ROOT, 'static', 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    stem = os.path.splitext(source_name)[0]
    result_name = f'{prefix}{stem}.png'
    result_path = os.path.join(upload_dir, result_name)
    iio.imwrite(result_path, image)
    return result_name, result_path


def _legacy_file_info(path):
    size_bytes = os.path.getsize(path)
    if size_bytes >= 1024 * 1024:
        size_text = f'{size_bytes / (1024 * 1024):.1f} MB'
    elif size_bytes >= 1024:
        size_text = f'{size_bytes / 1024:.1f} KB'
    else:
        size_text = f'{size_bytes} B'
    return {
        'file_size': size_text,
        'format': os.path.splitext(path)[1].upper().lstrip('.') or 'PNG',
    }


def _legacy_array_info(image):
    shape = image.shape
    return {
        'width': int(shape[1]),
        'height': int(shape[0]),
        'channels': 1 if len(shape) == 2 else int(shape[2]),
    }


def _legacy_pixel_stats(image):
    import numpy as np
    arr = np.asarray(image, dtype=np.float64)
    return {
        'min': int(arr.min()),
        'max': int(arr.max()),
        'mean': round(float(arr.mean()), 1),
        'std': round(float(arr.std()), 1),
    }


def _legacy_histogram(image, bins=256):
    import numpy as np
    hist, _ = np.histogram(np.asarray(image).ravel(), bins=bins, range=(0, 256))
    return [int(v) for v in hist]


def _legacy_history_snapshot():
    return [dict(entry) for entry in _LEGACY_HISTORY]


def _legacy_add_history_entry(module_key, module_title, original_image, result_image,
                              original_filename='', original_info=None, result_info=None,
                              stats=None, histogram=None, metadata=None):
    global _LEGACY_HISTORY_NEXT_ID
    entry = {
        'id': _LEGACY_HISTORY_NEXT_ID,
        'module_key': module_key,
        'module_title': module_title,
        'original_image': original_image,
        'result_image': result_image,
        'original_filename': original_filename,
        'original_info': original_info,
        'result_info': result_info,
        'stats': stats,
        'histogram': histogram,
        'metadata': metadata or {},
        'timestamp': datetime.now().isoformat(),
    }
    _LEGACY_HISTORY_NEXT_ID += 1
    _LEGACY_HISTORY.insert(0, entry)
    del _LEGACY_HISTORY[100:]
    return dict(entry)


def _legacy_remove_history_entry(entry_id):
    for idx, entry in enumerate(_LEGACY_HISTORY):
        if int(entry.get('id', -1)) == int(entry_id):
            del _LEGACY_HISTORY[idx]
            return True
    return False


def _legacy_clear_history(module_key=None):
    if module_key:
        before = len(_LEGACY_HISTORY)
        _LEGACY_HISTORY[:] = [entry for entry in _LEGACY_HISTORY if entry.get('module_key') != module_key]
        return before - len(_LEGACY_HISTORY)
    cleared = len(_LEGACY_HISTORY)
    _LEGACY_HISTORY.clear()
    return cleared


def _demo_fixture_path():
    """Return a stable built-in image for demo endpoints without uploads."""
    candidates = [
        os.path.join(PROJECT_ROOT, 'static', 'images', 'demo-street.jpg'),
        os.path.join(PROJECT_ROOT, 'bus.jpg'),
        os.path.join(PROJECT_ROOT, '1.jpg'),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _to_data_url(image):
    return f'data:image/png;base64,{to_base64(image)}'


STEP_FORMULA_FALLBACKS = {
    'original': 'I(x,y)',
    'input': 'I(x,y)',
    'left': 'I_l(x,y)',
    'right': 'I_r(x,y)',
    'frame1': 'I_t(x,y)',
    'frame2': 'I_{t+1}(x,y)',
    'gray': 'Y=0.299R+0.587G+0.114B',
    'grayscale': 'Y=0.299R+0.587G+0.114B',
    'histogram': 'h(k)=sum_{x,y} [I(x,y)=k]',
    'cdf': 'CDF(k)=sum_{i<=k} h(i)',
    'threshold': 'B(x,y)=1[I(x,y)>=T]',
    'result': 'Y=f(I)',
    'output': 'Y=f(I)',
    'features': 'phi(I)',
    'feature': 'phi(I)',
    'gradient': '|grad I|=sqrt(G_x^2+G_y^2)',
    'magnitude': '|grad I|=sqrt(G_x^2+G_y^2)',
    'angle': 'theta=atan2(G_y,G_x)',
    'kernel': 'Y=K*I',
    'kernels': 'K_{t+1}=K_t-eta grad_K L',
    'loss': 'L=mean((K*X-Y)^2)',
    'mask': 'M(x,y)=1[p(x,y)>tau]',
    'boxes': 'b=(x,y,w,h), score=p(class|b)',
    'attention': 'Attention(Q,K,V)=softmax(QK^T/sqrt(d))V',
    'tokens': 'z_0=[x_1E;...;x_NE]+p',
    'embedding': 'z=phi(x)/||phi(x)||',
}

MODULE_FORMULA_FALLBACKS = {
    'smoothing': 'I_denoised=f(I_noisy; filter)',
    'bilateral': "I'_p=(1/W_p) sum_q G_s(||p-q||)G_r(|I_p-I_q|)I_q",
    'convolution': 'Y(i,j)=sum_{u,v} K(u,v) I(i+u,j+v)',
    'gaussian': 'G(x,y)=1/(2*pi*sigma^2) exp(-(x^2+y^2)/(2*sigma^2))',
    'median': "I'(x,y)=median{I(u,v)|(u,v) in Omega}",
    'noise': 'I_noisy=I+n',
    'sobel': '|grad I|=sqrt((K_x*I)^2+(K_y*I)^2)',
    'edge': 'Canny=NMS(|grad(G_sigma*I)|)+hysteresis',
    'corner': 'R=det(M)-k trace(M)^2',
    'shitomasi': 'R=min(lambda_1,lambda_2)',
    'sift': 'D(x,y,sigma)=L(x,y,k sigma)-L(x,y,sigma)',
    'hough': 'rho=x cos(theta)+y sin(theta)',
    'morphology': 'A oplus B, A ominus B',
    'contour': 'C={(x,y)|B(x,y)=1 and neighbor(B)=0}',
    'nms': 'keep(p) iff R(p)=max_{q in N(p)} R(q)',
    'template_match': 'NCC=sum((I-mu_I)(T-mu_T))/(sigma_I sigma_T)',
    'kmeans': 'min sum_i ||x_i-mu_{c_i}||^2',
    'watershed': 'label=watershed(grad I, markers)',
    'grabcut': 'E(alpha,k,theta,z)=U(alpha,k,theta,z)+V(alpha,z)',
    'slic': 'D=sqrt(d_c^2+(m/S)^2 d_s^2)',
    'hog_svm': 'score=w^T HOG(x)+b',
    'optical_flow': 'I_x u+I_y v+I_t=0',
    'stereo': 'd*=argmin_d sum |I_l(x,y)-I_r(x-d,y)|',
    'frequency': 'F(u,v)=sum I(x,y)e^{-j2pi(ux/M+vy/N)}',
    'match': 'm*=argmin_j ||d_i-d_j||',
    'ncuts': 'min Ncut(A,B)=cut(A,B)/assoc(A,V)+cut(A,B)/assoc(B,V)',
    'bovw_spm': 'h_k=sum_i [q(d_i)=k]',
    'calibration': 's[u,v,1]^T=K[R|t][X,Y,Z,1]^T',
    'epipolar': "x'^T F x=0",
    'sfm': 'x=P X, x_prime=P_prime X',
    'detection': 'box=head(FPN(I)), score=softmax(cls)',
    'semantic': 'p_{cxy}=softmax(f_theta(I)_{cxy})',
    'instance': 'mask_i=sigmoid(f_theta(RoI_i))',
    'resnet': 'y=F(x)+x',
    'unet': 'D_l=concat(up(D_{l+1}),E_l)',
    'yolo': 'b,cls,obj=head(grid(I))',
    'gan': 'min_G max_D E log D(x)+E log(1-D(G(z)))',
    'diffusion': 'z_{t-1}=scheduler(z_t, epsilon_theta(z_t,t,c))',
    'ddpm': 'q(x_t|x_0)=N(sqrt(alpha_bar_t)x_0,(1-alpha_bar_t)I)',
    'lenet': 'y=softmax(W flatten(pool(conv(X)))+b)',
    'cnn_basics': 'Y=pool(ReLU(K*I+b))',
    'conv_training': 'K_{t+1}=K_t-eta grad_K L',
    'vit': 'Attention(Q,K,V)=softmax(QK^T/sqrt(d))V',
    'detr': 'boxes=Decoder(object_queries, Encoder(I))',
    'clip': 'sim(I,T)=<phi_I(I),phi_T(T)>/(||phi_I||||phi_T||)',
    'sam': 'M=MaskDecoder(ImageEncoder(I),PromptEncoder(p))',
    'stable_diffusion': 'z_{t-1}=scheduler.step(epsilon_theta(z_t,t,c),z_t)',
    'nerf': 'C(r)=sum_i T_i(1-exp(-sigma_i delta_i))c_i',
}


def _strip_data_url(value):
    if isinstance(value, str) and value.startswith('data:image') and ',' in value:
        return value.split(',', 1)[1]
    return value


def _fallback_formula(module_id, step):
    sid = str(step.get('id') or step.get('step') or '').lower()
    name = str(step.get('name') or step.get('title') or '').lower()
    for key, formula in STEP_FORMULA_FALLBACKS.items():
        if sid == key or key in sid or key in name:
            return formula
    return MODULE_FORMULA_FALLBACKS.get(module_id, f'Y=f_{{{module_id}}}(X)')


def _fallback_explanation(module_id, normalized):
    name = normalized.get('name') or normalized.get('id') or 'step'
    return f'该步骤展示 {name}，是 {module_id} 算法流水线中的中间结果；后续步骤会基于它继续计算或汇总。'


def _data_summary(data):
    if not isinstance(data, dict) or not data:
        return ''
    parts = []
    for key, value in list(data.items())[:4]:
        if isinstance(value, (list, tuple)):
            parts.append(f'{key}: len={len(value)}')
        elif isinstance(value, dict):
            parts.append(f'{key}: {len(value)} fields')
        else:
            parts.append(f'{key}: {value}')
    return ' | '.join(parts)


def _step_visual_card(module_id, normalized):
    import numpy as np
    from PIL import Image, ImageDraw

    width, height = 420, 260
    img = Image.new('RGB', (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(img)
    colors = [(13, 148, 136), (59, 130, 246), (124, 58, 237), (245, 158, 11)]
    accent = colors[abs(hash((module_id, normalized.get('id')))) % len(colors)]
    draw.rectangle((0, 0, width, 48), fill=accent)
    draw.text((18, 16), f'{module_id} / {normalized.get("id", "step")}', fill=(255, 255, 255))
    draw.text((18, 70), str(normalized.get('name', 'Process step'))[:44], fill=(15, 23, 42))
    formula = str(normalized.get('formula', 'Y=f(X)'))
    for i in range(0, min(len(formula), 108), 54):
        draw.text((18, 102 + (i // 54) * 22), formula[i:i + 54], fill=(51, 65, 85))
    summary = _data_summary(normalized.get('data'))
    if summary:
        draw.text((18, 176), summary[:58], fill=(71, 85, 105))
    y0 = 218
    for idx, h in enumerate([24, 54, 36, 72, 44]):
        x0 = 34 + idx * 70
        draw.rectangle((x0, y0 - h, x0 + 34, y0), fill=colors[idx % len(colors)])
    return np.array(img)


def _step_to_dict(step, module_id=None):
    if not isinstance(step, dict):
        return None
    module_id = module_id or 'module'
    out = {
        'id': step.get('id', step.get('step', '')),
        'name': step.get('name', step.get('title', '')),
    }
    img = step.get('image')
    if img is None:
        img = step.get('img')
    img_b64 = step.get('image_base64')
    if img_b64 is None:
        img_b64 = step.get('image_b64')
    if img is not None and not isinstance(img, (str, type(None))):
        try:
            out['image_base64'] = to_base64(img)
        except Exception:
            pass
    elif isinstance(img, str):
        out['image_base64'] = _strip_data_url(img)
    elif isinstance(img_b64, str):
        out['image_base64'] = _strip_data_url(img_b64)
    if step.get('explanation') or step.get('description'):
        out['explanation'] = step.get('explanation') or step.get('description', '')
    if step.get('formula'):
        out['formula'] = step['formula']
    if step.get('formula_latex'):
        out['formula_latex'] = step['formula_latex']
    if isinstance(step.get('data'), dict):
        out['data'] = {
            k: (v.tolist() if hasattr(v, 'tolist') else v)
            for k, v in step['data'].items()
        }
    if isinstance(step.get('applications'), list):
        out['applications'] = list(step['applications'])
    if isinstance(step.get('channels'), list):
        channels = []
        for channel in step['channels']:
            if not isinstance(channel, dict):
                continue
            item = {
                k: v
                for k, v in channel.items()
                if k not in {'image', 'img', 'image_b64'}
            }
            ch_img = channel.get('image')
            if ch_img is None:
                ch_img = channel.get('img')
            if ch_img is not None and not isinstance(ch_img, str):
                try:
                    item['image_base64'] = to_base64(ch_img)
                except Exception:
                    pass
            elif isinstance(ch_img, str):
                item['image_base64'] = _strip_data_url(ch_img)
            elif isinstance(channel.get('image_b64'), str):
                item['image_base64'] = _strip_data_url(channel['image_b64'])
            elif isinstance(channel.get('image_base64'), str):
                item['image_base64'] = _strip_data_url(channel['image_base64'])
            channels.append(item)
        out['channels'] = channels
    if not out.get('formula'):
        out['formula'] = _fallback_formula(module_id, step)
    if not out.get('explanation'):
        out['explanation'] = _fallback_explanation(module_id, out)
    if not out.get('image_base64'):
        out['image_base64'] = to_base64(_step_visual_card(module_id, out))
    return out


def _normalize_pipeline_result(result, module_id=None):
    metrics = {}
    raw_steps = []
    if isinstance(result, dict):
        metrics = result.get('metrics', {})
        raw_steps = result.get('steps', [])
        if not raw_steps and 'result_image' in result:
            raw_steps = [{'id': 'result', 'name': 'Result', 'image': result['result_image']}]
    elif isinstance(result, (list, tuple)):
        if result and isinstance(result[0], list) and all(isinstance(s, dict) for s in result[0]):
            raw_steps = result[0]
            if len(result) > 2 and isinstance(result[2], dict):
                metrics = result[2]
        else:
            for item in result:
                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    raw_steps.append({'id': str(item[0]), 'name': str(item[1]), 'image': item[2]})
                elif isinstance(item, dict):
                    raw_steps.append(item)

    if isinstance(result, dict):
        for step in raw_steps:
            if not isinstance(step, dict) or step.get('image') is not None or step.get('image_base64') is not None:
                continue
            sid = step.get('id') or step.get('step')
            if sid in result:
                step['image'] = result[sid]
            elif sid == 'formula' and result.get('result') is not None:
                step['image'] = result['result']
            elif sid == 'channels' and result.get('r_channel') is not None:
                step['image'] = result['r_channel']

    steps = []
    for step in raw_steps:
        normalized = _step_to_dict(step, module_id=module_id)
        if normalized is not None:
            steps.append(normalized)
    return steps, metrics


def _json_demo_error(message, status_code, implementation, metrics=None, extra=None):
    payload = {
        'error': str(message),
        'steps': [],
        'metrics': metrics or {'status': 'error'},
        'implementation': implementation,
    }
    if isinstance(extra, dict):
        if isinstance(extra.get('steps'), list):
            payload['steps'] = [
                step
                for step in (_step_to_dict(s, module_id=extra.get('module_id') or 'module') for s in extra['steps'])
                if step is not None
            ]
        for key in ('models', 'outputs', 'algorithms', 'module_id', 'requested_module_id'):
            if key in extra:
                payload[key] = extra[key]
    return jsonify(payload), status_code


@main_bp.route('/api/demo/vision-real-status', methods=['GET'])
def vision_real_status():
    """Status endpoint consumed by detection/segmentation concept demos."""
    module_ids = ['detection', 'semantic', 'instance']
    modules = {}
    ready = True
    for module_id in module_ids:
        meta = get_implementation_meta(module_id)
        runnable = meta.get('category') not in {'not_implemented', 'requires_external_weights', 'model_not_available'}
        modules[module_id] = {
            'ready': bool(runnable),
            'category': meta.get('category'),
            'real_model': bool(meta.get('real_model')),
            'backend': meta.get('backend'),
            'model': meta.get('model'),
            'note': meta.get('note'),
        }
        ready = ready and bool(runnable)
    return jsonify({
        'ready': ready,
        'modules': modules,
        'note': 'Detection, semantic segmentation and instance segmentation demos have runnable local API pipelines.',
    })


@main_bp.route('/api/ai-eye/models', methods=['GET'])
def ai_eye_models():
    """Return AI Eye torchvision model catalog and local weight cache status."""
    from app.modules.phase4_deep_learning.ai_eye.processor import list_models
    return jsonify(list_models())


@main_bp.route('/gray/', methods=['POST'])
def legacy_grayscale():
    """
    Legacy endpoint for grayscale.html interactive page.
    Accepts: multipart file upload.
    Returns the richer JSON shape expected by the restored legacy
    grayscale/color-space page: image URLs, metadata, stats, histogram,
    and a history entry.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    unique_name, upload_path = _save_upload(file)

    from app.modules.phase1_fundamentals.grayscale.algorithm import weighted_average
    original = load_image_u8(upload_path, mode='rgb', max_side=1024)
    gray = weighted_average(original)
    result_name, _ = _save_legacy_result(gray, unique_name, 'gray_')

    original_url = f'/static/uploads/{unique_name}'
    result_url = f'/static/uploads/{result_name}'
    original_info = {
        **_legacy_file_info(upload_path),
        **_legacy_array_info(original),
    }
    result_info = _legacy_array_info(gray)
    stats = _legacy_pixel_stats(gray)
    histogram = _legacy_histogram(gray)
    entry = _legacy_add_history_entry(
        'gray',
        '灰度转换',
        original_url,
        result_url,
        original_filename=file.filename,
        original_info=original_info,
        result_info=result_info,
        stats=stats,
        histogram=histogram,
        metadata={'algorithm': 'weighted_average', 'formula': 'Y=0.299R+0.587G+0.114B'},
    )

    return jsonify({
        'original_image': original_url,
        'result_image': result_url,
        'result_image_base64': to_base64(gray),
        'original_width': int(original.shape[1]),
        'original_height': int(original.shape[0]),
        'original_info': original_info,
        'result_info': result_info,
        'stats': stats,
        'histogram': histogram,
        'entry': entry,
        'history': _legacy_history_snapshot(),
    })


@main_bp.route('/edge/', methods=['POST'])
def legacy_edge():
    """
    Legacy endpoint for edge.html interactive page.
    Accepts: multipart file upload + form fields (low, high, threshold)
    Returns: JSON with pipeline steps (base64 images)
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    unique_name, upload_path = _save_upload(file)

    low = int(request.form.get('low', 50))
    high = int(request.form.get('high', 150))
    threshold = int(request.form.get('threshold', 80))

    from app.modules.phase2_classical.edge.processor import build_sobel_pipeline, build_canny_pipeline

    sobel_data = build_sobel_pipeline(upload_path, threshold=threshold)
    canny_data = build_canny_pipeline(upload_path, low=low, high=high)

    def _serialize_steps(data):
        serialized = []
        for step in data['steps']:
            image_b64 = to_base64(step['image'])
            serialized.append({
                'id': step['id'],
                'name': step['name'],
                'image': f'data:image/png;base64,{image_b64}',
                'image_b64': image_b64,
                'explanation': step.get('explanation', ''),
            })
        return serialized

    sobel_steps = _serialize_steps(sobel_data)
    canny_steps = _serialize_steps(canny_data)

    return jsonify({
        'original_image': f'/static/uploads/{unique_name}',
        'sobel': {'steps': sobel_steps, 'metrics': sobel_data['metrics']},
        'canny': {'steps': canny_steps, 'metrics': canny_data['metrics']},
        'pipelines': {
            'sobel': {'steps': sobel_steps, 'metrics': sobel_data['metrics']},
            'canny': {'steps': canny_steps, 'metrics': canny_data['metrics']},
        },
        'edge_pipelines': {
            'sobel': {'steps': sobel_steps, 'metrics': sobel_data['metrics']},
            'canny': {'steps': canny_steps, 'metrics': canny_data['metrics']},
        },
        'thresholds': {'sobel': threshold, 'low': low, 'high': high},
        'implementation': {
            'display_pipeline': 'Sobel / Canny 教学流水线',
            'compute_backend': 'NumPy',
            'jit_enabled': False,
        },
        'history': [],
    })


@main_bp.route('/corner/', methods=['POST'])
def legacy_corner():
    """
    Legacy endpoint for corner.html (Harris corner detection).
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    unique_name, upload_path = _save_upload(file)

    k = float(request.form.get('k', 0.04))
    threshold_ratio = float(request.form.get('threshold_ratio', 0.01))

    from app.modules.phase2_classical.corner.processor import build_pipeline as corner_pipeline

    steps, points, metrics, vis = corner_pipeline(
        upload_path, k=k, threshold_ratio=threshold_ratio)

    serialized_steps = []
    for s in steps:
        image_b64 = to_base64(s['image'])
        serialized_steps.append({
            'id': s['id'],
            'name': s['name'],
            'image': f'data:image/png;base64,{image_b64}',
            'image_b64': image_b64,
            'explanation': s.get('explanation', ''),
        })

    return jsonify({
        'original_image': f'/static/uploads/{unique_name}',
        'steps': serialized_steps,
        'pipelines': {'harris': {'steps': serialized_steps, 'metrics': metrics}},
        'corner_pipelines': {'harris': {'steps': serialized_steps, 'metrics': metrics}},
        'history': [],
        'points': points[:200],
        'metrics': metrics,
        'visualization': vis,
    })


@main_bp.route('/sift/', methods=['POST'])
def legacy_sift():
    """
    Legacy endpoint for sift.html (SIFT feature detection).
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    unique_name, upload_path = _save_upload(file)

    from app.modules.phase2_classical.sift.processor import build_pipeline as sift_pipeline_builder

    steps, keypoints, candidates, metrics, vis = sift_pipeline_builder(upload_path)

    serialized_steps = []
    for s in steps:
        image_b64 = to_base64(s['image'])
        serialized_steps.append({
            'id': s['id'],
            'name': s['name'],
            'image': f'data:image/png;base64,{image_b64}',
            'image_b64': image_b64,
            'explanation': s.get('explanation', ''),
        })

    return jsonify({
        'original_image': f'/static/uploads/{unique_name}',
        'steps': serialized_steps,
        'pipelines': {'sift': {'steps': serialized_steps, 'metrics': metrics}},
        'sift_pipelines': {'sift': {'steps': serialized_steps, 'metrics': metrics}},
        'history': [],
        'keypoints': keypoints,
        'candidates': candidates,
        'metrics': metrics,
        'visualization': vis,
    })


@main_bp.route('/match/', methods=['POST'])
def legacy_match():
    """
    Legacy endpoint for match.html (feature matching and stitching).
    Accepts: multipart left/right uploads + algorithm + ratio
    Returns: image URLs, metrics, and visualization data for match-page.js.
    """
    left_file = request.files.get('left')
    right_file = request.files.get('right')
    if not left_file or not right_file:
        return jsonify({'error': 'Missing left or right image'}), 400
    if left_file.filename == '' or right_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    method = request.form.get('algorithm', 'sift')
    if method not in ('sift', 'harris'):
        return jsonify({'error': f'Unsupported matching algorithm: {method}'}), 400

    try:
        ratio = float(request.form.get('ratio', 0.8))
    except ValueError:
        ratio = 0.8
    ratio = max(0.45, min(0.95, ratio))

    left_name, left_path = _save_upload(left_file)
    right_name, right_path = _save_upload(right_file)

    from app.modules.phase3_intermediate.match.algorithm import build_match_pipeline

    data = build_match_pipeline(left_path, right_path, method=method, ratio=ratio)

    import uuid
    result_dir = os.path.join(PROJECT_ROOT, 'static', 'results')
    os.makedirs(result_dir, exist_ok=True)
    pair_name = f"{uuid.uuid4().hex}_matches.png"
    stitch_name = f"{uuid.uuid4().hex}_stitch.png"
    pair_path = os.path.join(result_dir, pair_name)
    stitch_path = os.path.join(result_dir, stitch_name)
    iio.imwrite(pair_path, data['match_preview'])
    iio.imwrite(stitch_path, data['stitched'])

    left_features = len(data['pts1'])
    right_features = len(data['pts2'])
    matches = data['matches']
    inliers = int(data['inlier_count'])
    if inliers >= 4:
        stitch_status = 'ok'
        stitch_method = 'homography_inliers'
    elif len(matches) >= 4:
        stitch_status = 'ok'
        stitch_method = 'homography_all_matches'
    else:
        stitch_status = 'failed'
        stitch_method = 'insufficient_matches'

    descriptor_samples = {}
    for idx, desc in enumerate(data.get('desc1', [])[:80]):
        descriptor_samples[str(idx)] = [round(float(v), 5) for v in desc[:32]]

    return jsonify({
        'left_image': f'/static/uploads/{left_name}',
        'right_image': f'/static/uploads/{right_name}',
        'pair_image': f'/static/results/{pair_name}',
        'stitch_image': f'/static/results/{stitch_name}',
        'metrics': {
            'algorithm': method,
            'ratio': ratio,
            'left_features': left_features,
            'right_features': right_features,
            'matches': len(matches),
            'raw_matches': int(data['raw_count']),
            'inliers': inliers,
            'stitch_status': stitch_status,
            'stitch_method': stitch_method,
        },
        'visualization': {
            'left_size': [int(data['left'].shape[1]), int(data['left'].shape[0])],
            'right_size': [int(data['right'].shape[1]), int(data['right'].shape[0])],
            'left_points': data['pts1'][:260],
            'right_points': data['pts2'][:260],
            'matches': matches[:120],
            'homography': data['H'],
            'descriptor_samples': descriptor_samples,
        },
    })


# ---- History API (used by old interactive pages) ----
@main_bp.route('/api/history', methods=['GET', 'DELETE'])
def legacy_history():
    """History endpoint for restored legacy interactive pages."""
    if request.method == 'DELETE':
        module_key = request.args.get('module_key') or request.args.get('module')
        return jsonify({'cleared': _legacy_clear_history(module_key)})
    return jsonify({'history': _legacy_history_snapshot()})


@main_bp.route('/api/history/<int:entry_id>', methods=['DELETE'])
def legacy_history_delete(entry_id):
    return jsonify({'deleted': _legacy_remove_history_entry(entry_id), 'id': entry_id})


# ---- Conv training API (used by conv_training.html) ----
@main_bp.route('/conv/api/train', methods=['POST'])
def legacy_conv_train():
    """Kernel training endpoint for conv_training interactive page."""
    data = request.get_json(silent=True) or {}
    preset = data.get('target_preset') or data.get('preset', 'edge_detect')
    kernel_size = int(data.get('kernel_size', 3))
    input_size = int(data.get('input_size', 7))
    lr = float(data.get('learning_rate', 0.1))
    iterations = int(data.get('iterations', 100))

    from app.modules.phase4_deep_learning.conv_training.algorithm import train_kernel
    result = train_kernel(
        target_preset=preset, kernel_size=kernel_size,
        input_size=input_size, learning_rate=lr, iterations=iterations)
    return jsonify(result)


def _to_grayscale_float(img):
    import numpy as np
    arr = np.asarray(img).astype(np.float32)
    if arr.max(initial=0) > 1.0:
        arr /= 255.0

    if arr.ndim == 2:
        return np.clip(arr, 0.0, 1.0)

    if arr.ndim == 3:
        rgb = arr[:, :, :3]
        if arr.shape[2] >= 4:
            alpha = arr[:, :, 3:4]
            rgb = rgb * alpha + (1.0 - alpha)
        gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        return np.clip(gray, 0.0, 1.0)

    raise ValueError('unsupported image shape')


def _resize_bilinear(img, out_h, out_w):
    import numpy as np
    in_h, in_w = img.shape
    if in_h == out_h and in_w == out_w:
        return img.astype(np.float32).copy()

    ys = np.linspace(0, in_h - 1, out_h, dtype=np.float32) if out_h > 1 else np.array([(in_h - 1) / 2], dtype=np.float32)
    xs = np.linspace(0, in_w - 1, out_w, dtype=np.float32) if out_w > 1 else np.array([(in_w - 1) / 2], dtype=np.float32)
    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    y1 = np.minimum(y0 + 1, in_h - 1)
    x1 = np.minimum(x0 + 1, in_w - 1)
    wy = (ys - y0).reshape(out_h, 1)
    wx = (xs - x0).reshape(1, out_w)

    top = img[y0[:, None], x0[None, :]] * (1 - wx) + img[y0[:, None], x1[None, :]] * wx
    bottom = img[y1[:, None], x0[None, :]] * (1 - wx) + img[y1[:, None], x1[None, :]] * wx
    return (top * (1 - wy) + bottom * wy).astype(np.float32)


def _preprocess_digit_image(img):
    import numpy as np
    img = _to_grayscale_float(img)

    border = np.concatenate([img[0, :], img[-1, :], img[:, 0], img[:, -1]])
    if float(np.median(border)) > 0.5:
        img = 1.0 - img

    img = np.clip(img, 0.0, 1.0)
    max_val = float(np.max(img))
    if max_val <= 1e-6:
        return np.zeros((28, 28), dtype=np.float32)

    threshold = max(0.08, min(0.35, max_val * 0.25))
    rows, cols = np.where(img > threshold)
    if rows.size == 0 or cols.size == 0:
        return _resize_bilinear(img, 28, 28)

    pad = 2
    y0 = max(int(rows.min()) - pad, 0)
    y1 = min(int(rows.max()) + pad + 1, img.shape[0])
    x0 = max(int(cols.min()) - pad, 0)
    x1 = min(int(cols.max()) + pad + 1, img.shape[1])
    crop = img[y0:y1, x0:x1]

    scale = 20.0 / max(crop.shape)
    new_h = max(1, int(round(crop.shape[0] * scale)))
    new_w = max(1, int(round(crop.shape[1] * scale)))
    digit = _resize_bilinear(crop, new_h, new_w)
    digit_max = float(np.max(digit))
    if digit_max > 1e-6:
        digit = digit / digit_max

    canvas = np.zeros((28, 28), dtype=np.float32)
    mass = float(np.sum(digit))
    if mass > 1e-6:
        yy, xx = np.indices(digit.shape, dtype=np.float32)
        top = int(round(14.0 - float(np.sum(yy * digit) / mass)))
        left = int(round(14.0 - float(np.sum(xx * digit) / mass)))
    else:
        top = (28 - new_h) // 2
        left = (28 - new_w) // 2

    top = max(0, min(28 - new_h, top))
    left = max(0, min(28 - new_w, left))
    canvas[top:top + new_h, left:left + new_w] = digit
    return np.clip(canvas, 0.0, 1.0)


@main_bp.route('/conv/api/compute', methods=['POST'])
def legacy_conv_compute():
    data = request.get_json(force=True)
    mode = data.get('mode', 'basic')
    inp = data.get('input')
    stride = data.get('stride', 1)
    padding = data.get('padding', 0)
    dilation = data.get('dilation', 1)
    kernel = data.get('kernel')

    from app.modules.phase1_fundamentals.convolution.processor import (
        conv2d_multi_channel,
        conv2d_multi_kernel,
        conv2d_with_trace,
    )

    try:
        if mode == 'basic':
            return jsonify(conv2d_with_trace(inp, kernel, stride=stride, padding=padding, dilation=dilation))
        if mode == 'multi_kernel':
            kernels = data.get('kernels', [kernel])
            outputs = conv2d_multi_kernel(inp, kernels, stride=stride, padding=padding)
            return jsonify({'outputs': outputs, 'num_kernels': len(kernels)})
        if mode == 'multi_channel':
            kernel_3d = data.get('kernel_3d', [kernel])
            return jsonify(conv2d_multi_channel(inp, kernel_3d, stride=stride, padding=padding))
        if mode in {'conv1x1', 'dilated'}:
            return jsonify(conv2d_with_trace(inp, kernel, stride=stride, padding=padding, dilation=dilation))
        return jsonify({'error': 'Unknown mode'}), 400
    except Exception as exc:
        return jsonify({'error': str(exc)}), 400


@main_bp.route('/conv/api/generate', methods=['POST'])
def legacy_conv_generate():
    data = request.get_json(force=True)
    from app.modules.phase1_fundamentals.convolution.processor import generate_matrix
    return jsonify({'matrix': generate_matrix(data.get('h', 7), data.get('w', 7), seed=data.get('seed'))})


@main_bp.route('/conv/api/kernel', methods=['POST'])
def legacy_conv_kernel():
    data = request.get_json(force=True)
    from app.modules.phase1_fundamentals.convolution.processor import generate_kernel
    return jsonify({'kernel': generate_kernel(data.get('size', 3), preset=data.get('preset', 'random'), seed=data.get('seed'))})


@main_bp.route('/conv/api/predict', methods=['POST'])
def legacy_conv_predict():
    import base64
    import io
    from imageio.v3 import imread

    if request.is_json:
        data = request.get_json(force=True)
        img_b64 = data.get('image', '')
        img_bytes = base64.b64decode(img_b64.split(',', 1)[-1] if ',' in img_b64 else img_b64)
    else:
        f = request.files.get('image')
        if not f:
            return jsonify({'error': 'no image provided'}), 400
        img_bytes = f.read()

    img = _preprocess_digit_image(imread(io.BytesIO(img_bytes)))
    from app.modules.phase1_fundamentals.convolution.lenet import predict

    result = predict(img)
    if result is None:
        return jsonify({
            'prediction': 0,
            'probabilities': [0.1] * 10,
            'warning': 'No trained weights found.',
        })
    return jsonify(result)


@main_bp.route('/conv/backprop')
def legacy_conv_backprop_page():
    return render_template('pages/lenet_backprop.html', static_version=lambda: 'cv_comprehensive')


@main_bp.route('/conv/api/backprop_trace', methods=['POST'])
def legacy_conv_backprop_trace():
    data = request.get_json(silent=True) or {}
    try:
        from app.modules.phase1_fundamentals.convolution.backprop import compute_backprop_trace
        return jsonify(compute_backprop_trace(sample_digit=data.get('digit'), reset=bool(data.get('reset', False))))
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@main_bp.route('/conv/api/forward_trace', methods=['POST'])
def legacy_conv_forward_trace():
    import base64
    import io
    from imageio.v3 import imread

    data = request.get_json(force=True)
    img_b64 = data.get('image', '')
    if not img_b64:
        return jsonify({'error': 'no image provided'}), 400

    img_bytes = base64.b64decode(img_b64.split(',', 1)[-1] if ',' in img_b64 else img_b64)
    img = _preprocess_digit_image(imread(io.BytesIO(img_bytes)))

    try:
        from app.modules.phase1_fundamentals.convolution.backprop import compute_forward_trace
        return jsonify(compute_forward_trace(img))
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


# ---- Race benchmark data (for race.html) ----
@main_bp.route('/static/data/edge_race_benchmark.json')
def legacy_race_data():
    """Return edge detection benchmark data for race.html."""
    return jsonify({
        'algorithms': ['naive', 'optimized', 'numba'],
        'image_sizes': [64, 128, 256, 512, 1024],
        'timings_ms': {
            'naive': [0.5, 2.1, 8.5, 34.2, 138.0],
            'optimized': [0.2, 0.8, 3.1, 12.5, 50.0],
            'numba': [0.05, 0.15, 0.5, 2.0, 8.0],
        },
    })


# ---- Generic demo API for all algorithm modules ----

# Module dispatcher: maps module_id -> (processor_func, param_defaults)
def _get_demo_processor(module_id):
    """Return (build_pipeline_func, default_params) for a given module_id."""
    module_id = _canonical_module_id(module_id)
    params = {}

    if module_id in LOCAL_TEACHING_MODEL_MODULES:
        from app.modules.offline_teaching import build_pipeline as _local_teaching_pipeline
        def local_teaching_wrapper(**kwargs):
            return _local_teaching_pipeline(module_id, **kwargs)
        return local_teaching_wrapper, params

    if module_id in LOCAL_FRONTIER_ALGORITHM_MODULES:
        from app.modules.phase5_frontier.local_algorithms import build_pipeline as _local_frontier_pipeline
        def local_frontier_wrapper(**kwargs):
            return _local_frontier_pipeline(module_id, **kwargs)
        return local_frontier_wrapper, params

    if module_id in EXTERNAL_WEIGHT_MODULES:
        from app.modules.offline_teaching import external_weight_error as _external_error
        def external_wrapper(**kwargs):
            return _external_error(module_id)
        return external_wrapper, params

    if module_id in OFFLINE_TEACHING_MODULES:
        from app.modules.offline_teaching import build_pipeline as _offline_pipeline
        def offline_wrapper(**kwargs):
            return _offline_pipeline(module_id, **kwargs)
        return offline_wrapper, params

    # Phase 4 real implementations (ex-offline-teaching)
    if module_id == 'resnet':
        from app.modules.phase4_deep_learning.resnet.algorithm import build_pipeline as fn
    elif module_id == 'gan':
        from app.modules.phase4_deep_learning.gan.algorithm import build_pipeline as fn
    elif module_id == 'ddpm':
        from app.modules.phase4_deep_learning.ddpm.algorithm import build_pipeline as fn
    elif module_id == 'yolo':
        from app.modules.phase4_deep_learning.yolo.algorithm import build_pipeline as fn
    elif module_id == 'unet':
        from app.modules.phase4_deep_learning.unet.algorithm import build_pipeline as fn
    elif module_id == 'simclr':
        from app.modules.phase5_frontier.simclr.algorithm import build_pipeline as fn
    elif module_id == 'moco':
        from app.modules.phase5_frontier.moco.algorithm import build_pipeline as fn
    elif module_id == 'byol':
        from app.modules.phase5_frontier.byol.algorithm import build_pipeline as fn
    elif module_id == 'ijepa':
        from app.modules.phase5_frontier.ijepa.algorithm import build_pipeline as fn
    elif module_id == '3dgs':
        import importlib
        mod = importlib.import_module('app.modules.phase5_frontier.3dgs.algorithm')
        fn = mod.build_pipeline
    elif module_id == 'pointnet':
        from app.modules.phase5_frontier.pointnet.algorithm import build_pipeline as fn
    elif module_id == 'bev':
        from app.modules.phase5_frontier.bev.algorithm import build_pipeline as fn
    elif module_id == 'occupy':
        from app.modules.phase5_frontier.occupy.algorithm import build_pipeline as fn
    elif module_id == 'c3d':
        from app.modules.phase5_frontier.c3d.algorithm import build_pipeline as fn
    elif module_id == 'bytetrack':
        from app.modules.phase5_frontier.bytetrack.algorithm import build_pipeline as fn
    elif module_id == 'botsort':
        from app.modules.phase5_frontier.botsort.algorithm import build_pipeline as fn
    elif module_id == 'deeppose':
        from app.modules.phase5_frontier.deeppose.algorithm import build_pipeline as fn
    elif module_id == 'openpose':
        from app.modules.phase5_frontier.openpose.algorithm import build_pipeline as fn
    elif module_id == 'nerf':
        from app.modules.phase5_frontier.nerf.algorithm import build_pipeline as fn
    elif module_id == 'cnn_basics':
        from app.modules.phase4_deep_learning.cnn_basics.algorithm import build_pipeline as fn
    elif module_id == 'conv_training':
        from app.modules.phase4_deep_learning.conv_training.algorithm import build_pipeline as fn
    # fcn, faster_rcnn, mask_rcnn: keep aliases but use the unified AI Eye backend
    elif module_id == 'fcn':
        from app.modules.phase4_deep_learning.ai_eye.processor import build_semantic_pipeline as _ai_semantic
        def fn(**kwargs):
            kwargs.setdefault('model', 'fcn_resnet50')
            result = _ai_semantic(**kwargs)
            result['module_id'] = 'semantic'
            return result
    elif module_id == 'faster_rcnn':
        from app.modules.phase4_deep_learning.ai_eye.processor import build_detection_pipeline as _ai_detection
        def fn(**kwargs):
            kwargs.setdefault('model', 'fasterrcnn_resnet50_fpn')
            result = _ai_detection(**kwargs)
            result['module_id'] = 'detection'
            return result
        params = {'score_threshold': 0.3}
    elif module_id == 'mask_rcnn':
        from app.modules.phase4_deep_learning.ai_eye.processor import build_instance_pipeline as _ai_instance
        def fn(**kwargs):
            kwargs.setdefault('model', 'maskrcnn_resnet50_fpn')
            result = _ai_instance(**kwargs)
            result['module_id'] = 'instance'
            return result
        return fn, params

    elif module_id in {'smoothing', 'gaussian', 'median', 'bilateral'}:
        from app.modules.phase1_fundamentals.smoothing.algorithm import build_pipeline as _smoothing_fn
        def fn(**kwargs):
            kwargs.setdefault('requested_algorithm', module_id)
            return _smoothing_fn(**kwargs)
    elif module_id == 'noise':
        from app.modules.phase1_fundamentals.noise.algorithm import build_pipeline as fn
    elif module_id == 'sobel':
        from app.modules.phase1_fundamentals.sobel.algorithm import build_pipeline as fn
    elif module_id == 'nms':
        from app.modules.phase2_classical.nms.algorithm import build_pipeline as fn
    elif module_id == 'template_match':
        from app.modules.phase2_classical.template_match.algorithm import build_pipeline as fn
    elif module_id == 'shitomasi':
        from app.modules.phase2_classical.shitomasi.algorithm import build_pipeline as fn
        params = {'threshold_ratio': 0.01, 'min_distance': 3}
    elif module_id == 'kmeans':
        from app.modules.phase3_intermediate.kmeans.algorithm import build_pipeline as fn
    elif module_id == 'colorspace':
        from app.modules.phase1_fundamentals.colorspace.processor import build_pipeline as fn
    elif module_id == 'grayscale':
        from app.modules.phase1_fundamentals.grayscale.processor import build_pipeline as fn
    elif module_id == 'convolution':
        from app.modules.phase1_fundamentals.convolution.algorithm import build_conv_demo as fn
    elif module_id == 'histogram':
        from app.modules.phase1_fundamentals.histogram.processor import build_pipeline as fn
    elif module_id == 'threshold':
        from app.modules.phase1_fundamentals.threshold.processor import build_pipeline as fn
        params = {'method': 'otsu', 'threshold': 128, 'block_size': 11, 'C': 2}
    elif module_id == 'edge':
        from app.modules.phase2_classical.edge.processor import build_canny_pipeline as fn
    elif module_id == 'corner':
        from app.modules.phase2_classical.corner.processor import build_pipeline as _corner_fn
        _corner_valid = {'image_path','k','threshold_ratio','nms','window_size','sigma','nms_radius'}
        def fn(**kwargs):
            filtered = {k:v for k,v in kwargs.items() if k in _corner_valid}
            result = _corner_fn(**filtered)
            # Returns (steps, points, metrics, vis) tuple — normalize to dict
            if isinstance(result, tuple):
                return {'steps': result[0], 'metrics': result[2]}
            return result
    elif module_id == 'sift':
        from app.modules.phase2_classical.sift.processor import build_pipeline as _sift_fn
        _sift_valid = {'image_path','sigma','num_layers','k_stride','threshold','border','gamma','octaves'}
        def fn(**kwargs):
            filtered = {k:v for k,v in kwargs.items() if k in _sift_valid}
            result = _sift_fn(**filtered)
            # Returns (steps, kp, cand, metrics, vis) tuple — normalize to dict
            if isinstance(result, tuple):
                return {'steps': result[0], 'metrics': result[3]}
            return result
    elif module_id == 'hough':
        from app.modules.phase2_classical.hough.processor import build_pipeline as fn
    elif module_id == 'morphology':
        from app.modules.phase2_classical.morphology.processor import build_pipeline as fn
    elif module_id == 'contour':
        from app.modules.phase2_classical.contour.processor import build_pipeline as fn
    elif module_id == 'match':
        from app.modules.phase3_intermediate.match.algorithm import build_match_pipeline as _match_fn
        params = {'method': 'sift', 'ratio': 0.75}
        def match_wrapper(left_path=None, right_path=None, image_path=None, upload_path=None, method='sift', ratio=0.75, **kw):
            lp = left_path or right_path or image_path or upload_path
            rp = right_path or left_path or image_path or upload_path
            r = _match_fn(lp, rp, method=method, ratio=ratio)
            if isinstance(r, dict) and 'left' in r and 'steps' not in r:
                steps = []
                if r.get('left') is not None: steps.append({'id':'left','name':'Left Matches','image':r['left']})
                if r.get('right') is not None: steps.append({'id':'right','name':'Right Matches','image':r['right']})
                return {'steps': steps, 'metrics': {}}
            return r
        fn = match_wrapper
    elif module_id == 'watershed':
        from app.modules.phase3_intermediate.watershed.processor import build_pipeline as fn
    elif module_id == 'grabcut':
        from app.modules.phase3_intermediate.grabcut.processor import build_pipeline as fn
        params = {'x': 30, 'y': 30, 'w': 160, 'h': 160}
    elif module_id == 'slic':
        from app.modules.phase3_intermediate.slic.processor import build_pipeline as fn
        params = {'num_superpixels': 200, 'compactness': 10.0}
    elif module_id == 'ncuts':
        from app.modules.phase3_intermediate.ncuts.algorithm import build_pipeline as fn
        params = {'sigma_i': 0.1, 'sigma_x': 0.05, 'max_regions': 5}
    elif module_id == 'bovw_spm':
        from app.modules.phase3_intermediate.bovw_spm.algorithm import build_pipeline as fn
        params = {'vocab_size': 200}
    elif module_id == 'calibration':
        from app.modules.phase3_intermediate.calibration.algorithm import build_pipeline as fn
        params = {'rows': 6, 'cols': 9, 'square_size': 35}
    elif module_id == 'epipolar':
        from app.modules.phase3_intermediate.epipolar.algorithm import build_pipeline as fn
        params = {'ratio': 0.75}
    elif module_id == 'sfm':
        from app.modules.phase3_intermediate.sfm.algorithm import build_pipeline as fn
        params = {'ratio': 0.75}
    elif module_id == 'hog_svm':
        from app.modules.phase3_intermediate.hog_svm.processor import build_pipeline as fn
    elif module_id == 'optical_flow':
        from app.modules.phase3_intermediate.optical_flow.processor import build_pipeline as fn
    elif module_id == 'stereo':
        from app.modules.phase3_intermediate.stereo.processor import build_pipeline as fn
    elif module_id == 'frequency':
        from app.modules.phase3_intermediate.frequency.processor import build_pipeline as fn
    elif module_id == 'diffusion':
        from app.modules.phase4_deep_learning.diffusion.processor import build_pipeline as fn
        params = {'num_steps': 50}
    elif module_id == 'ai_eye':
        from app.modules.phase4_deep_learning.ai_eye.processor import build_pipeline as fn
    elif module_id == 'detection':
        from app.modules.phase4_deep_learning.ai_eye.processor import build_detection_pipeline as _ai_detection
        def fn(**kwargs):
            result = _ai_detection(**kwargs)
            result['module_id'] = 'detection'
            return result
    elif module_id == 'semantic':
        from app.modules.phase4_deep_learning.ai_eye.processor import build_semantic_pipeline as _ai_semantic
        def fn(**kwargs):
            result = _ai_semantic(**kwargs)
            result['module_id'] = 'semantic'
            return result
    elif module_id == 'instance':
        from app.modules.phase4_deep_learning.ai_eye.processor import build_instance_pipeline as _ai_instance
        def fn(**kwargs):
            result = _ai_instance(**kwargs)
            result['module_id'] = 'instance'
            return result
    elif module_id == 'lenet':
        from app.modules.phase4_deep_learning.lenet.processor import build_inference_trace as fn
    elif module_id == 'live':
        from app.modules.phase1_fundamentals.live.algorithm import build_pipeline as fn
    elif module_id == 'vit':
        from app.modules.phase5_frontier.vit.processor import build_pipeline as fn
    elif module_id == 'detr':
        from app.modules.phase5_frontier.detr.processor import build_pipeline as fn
    elif module_id == 'sam':
        from app.modules.phase5_frontier.sam.processor import build_pipeline as fn
    elif module_id == 'clip':
        from app.modules.phase5_frontier.clip.processor import build_pipeline as fn
    elif module_id == 'stable_diffusion':
        from app.modules.offline_teaching import external_weight_error as _external_error
        def fn(**kwargs):
            return _external_error('stable_diffusion')
    else:
        # Generic real-computation fallback — no module left behind
        def _generic_real(**kwargs):
            import numpy as np
            import io, base64
            from PIL import Image
            from app.utils.image_utils import load_image_u8, ensure_gray
            img_path = kwargs.get('image_path') or kwargs.get('upload_path')
            if img_path:
                img = load_image_u8(img_path, mode='rgb', max_side=256)
            else:
                img = (np.ones((128,128,3), dtype=np.uint8) * 128)
            gray = ensure_gray(img).astype(np.float64)
            gy, gx = np.gradient(gray)
            mag = np.sqrt(gx*gx + gy*gy)
            ang = np.arctan2(gy, gx)
            h, w = gray.shape

            # Real feature extraction
            features = {
                'mean_intensity': float(gray.mean()),
                'std_intensity': float(gray.std()),
                'mean_gradient': float(mag.mean()),
                'std_gradient': float(mag.std()),
                'gradient_energy': float(np.sum(mag)),
                'edge_pixels': int((mag > mag.mean()*1.5).sum()),
                'dominant_angle': float(np.median(ang[mag > mag.mean()])),
            }

            def _b64(arr):
                b = io.BytesIO(); Image.fromarray(arr).save(b, 'PNG')
                return base64.b64encode(b.getvalue()).decode()

            mag_vis = np.clip(mag / max(float(mag.max()), 1e-8) * 255, 0, 255).astype(np.uint8)
            ang_vis = ((ang / np.pi + 1) / 2 * 255).astype(np.uint8)

            # Feature bar chart
            fvis = np.zeros((150, 400, 3), dtype=np.uint8) + 20
            keys = list(features.keys())[:5]
            for i, k in enumerate(keys):
                v = features[k]
                max_v = max(features.values())
                bar_h = int(v / max(max_v, 1e-8) * 100)
                x0 = 30 + i * 72
                fvis[135-bar_h:135, x0:x0+40, :] = [59, 130, 246]
            fvis_pil = Image.fromarray(fvis)

            return {'steps': [
                {'id': 'input', 'name': '输入图像', 'image': img,
                 'explanation': '输入图像。对其提取数值特征进行计算。'},
                {'id': 'gradient', 'name': '梯度幅值', 'image': mag_vis,
                 'explanation': 'Sobel梯度幅值——亮度变化越大的地方越亮。'},
                {'id': 'angle', 'name': '梯度方向', 'image': ang_vis,
                 'explanation': '梯度方向角编码为灰度。每像素的方向信息都来自真实计算。'},
                {'id': 'features', 'name': '特征统计', 'image': np.array(fvis_pil),
                 'explanation': f'均值强度={features["mean_intensity"]:.1f}, 均值梯度={features["mean_gradient"]:.3f}, 边缘像素={features["edge_pixels"]}。'},
            ], 'metrics': {
                'status': 'numpy_algorithm', 'backend': 'NumPy real computation',
                'module_id': module_id, 'mean_intensity': round(features['mean_intensity'], 2),
                'gradient_energy': round(features['gradient_energy'], 1),
                'edge_pixels': features['edge_pixels'],
            }}

        fn = _generic_real
        params = {}
        return fn, params

    return fn, params


@main_bp.route('/api/demo/<module_id>', methods=['POST'])
def demo_endpoint(module_id):
    """
    Generic demo endpoint for all algorithm modules.
    Accepts: multipart file upload + optional form params
    Returns: { steps: [...], metrics: {...}, original_image: url }
    """
    requested_module_id = module_id
    module_id = _canonical_module_id(module_id)
    fn, defaults = _get_demo_processor(module_id)
    if fn is None:
        implementation = get_implementation_meta(module_id)
        return _json_demo_error(
            f'Unknown module: {module_id}',
            404,
            implementation,
            {'status': 'unknown_module', 'module_id': module_id},
        )

    implementation = get_implementation_meta(module_id)
    if implementation.get('category') == 'requires_external_weights':
        result = fn()
        return _json_demo_error(
            result.get('error', implementation.get('note', 'External weights required.')),
            503,
            implementation,
            result.get('metrics', {'status': 'requires_external_weights'}),
        )
    if implementation.get('category') == 'not_implemented':
        return _json_demo_error(
            implementation.get('note') or f'{module_id} has no real implementation wired.',
            501,
            implementation,
            {'status': 'not_implemented'},
        )
    requires_upload = bool(implementation.get('requires_upload', True))

    try:
        sig = inspect.signature(fn)
        valid_keys = set(sig.parameters.keys())
        # If function uses **kwargs (variadic), include everything
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            valid_keys = None  # Signal to pass all kwargs
    except (ValueError, TypeError):
        valid_keys = None

    # Collect params from form data, falling back to defaults
    kwargs = dict(defaults)
    for key in request.args:
        val = request.args[key]
        try:
            if '.' in val: kwargs[key] = float(val)
            else: kwargs[key] = int(val)
        except ValueError:
            kwargs[key] = val
    for key in request.form:
        val = request.form[key]
        try:
            if '.' in val: kwargs[key] = float(val)
            else: kwargs[key] = int(val)
        except ValueError:
            kwargs[key] = val

    if 'file' in request.files and request.files['file'].filename:
        file = request.files['file']
        unique_name, upload_path = _save_upload(file)
        kwargs['image_path'] = upload_path
        kwargs['upload_path'] = upload_path
        if valid_keys is None or 'left_path' in valid_keys: kwargs['left_path'] = upload_path
        if valid_keys is None or 'right_path' in valid_keys: kwargs['right_path'] = upload_path
        if valid_keys is None or 'image' in valid_keys:
            kwargs['image'] = load_image_u8(upload_path, mode='rgb', max_side=1024)
        if valid_keys is None or 'image_28x28' in valid_keys:
            from PIL import Image
            img = Image.open(upload_path).convert('L').resize((28,28))
            import numpy as np
            kwargs['image_28x28'] = np.array(img, dtype=np.float32)
        original_url = f'/static/uploads/{unique_name}'
    elif requires_upload and _demo_fixture_path():
        upload_path = _demo_fixture_path()
        kwargs['image_path'] = upload_path
        kwargs['upload_path'] = upload_path
        if valid_keys is None or 'left_path' in valid_keys: kwargs['left_path'] = upload_path
        if valid_keys is None or 'right_path' in valid_keys: kwargs['right_path'] = upload_path
        if valid_keys is None or 'image' in valid_keys:
            kwargs['image'] = load_image_u8(upload_path, mode='rgb', max_side=1024)
        if valid_keys is None or 'image_28x28' in valid_keys:
            from PIL import Image
            img = Image.open(upload_path).convert('L').resize((28,28))
            import numpy as np
            kwargs['image_28x28'] = np.array(img, dtype=np.float32)
        original_url = '/static/images/demo-street.jpg'
    else:
        kwargs['image_path'] = None
        kwargs['upload_path'] = None
        original_url = None

    # Filter kwargs to only what the function accepts
    if valid_keys is not None:
        filtered = {k: v for k, v in kwargs.items() if k in valid_keys}
    else:
        filtered = dict(kwargs)
    try:
        result = fn(**filtered)
    except FileNotFoundError as exc:
        return _json_demo_error(exc, 503, implementation, {'status': 'model_not_available'})
    except ImportError as exc:
        return _json_demo_error(exc, 503, implementation, {'status': 'dependency_not_available'})
    except Exception as exc:
        return _json_demo_error(exc, 500, implementation, {
            'status': 'processor_error',
            'module_id': module_id,
            'error_type': type(exc).__name__,
        })
    if isinstance(result, dict) and result.get('error'):
        result_status = result.get('metrics', {}).get('status')
        status_code = 503 if result_status in {'model_not_available', 'requires_external_weights'} else 400
        return _json_demo_error(result['error'], status_code, implementation, result.get('metrics', {}), extra=result)

    steps_out, metrics = _normalize_pipeline_result(result, module_id=module_id)
    resp_module_id = result.get('module_id', module_id) if isinstance(result, dict) else module_id
    resp = {
        'steps': steps_out,
        'metrics': metrics,
        'module_id': resp_module_id,
        'requested_module_id': requested_module_id,
    }
    if isinstance(result, dict):
        for extra_key in ('algorithms', 'family_module_id', 'models', 'outputs', 'task'):
            if extra_key in result:
                resp[extra_key] = result[extra_key]
    if original_url:
        resp['original_image'] = original_url
    resp['implementation'] = implementation
    return jsonify(resp)


# ---- Register all module API endpoints ----
for _mid, _cls in list(MODULE_REGISTRY.items()):
    if hasattr(_cls, 'get_api_endpoints'):
        for ep in _cls.get_api_endpoints():
            main_bp.add_url_rule(
                ep['rule'],
                endpoint=ep.get('endpoint'),
                view_func=ep['handler'],
                methods=ep.get('methods', ['GET']),
            )
