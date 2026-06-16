"""
Route hub.
- Root path returns the main shell page
- /api/modules returns module registry info (for frontend navigation)
- Legacy API routes for interactive algorithm pages (compatible with old CV project)
"""
import os
import inspect
import imageio.v3 as iio
from flask import Blueprint, render_template, jsonify, request
from app.modules import MODULE_REGISTRY, get_modules_by_phase
from app.utils.image_utils import to_base64, load_image_u8

main_bp = Blueprint('main', __name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEACHING_PAGE_IDS = {
    'grayscale', 'histogram', 'threshold', 'noise', 'gaussian', 'sobel', 'median', 'bilateral',
    'hough', 'morphology', 'contour', 'nms', 'template_match',
    'kmeans', 'watershed', 'grabcut', 'slic', 'hog_svm', 'optical_flow', 'stereo',
}


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
    for phase in phases:
        for mod in phase['modules']:
            cls = MODULE_REGISTRY.get(mod['id'])
            if cls and hasattr(cls, 'get_page'):
                if mod['id'] in TEACHING_PAGE_IDS:
                    mod['page'] = f'teaching.html?id={mod["id"]}'
                else:
                    mod['page'] = cls.get_page()

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


def _to_data_url(image):
    return f'data:image/png;base64,{to_base64(image)}'


def _step_to_dict(step):
    if not isinstance(step, dict):
        return None
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
    elif isinstance(img_b64, str):
        out['image_base64'] = img_b64
    if step.get('explanation') or step.get('description'):
        out['explanation'] = step.get('explanation') or step.get('description', '')
    if step.get('formula'):
        out['formula'] = step['formula']
    if isinstance(step.get('data'), dict):
        out['data'] = {
            k: (v.tolist() if hasattr(v, 'tolist') else v)
            for k, v in step['data'].items()
        }
    return out


def _normalize_pipeline_result(result):
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
        normalized = _step_to_dict(step)
        if normalized is not None:
            steps.append(normalized)
    return steps, metrics


@main_bp.route('/gray/', methods=['POST'])
def legacy_grayscale():
    """
    Legacy endpoint for grayscale.html interactive page.
    Accepts: multipart file upload
    Returns: JSON with original and result images (base64 PNG)
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

    return jsonify({
        'original_image': f'/static/uploads/{unique_name}',
        'result_image_base64': to_base64(gray),
        'original_width': int(original.shape[1]),
        'original_height': int(original.shape[0]),
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
    """Minimal history endpoint for old interactive pages."""
    if request.method == 'DELETE':
        return jsonify({'cleared': 0})
    return jsonify({'history': []})


@main_bp.route('/api/history/<int:entry_id>', methods=['DELETE'])
def legacy_history_delete(entry_id):
    return jsonify({'deleted': entry_id})


# ---- Conv training API (used by conv_training.html) ----
@main_bp.route('/conv/api/train', methods=['POST'])
def legacy_conv_train():
    """Kernel training endpoint for conv_training interactive page."""
    data = request.get_json(silent=True) or {}
    preset = data.get('preset', 'edge_detect')
    kernel_size = int(data.get('kernel_size', 3))
    input_size = int(data.get('input_size', 7))
    lr = float(data.get('learning_rate', 0.1))
    iterations = int(data.get('iterations', 100))

    from app.modules.phase1_fundamentals.convolution.algorithm import train_kernel
    result = train_kernel(preset, kernel_size, input_size, lr=lr, iterations=iterations)
    return jsonify(result)


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
    params = {}

    if module_id == 'noise':
        from app.modules.phase1_fundamentals.noise.algorithm import build_pipeline as fn
    elif module_id == 'gaussian':
        from app.modules.phase1_fundamentals.gaussian.algorithm import build_pipeline as fn
    elif module_id == 'sobel':
        from app.modules.phase1_fundamentals.sobel.algorithm import build_pipeline as fn
    elif module_id == 'median':
        from app.modules.phase1_fundamentals.median.algorithm import build_pipeline as fn
    elif module_id == 'bilateral':
        from app.modules.phase1_fundamentals.bilateral.algorithm import build_pipeline as fn
    elif module_id == 'nms':
        from app.modules.phase2_classical.nms.algorithm import build_pipeline as fn
    elif module_id == 'template_match':
        from app.modules.phase2_classical.template_match.algorithm import build_pipeline as fn
    elif module_id == 'kmeans':
        from app.modules.phase3_intermediate.kmeans.algorithm import build_pipeline as fn
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
        from app.modules.phase2_classical.corner.processor import build_pipeline as fn
    elif module_id == 'sift':
        from app.modules.phase2_classical.sift.processor import build_pipeline as fn
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
    elif module_id == 'hog_svm':
        from app.modules.phase3_intermediate.hog_svm.processor import build_pipeline as fn
    elif module_id == 'optical_flow':
        from app.modules.phase3_intermediate.optical_flow.processor import build_pipeline as fn
    elif module_id == 'stereo':
        from app.modules.phase3_intermediate.stereo.processor import build_pipeline as fn
    elif module_id == 'frequency':
        from app.modules.phase3_intermediate.frequency.processor import build_pipeline as fn
    elif module_id == 'gan':
        from app.modules.phase4_deep_learning.gan.processor import build_pipeline as fn
        params = {'noise_dim': 10, 'steps': 50}
    elif module_id == 'diffusion':
        from app.modules.phase4_deep_learning.diffusion.processor import build_pipeline as fn
        params = {'num_steps': 50}
    elif module_id == 'detection':
        from app.modules.phase4_deep_learning.detection.processor import build_pipeline as fn
    elif module_id == 'semantic':
        from app.modules.phase4_deep_learning.semantic.processor import build_pipeline as fn
        params = {'num_classes': 5}
    elif module_id == 'instance':
        from app.modules.phase4_deep_learning.instance.processor import build_pipeline as fn
        params = {'num_instances': 3}
    elif module_id == 'lenet':
        from app.modules.phase4_deep_learning.lenet.processor import build_inference_trace as fn
    else:
        return None, None

    return fn, params


@main_bp.route('/api/demo/<module_id>', methods=['POST'])
def demo_endpoint(module_id):
    """
    Generic demo endpoint for all algorithm modules.
    Accepts: multipart file upload + optional form params
    Returns: { steps: [...], metrics: {...}, original_image: url }
    """
    fn, defaults = _get_demo_processor(module_id)
    if fn is None:
        return jsonify({'error': f'Unknown module: {module_id}'}), 404

    if 'file' not in request.files and module_id not in ('gan', 'diffusion', 'detection', 'conv_training'):
        return jsonify({'error': 'No file part'}), 400

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
        if valid_keys is None or 'image_28x28' in valid_keys:
            from PIL import Image
            img = Image.open(upload_path).convert('L').resize((28,28))
            import numpy as np
            kwargs['image_28x28'] = np.array(img, dtype=np.float32)
        original_url = f'/static/uploads/{unique_name}'
    else:
        kwargs['image_path'] = None
        kwargs['upload_path'] = None
        original_url = None

    # Filter kwargs to only what the function accepts
    if valid_keys is not None:
        filtered = {k: v for k, v in kwargs.items() if k in valid_keys}
    else:
        filtered = dict(kwargs)
    result = fn(**filtered)

    steps_out, metrics = _normalize_pipeline_result(result)
    resp = {'steps': steps_out, 'metrics': metrics}
    if original_url:
        resp['original_image'] = original_url
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
