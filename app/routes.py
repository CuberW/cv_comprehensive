"""
Route hub.
- Root path returns the main shell page
- /api/modules returns module registry info (for frontend navigation)
- Legacy API routes for interactive algorithm pages (compatible with old CV project)
"""
import os
import numpy as np
import imageio.v3 as iio
from flask import Blueprint, render_template, jsonify, request, current_app, send_file
from app.modules import MODULE_REGISTRY, get_modules_by_phase
from app.utils.image_utils import to_base64

main_bp = Blueprint('main', __name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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

    from app.modules.phase1_fundamentals.grayscale.algorithm import weighted_average, to_uint8
    img = iio.imread(upload_path)
    original = to_uint8(img)
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

    # Convert step images to base64
    sobel_steps = []
    for step in sobel_data['steps']:
        sobel_steps.append({
            'id': step['id'],
            'name': step['name'],
            'image_b64': to_base64(step['image']),
        })

    canny_steps = []
    for step in canny_data['steps']:
        canny_steps.append({
            'id': step['id'],
            'name': step['name'],
            'image_b64': to_base64(step['image']),
        })

    return jsonify({
        'original_image': f'/static/uploads/{unique_name}',
        'sobel': {'steps': sobel_steps, 'metrics': sobel_data['metrics']},
        'canny': {'steps': canny_steps, 'metrics': canny_data['metrics']},
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

    return jsonify({
        'original_image': f'/static/uploads/{unique_name}',
        'steps': [{'id': s['id'], 'name': s['name'], 'image_b64': to_base64(s['image'])} for s in steps],
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

    return jsonify({
        'original_image': f'/static/uploads/{unique_name}',
        'steps': [{'id': s['id'], 'name': s['name'], 'image_b64': to_base64(s['image'])} for s in steps],
        'keypoints': keypoints,
        'candidates': candidates,
        'metrics': metrics,
        'visualization': vis,
    })


# ---- Generic demo API for all algorithm modules ----

# Module dispatcher: maps module_id -> (processor_func, param_defaults)
def _get_demo_processor(module_id):
    """Return (build_pipeline_func, default_params) for a given module_id."""
    params = {}

    if module_id == 'histogram':
        from app.modules.phase1_fundamentals.histogram.processor import build_pipeline as fn
    elif module_id == 'threshold':
        from app.modules.phase1_fundamentals.threshold.processor import build_pipeline as fn
        params = {'method': 'otsu', 'threshold': 128}
    elif module_id == 'hough':
        from app.modules.phase2_classical.hough.processor import build_pipeline as fn
    elif module_id == 'morphology':
        from app.modules.phase2_classical.morphology.processor import build_pipeline as fn
    elif module_id == 'contour':
        from app.modules.phase2_classical.contour.processor import build_pipeline as fn
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
    elif module_id == 'conv_training':
        from app.modules.phase4_deep_learning.conv_training.processor import build_pipeline as fn
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

    # Collect params from form data, falling back to defaults
    kwargs = dict(defaults)
    for key in request.form:
        val = request.form[key]
        # Try to convert to appropriate type
        try:
            if '.' in val:
                kwargs[key] = float(val)
            else:
                kwargs[key] = int(val)
        except ValueError:
            kwargs[key] = val

    if 'file' in request.files and request.files['file'].filename:
        file = request.files['file']
        unique_name, upload_path = _save_upload(file)
        kwargs['image_path'] = upload_path
        original_url = f'/static/uploads/{unique_name}'
    else:
        # Modules that don't need upload: generate demo internally
        kwargs['image_path'] = None
        original_url = None

    result = fn(**kwargs)

    # Convert step images to base64
    steps_out = []
    for step in result.get('steps', []):
        step_out = {'id': step['id'], 'name': step['name']}
        if 'image' in step and step['image'] is not None:
            step_out['image_b64'] = to_base64(step['image'])
        if 'explanation' in step:
            step_out['explanation'] = step['explanation']
        steps_out.append(step_out)

    resp = {
        'steps': steps_out,
        'metrics': result.get('metrics', {}),
    }
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
