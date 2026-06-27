import io
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
from imageio.v3 import imwrite

from app import create_app
from app.modules.offline_teaching import (
    BLUEPRINT_MODULES,
    EXTERNAL_WEIGHT_MODULES,
    LOCAL_FRONTIER_ALGORITHM_MODULES,
    LOCAL_TEACHING_MODEL_MODULES,
    OFFLINE_TEACHING_MODULES,
)


def _fixture_image():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[16:48, 16:48] = [220, 120, 60]
    fd, path = tempfile.mkstemp(suffix='.png')
    os.close(fd)
    imwrite(path, img)
    return path


def _fixture_png_bytes():
    img = np.zeros((72, 96, 3), dtype=np.uint8)
    img[12:56, 18:62] = [220, 120, 60]
    img[28:68, 52:88] = [40, 170, 220]
    buf = io.BytesIO()
    imwrite(buf, img, extension='.png')
    buf.seek(0)
    return buf


def _assert_explainable_steps(payload, module_id):
    assert payload.get('steps'), module_id
    for step in payload['steps']:
        assert step.get('id'), module_id
        assert step.get('name'), f'{module_id}:{step.get("id")}'
        assert step.get('explanation'), f'{module_id}:{step.get("id")}'
        assert step.get('formula'), f'{module_id}:{step.get("id")}'
        assert step.get('image_base64'), f'{module_id}:{step.get("id")}'


def test_demo_endpoint_returns_steps_and_metrics():
    app = create_app()
    client = app.test_client()
    path = _fixture_image()
    try:
        with open(path, 'rb') as fh:
            resp = client.post(
                '/api/demo/threshold',
                data={'file': fh},
                content_type='multipart/form-data',
            )
        assert resp.status_code == 200
        payload = resp.get_json()
        assert 'steps' in payload and 'metrics' in payload
        assert 'implementation' in payload
        assert isinstance(payload['steps'], list) and payload['steps']
        assert all('id' in s and 'name' in s for s in payload['steps'])
    finally:
        os.remove(path)


def test_modules_report_real_implementation_status():
    app = create_app()
    client = app.test_client()

    resp = client.get('/api/modules')
    assert resp.status_code == 200
    payload = resp.get_json()
    modules = {
        mod['id']: mod
        for phase in payload['phases']
        for mod in phase['modules']
    }

    assert modules['vit']['implementation']['real_model'] is True
    assert modules['vit']['implementation']['category'] == 'pretrained_model'
    assert modules['vit']['implementation']['requires_upload'] is True
    assert modules['detection']['implementation']['real_model'] is True
    assert modules['detection']['implementation']['category'] != 'requires_external_weights'
    assert modules['semantic']['implementation']['real_model'] is True
    assert modules['semantic']['implementation']['category'] != 'requires_external_weights'
    assert modules['instance']['implementation']['real_model'] is True
    assert modules['instance']['implementation']['category'] != 'requires_external_weights'
    assert modules['diffusion']['implementation']['category'] == 'local_mechanism'
    assert modules['diffusion']['implementation']['real_model'] is False
    assert modules['gan']['implementation']['category'] == 'local_mechanism'


def test_local_frontier_modules_return_algorithm_mechanism_steps():
    app = create_app()
    client = app.test_client()

    for module_id in sorted(LOCAL_FRONTIER_ALGORITHM_MODULES):
        resp = client.post(f'/api/demo/{module_id}', data={}, content_type='multipart/form-data')
        payload = resp.get_json()
        assert resp.status_code == 200, module_id
        assert payload is not None, module_id
        assert payload['implementation']['category'] == 'numpy_algorithm', module_id
        assert payload['implementation']['real_model'] is False, module_id
        assert payload['metrics']['status'] in {'numpy_algorithm', 'local_algorithm'}, module_id
        assert len(payload['steps']) >= 4, module_id
        for step in payload['steps']:
            assert step.get('id'), module_id
            assert step.get('name'), module_id
            assert step.get('explanation'), f'{module_id}:{step.get("id")}'
            assert step.get('formula'), f'{module_id}:{step.get("id")}'
            assert step.get('image_base64'), f'{module_id}:{step.get("id")}'


def test_demo_without_upload_uses_builtin_fixture_for_upload_modules():
    app = create_app()
    client = app.test_client()

    for module_id in ['threshold', 'template_match']:
        resp = client.post(f'/api/demo/{module_id}', data={}, content_type='multipart/form-data')
        payload = resp.get_json()
        assert resp.status_code == 200, module_id
        assert payload['implementation']['requires_upload'] is True
        assert payload['original_image'] == '/static/images/demo-street.jpg'
        _assert_explainable_steps(payload, module_id)

    for module_id in ['sam']:
        resp = client.post(f'/api/demo/{module_id}', data={}, content_type='multipart/form-data')
        payload = resp.get_json()
        assert payload['implementation']['real_model'] is True
        assert payload['implementation']['requires_upload'] is True
        if resp.status_code == 200:
            assert payload['metrics']['status'] == 'pretrained_model', module_id
            _assert_explainable_steps(payload, module_id)
        else:
            assert resp.status_code in {400, 503}, module_id
            assert payload.get('error'), module_id


def test_sample_button_modules_run_without_frontend_upload():
    app = create_app()
    client = app.test_client()
    sample_modules = [
        'gaussian', 'histogram', 'hough', 'median', 'morphology',
        'sobel', 'template_match', 'threshold',
    ]

    for module_id in sample_modules:
        start = time.perf_counter()
        resp = client.post(f'/api/demo/{module_id}', data={}, content_type='multipart/form-data')
        elapsed = time.perf_counter() - start
        payload = resp.get_json()
        assert resp.status_code == 200, module_id
        assert elapsed < 12.0, f'{module_id} took {elapsed:.2f}s'
        assert payload['original_image'] == '/static/images/demo-street.jpg'
        _assert_explainable_steps(payload, module_id)


def test_core_demo_modules_return_json_contract():
    app = create_app()
    client = app.test_client()
    path = _fixture_image()
    try:
        for module_id in ['colorspace', 'grayscale', 'threshold', 'convolution', 'live', 'lenet']:
            data = {}
            if module_id not in {'diffusion'}:
                with open(path, 'rb') as fh:
                    data = {'file': (fh, f'{module_id}.png')}
                    resp = client.post(
                        f'/api/demo/{module_id}',
                        data=data,
                        content_type='multipart/form-data',
                    )
            else:
                resp = client.post(f'/api/demo/{module_id}', data={}, content_type='multipart/form-data')
            assert resp.status_code in (200, 400, 501, 503)
            payload = resp.get_json()
            assert payload is not None
            assert 'implementation' in payload
            if resp.status_code == 200:
                assert 'steps' in payload and 'metrics' in payload
    finally:
        os.remove(path)


def test_required_detection_segmentation_modules_are_runnable():
    app = create_app()
    client = app.test_client()
    path = _fixture_image()
    try:
        for module_id in ['detection', 'semantic', 'instance']:
            with open(path, 'rb') as fh:
                resp = client.post(
                    f'/api/demo/{module_id}',
                    data={'file': (fh, f'{module_id}.png')},
                    content_type='multipart/form-data',
                )
            payload = resp.get_json()
            assert resp.status_code in {200, 503}, module_id
            assert payload is not None, module_id
            assert payload['implementation']['category'] != 'requires_external_weights'
            assert payload['implementation']['category'] == 'pretrained_model'
            assert payload['implementation']['real_model'] is True
            if resp.status_code == 200:
                assert payload['steps'], module_id
                assert payload['metrics']['status'] in {'pretrained_model', 'partial_error'}
            else:
                assert payload['metrics']['status'] == 'model_not_available'
    finally:
        os.remove(path)


def test_ai_eye_model_catalog_reports_weight_status_and_defaults():
    app = create_app()
    client = app.test_client()

    resp = client.get('/api/ai-eye/models')
    payload = resp.get_json()

    assert resp.status_code == 200
    assert payload['implementation']['category'] == 'pretrained_model'
    assert payload['implementation']['real_model'] is True
    assert payload['defaults']['detection'] == 'fasterrcnn_resnet50_fpn'
    assert payload['defaults']['semantic'] == 'deeplabv3_resnet50'
    assert payload['defaults']['instance'] == 'maskrcnn_resnet50_fpn'
    for model_id in [
        'fasterrcnn_resnet50_fpn',
        'retinanet_resnet50_fpn',
        'deeplabv3_resnet50',
        'lraspp_mobilenet_v3_large',
        'maskrcnn_resnet50_fpn',
    ]:
        assert model_id in payload['models']
        assert payload['models'][model_id]['download_command']
        assert 'cache_dir' in payload['models'][model_id]


def test_ai_eye_unified_endpoint_returns_real_steps_outputs_and_models():
    app = create_app()
    client = app.test_client()

    resp = client.post(
        '/api/demo/ai_eye?task=detection',
        data={
            'file': (_fixture_png_bytes(), 'ai-eye-detection.png'),
            'score_threshold': '0.95',
        },
        content_type='multipart/form-data',
    )
    payload = resp.get_json()

    assert resp.status_code in {200, 503}
    assert payload is not None
    assert payload['module_id'] == 'ai_eye'
    assert payload['implementation']['category'] == 'pretrained_model'
    assert payload['implementation']['real_model'] is True
    assert 'models' in payload
    if resp.status_code == 200:
        assert payload['metrics']['status'] in {'pretrained_model', 'partial_error'}
        assert 'detection' in payload['algorithms']
        assert 'detection' in payload['outputs']
        assert len(payload['steps']) >= 5
        for step in payload['steps']:
            assert step.get('id')
            assert step.get('name')
            assert step.get('explanation')
            assert step.get('formula')
            assert step.get('image_base64')
        assert 'detections' in payload['outputs']['detection']


def test_ai_eye_legacy_detection_semantic_instance_endpoints_use_unified_contract():
    app = create_app()
    client = app.test_client()

    for module_id, output_key in [
        ('detection', 'detections'),
        ('semantic', 'labels'),
        ('instance', 'instances'),
    ]:
        resp = client.post(
            f'/api/demo/{module_id}',
            data={
                'file': (_fixture_png_bytes(), f'{module_id}.png'),
                'score_threshold': '0.95',
            },
            content_type='multipart/form-data',
        )
        payload = resp.get_json()
        assert resp.status_code in {200, 503}, module_id
        assert payload is not None, module_id
        assert payload['module_id'] == module_id
        assert payload['implementation']['category'] == 'pretrained_model'
        assert payload['implementation']['real_model'] is True
        if resp.status_code == 200:
            assert payload['metrics']['status'] in {'pretrained_model', 'partial_error'}, module_id
            assert module_id in payload['outputs'], module_id
            assert output_key in payload['outputs'][module_id], module_id
            assert payload['steps'], module_id


def test_upload_required_modules_do_not_return_500():
    app = create_app()
    client = app.test_client()
    modules_payload = client.get('/api/modules').get_json()
    module_ids = []
    for phase in modules_payload['phases']:
        for module in phase['modules']:
            implementation = module.get('implementation') or {}
            if implementation.get('requires_upload', True) and implementation.get('category') != 'requires_external_weights':
                module_ids.append(module['id'])

    script = r"""
import os
import sys
import tempfile
import numpy as np
from imageio.v3 import imwrite
from app import create_app

module_id = sys.argv[1]
app = create_app()
client = app.test_client()
img = np.zeros((64, 64, 3), dtype=np.uint8)
img[16:48, 16:48] = [220, 120, 60]
fd, path = tempfile.mkstemp(suffix='.png')
os.close(fd)
imwrite(path, img)
try:
    with open(path, 'rb') as fh:
        resp = client.post(
            f'/api/demo/{module_id}',
            data={'file': (fh, f'{module_id}.png')},
            content_type='multipart/form-data',
        )
    payload = resp.get_json()
    assert resp.status_code != 500, resp.status_code
    assert payload is not None
    assert 'implementation' in payload
    assert 'steps' in payload
    assert 'metrics' in payload
    if resp.status_code == 200:
        assert payload['steps']
        for step in payload['steps']:
            assert step.get('id')
            assert step.get('name')
            assert step.get('explanation')
            assert step.get('formula')
            assert step.get('image_base64')
finally:
    os.remove(path)
"""
    for module_id in sorted(module_ids):
        result = subprocess.run(
            [sys.executable, '-c', script, module_id],
            cwd=os.getcwd(),
            text=True,
            capture_output=True,
            timeout=45,
        )
        assert result.returncode == 0, (
            f'{module_id} failed upload contract\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}'
        )


def test_demo_endpoint_accepts_frontend_aliases():
    app = create_app()
    client = app.test_client()
    path = _fixture_image()
    aliases = {
        'canny': 'edge',
        'harris': 'corner',
        'faster_rcnn': 'detection',
        'fcn': 'semantic',
        'mask_rcnn': 'instance',
        'tpl_match': 'template_match',
        'sd': 'stable_diffusion',
    }
    try:
        for alias, canonical in aliases.items():
            with open(path, 'rb') as fh:
                resp = client.post(
                    f'/api/demo/{alias}',
                    data={'file': (fh, f'{alias}.png')},
                    content_type='multipart/form-data',
                )
            payload = resp.get_json()
            assert resp.status_code != 404, alias
            assert resp.status_code != 500, alias
            assert payload['requested_module_id'] == alias
            assert payload['module_id'] == canonical
    finally:
        os.remove(path)


def test_colorspace_demo_returns_four_spaces_and_channel_images():
    app = create_app()
    client = app.test_client()

    resp = client.post(
        '/api/demo/colorspace',
        data={'file': (_fixture_png_bytes(), 'colorspace.png')},
        content_type='multipart/form-data',
    )
    payload = resp.get_json()
    assert resp.status_code == 200
    assert payload['module_id'] == 'colorspace'
    assert payload['requested_module_id'] == 'colorspace'
    assert payload['implementation']['category'] == 'local_algorithm'
    assert payload['implementation']['real_model'] is False

    steps = {step['id']: step for step in payload['steps']}
    assert set(steps) == {'rgb_space', 'hsv_space', 'lab_space', 'cmyk_space'}
    assert [len(steps[key]['channels']) for key in ['rgb_space', 'hsv_space', 'lab_space', 'cmyk_space']] == [3, 3, 3, 4]
    for step in steps.values():
        assert step['formula']
        assert step['explanation']
        assert step['applications']
        for channel in step['channels']:
            assert channel['id']
            assert channel['name']
            assert channel['meaning']
            assert channel['range']
            assert channel['formula']
            assert channel['image_base64']
            assert channel['visualization_note']


def test_colorspace_cmyk_handles_black_white_and_rgba_without_divide_by_zero():
    app = create_app()
    client = app.test_client()
    img = np.zeros((12, 16, 4), dtype=np.uint8)
    img[:, :8, :3] = 0
    img[:, 8:, :3] = 255
    img[..., 3] = 255
    buf = io.BytesIO()
    imwrite(buf, img, extension='.png')
    buf.seek(0)

    resp = client.post(
        '/api/demo/colorspace',
        data={'file': (buf, 'black-white-rgba.png')},
        content_type='multipart/form-data',
    )
    payload = resp.get_json()
    assert resp.status_code == 200
    cmyk = next(step for step in payload['steps'] if step['id'] == 'cmyk_space')
    assert len(cmyk['channels']) == 4
    assert payload['metrics']['cmyk_k_mean'] >= 0


def test_histogram_equalization_returns_animation_data_and_stable_steps():
    app = create_app()
    client = app.test_client()

    resp = client.post(
        '/api/demo/histogram',
        data={'file': (_fixture_png_bytes(), 'histogram.png')},
        content_type='multipart/form-data',
    )
    payload = resp.get_json()
    assert resp.status_code == 200
    steps = {step['id']: step for step in payload['steps']}
    assert ['original', 'gray', 'histogram', 'cdf', 'mapping', 'equalized', 'equalized_histogram'] == [s['id'] for s in payload['steps']]
    for step in payload['steps']:
        assert step['formula']
        assert step['explanation']
        assert step['image_base64']

    data = steps['histogram']['data']
    assert len(data['histogram']) == 256
    assert len(data['cdf']) == 256
    assert len(data['mapping']) == 256
    assert len(data['equalized_histogram']) == 256
    assert all(data['cdf'][i] <= data['cdf'][i + 1] for i in range(255))
    assert all(0 <= x <= 255 for x in data['mapping'])
    assert all(data['mapping'][i] <= data['mapping'][i + 1] for i in range(255))
    assert 'stats' in data


def test_histogram_equalization_handles_constant_and_rgba_images():
    app = create_app()
    client = app.test_client()
    for value in [0, 255, 128]:
        img = np.zeros((18, 22, 4), dtype=np.uint8)
        img[..., :3] = value
        img[..., 3] = 255
        buf = io.BytesIO()
        imwrite(buf, img, extension='.png')
        buf.seek(0)
        resp = client.post(
            '/api/demo/histogram',
            data={'file': (buf, f'constant-{value}.png')},
            content_type='multipart/form-data',
        )
        payload = resp.get_json()
        assert resp.status_code == 200
        data = next(step for step in payload['steps'] if step['id'] == 'histogram')['data']
        assert len(data['mapping']) == 256
        assert all(np.isfinite(x) for x in data['mapping'])
        assert all(0 <= x <= 255 for x in data['mapping'])


def test_threshold_returns_otsu_process_data_and_overlay():
    app = create_app()
    client = app.test_client()

    resp = client.post(
        '/api/demo/threshold',
        data={'file': (_fixture_png_bytes(), 'threshold.png')},
        content_type='multipart/form-data',
    )
    payload = resp.get_json()
    assert resp.status_code == 200
    assert ['original', 'gray', 'histogram', 'otsu_score', 'decision_rule', 'result', 'overlay'] == [s['id'] for s in payload['steps']]
    steps = {step['id']: step for step in payload['steps']}
    data = steps['histogram']['data']
    assert len(data['histogram']) == 256
    assert len(data['otsu_scores']) == 256
    assert 0 <= data['threshold'] <= 255
    assert payload['metrics']['method'] == 'otsu'
    assert payload['metrics']['threshold'] == data['threshold']
    assert steps['overlay']['image_base64']


def test_noise_model_returns_distinct_models_and_statistics():
    app = create_app()
    client = app.test_client()

    resp = client.post(
        '/api/demo/noise',
        data={'file': (_fixture_png_bytes(), 'noise.png')},
        content_type='multipart/form-data',
    )
    payload = resp.get_json()
    assert resp.status_code == 200
    ids = [step['id'] for step in payload['steps']]
    assert {'original', 'salt_pepper', 'impulse_mask', 'gaussian_noise', 'gaussian_distribution', 'poisson_noise', 'residual_map'} <= set(ids)
    for step in payload['steps']:
        assert step['formula']
        assert step['explanation']
        assert step['image_base64']
    assert payload['metrics']['recommended_filters']['salt_pepper'] == 'median filter'
    assert payload['metrics']['gaussian_psnr_db'] > 0


def test_smoothing_demo_unifies_gaussian_median_and_bilateral():
    app = create_app()
    client = app.test_client()

    resp = client.post(
        '/api/demo/smoothing',
        data={'file': (_fixture_png_bytes(), 'smoothing.png')},
        content_type='multipart/form-data',
    )
    payload = resp.get_json()
    assert resp.status_code == 200
    assert payload['module_id'] == 'smoothing'
    assert {'gaussian', 'median', 'bilateral'} <= set(payload['algorithms'])
    assert payload['implementation']['category'] == 'numpy_algorithm'
    assert payload['implementation']['real_model'] is False

    for algorithm_id in ['gaussian', 'median', 'bilateral']:
        steps = payload['algorithms'][algorithm_id]['steps']
        assert len(steps) >= 4, algorithm_id
        for step in steps:
            assert step['id']
            assert step['name']
            assert step['explanation']
            assert step['formula']
            assert step['image_base64']

    kernel_step = next(step for step in payload['algorithms']['gaussian']['steps'] if step['id'] == 'gaussian_kernel')
    assert abs(kernel_step['data']['kernel_sum'] - 1.0) < 1e-6
    median_step = next(step for step in payload['algorithms']['median']['steps'] if step['id'] == 'median_sort')
    sorted_values = median_step['data']['sorted_values']
    assert sorted_values == sorted(sorted_values)
    assert sorted_values[median_step['data']['median_index']] == median_step['data']['median_value']
    bilateral_step = next(step for step in payload['algorithms']['bilateral']['steps'] if step['id'] == 'bilateral_combined_weights')
    weights = np.array(bilateral_step['data']['combined_weights'], dtype=float)
    assert np.all(np.isfinite(weights))
    assert np.all(weights >= 0)
    assert abs(float(weights.sum()) - 1.0) < 1e-6


def test_smoothing_legacy_filter_endpoints_share_unified_pipeline():
    app = create_app()
    client = app.test_client()

    for module_id in ['gaussian', 'median', 'bilateral']:
        resp = client.post(
            f'/api/demo/{module_id}',
            data={'file': (_fixture_png_bytes(), f'{module_id}.png')},
            content_type='multipart/form-data',
        )
        payload = resp.get_json()
        assert resp.status_code == 200, module_id
        assert payload['requested_module_id'] == module_id
        assert payload['module_id'] == 'smoothing'
        assert {'gaussian', 'median', 'bilateral'} <= set(payload['algorithms'])
        assert payload['steps']


def test_blueprint_modules_are_runnable_or_controlled():
    app = create_app()
    client = app.test_client()
    path = _fixture_image()
    try:
        for module_id in sorted(BLUEPRINT_MODULES):
            data = {}
            if module_id not in EXTERNAL_WEIGHT_MODULES:
                with open(path, 'rb') as fh:
                    data = {'file': (fh, f'{module_id}.png')}
                    resp = client.post(
                        f'/api/demo/{module_id}',
                        data=data,
                        content_type='multipart/form-data',
                    )
            else:
                resp = client.post(f'/api/demo/{module_id}', data={}, content_type='multipart/form-data')
            payload = resp.get_json()
            assert payload is not None, module_id
            assert 'implementation' in payload, module_id
            assert 'steps' in payload, module_id
            assert 'metrics' in payload, module_id
            if module_id in EXTERNAL_WEIGHT_MODULES:
                assert resp.status_code == 503, module_id
                assert payload['implementation']['category'] == 'requires_external_weights'
                assert payload['metrics']['status'] == 'requires_external_weights'
            elif module_id in LOCAL_FRONTIER_ALGORITHM_MODULES:
                assert resp.status_code == 200, module_id
                assert payload['implementation']['category'] == 'numpy_algorithm'
                assert payload['implementation']['real_model'] is False
                assert len(payload['steps']) >= 4, module_id
            elif module_id in LOCAL_TEACHING_MODEL_MODULES:
                assert resp.status_code == 200, module_id
                assert payload['implementation']['category'] == 'numpy_algorithm'
                assert payload['implementation']['real_model'] is False
                assert payload['steps'], module_id
            elif module_id in OFFLINE_TEACHING_MODULES:
                assert resp.status_code == 200, module_id
                assert payload['implementation']['category'] == 'teaching_simulation'
                assert payload['steps'], module_id
    finally:
        os.remove(path)


def test_all_registered_modules_demo_endpoint_has_json_contract():
    app = create_app()
    client = app.test_client()
    modules_payload = client.get('/api/modules').get_json()

    for phase in modules_payload['phases']:
        for module in phase['modules']:
            module_id = module['id']
            impl = module.get('implementation') or {}
            if impl.get('requires_upload', True):
                # Upload-required modules are covered in subprocess isolation by
                # test_upload_required_modules_do_not_return_500. Some CV/torch
                # stacks abort the host process only after many mixed requests.
                continue
            resp = client.post(f'/api/demo/{module_id}', data={}, content_type='multipart/form-data')
            payload = resp.get_json()
            assert resp.status_code not in {404, 500}, module_id
            assert payload is not None, module_id
            assert 'steps' in payload, module_id
            assert 'metrics' in payload, module_id
            assert 'implementation' in payload, module_id
            category = payload['implementation'].get('category')
            if category in {'pretrained_model', 'model_not_available'} and resp.status_code in {400, 503}:
                assert payload.get('error'), module_id
                continue
            if category != 'requires_external_weights':
                assert resp.status_code == 200, module_id
                assert payload['steps'], module_id


def test_all_registered_modules_return_explainable_visual_steps():
    app = create_app()
    client = app.test_client()
    modules_payload = client.get('/api/modules').get_json()

    for phase in modules_payload['phases']:
        for module in phase['modules']:
            module_id = module['id']
            impl = module.get('implementation') or {}
            if impl.get('requires_upload', True):
                # Upload-required modules are asserted in subprocess isolation by
                # test_upload_required_modules_do_not_return_500. Mixing all CV,
                # SciPy and torch stacks in one process can abort the host.
                continue
            data = {}
            resp = client.post(
                f'/api/demo/{module_id}',
                data=data,
                content_type='multipart/form-data',
            )
            payload = resp.get_json()
            category = (payload.get('implementation') or {}).get('category') if payload else None
            if category in {'pretrained_model', 'model_not_available'} and resp.status_code in {400, 503}:
                assert payload.get('error'), module_id
                continue
            assert resp.status_code == 200, module_id
            assert payload is not None, module_id
            _assert_explainable_steps(payload, module_id)
