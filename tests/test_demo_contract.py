import io
import os
import subprocess
import sys
import tempfile

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

    assert modules['vit']['implementation']['real_model'] is False
    assert modules['vit']['implementation']['category'] == 'numpy_algorithm'
    assert modules['vit']['implementation']['requires_upload'] is False
    assert modules['detection']['implementation']['real_model'] is True
    assert modules['detection']['implementation']['category'] != 'requires_external_weights'
    assert modules['semantic']['implementation']['real_model'] is True
    assert modules['semantic']['implementation']['category'] != 'requires_external_weights'
    assert modules['instance']['implementation']['real_model'] is True
    assert modules['instance']['implementation']['category'] != 'requires_external_weights'
    assert modules['diffusion']['implementation']['category'] == 'pretrained_model'
    assert modules['gan']['implementation']['category'] == 'numpy_algorithm'


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


def test_demo_requires_upload_and_returns_controlled_error():
    app = create_app()
    client = app.test_client()

    for module_id in ['threshold']:
        resp = client.post(f'/api/demo/{module_id}', data={}, content_type='multipart/form-data')
        payload = resp.get_json()
        assert payload['steps'] == []
        if payload['implementation']['category'] == 'requires_external_weights':
            assert resp.status_code == 503
            assert payload['metrics']['status'] == 'requires_external_weights'
        else:
            assert resp.status_code == 400
            assert payload['implementation']['requires_upload'] is True
            assert payload['metrics']['status'] == 'missing_upload'

    for module_id in ['detr', 'clip', 'sam']:
        resp = client.post(f'/api/demo/{module_id}', data={}, content_type='multipart/form-data')
        payload = resp.get_json()
        assert resp.status_code == 200, module_id
        assert payload['steps'], module_id
        assert payload['implementation']['category'] == 'numpy_algorithm'
        assert payload['implementation']['real_model'] is False
        assert payload['implementation']['requires_upload'] is False


def test_core_demo_modules_return_json_contract():
    app = create_app()
    client = app.test_client()
    path = _fixture_image()
    try:
        for module_id in ['grayscale', 'threshold', 'convolution', 'live', 'lenet']:
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
            assert resp.status_code == 200, module_id
            assert payload is not None, module_id
            assert payload['implementation']['category'] != 'requires_external_weights'
            assert payload['steps'], module_id
            assert payload['metrics']['status'] in {'local_teaching_fallback', 'pretrained_model'}
    finally:
        os.remove(path)


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
            if payload['implementation'].get('category') != 'requires_external_weights':
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
            assert resp.status_code == 200, module_id
            assert payload is not None, module_id
            _assert_explainable_steps(payload, module_id)
