import os
import tempfile

import numpy as np
from imageio.v3 import imwrite

from app import create_app
from app.modules.offline_teaching import BLUEPRINT_MODULES, EXTERNAL_WEIGHT_MODULES, OFFLINE_TEACHING_MODULES


def _fixture_image():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[16:48, 16:48] = [220, 120, 60]
    fd, path = tempfile.mkstemp(suffix='.png')
    os.close(fd)
    imwrite(path, img)
    return path


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
    assert modules['vit']['implementation']['category'] == 'requires_external_weights'
    assert modules['detection']['implementation']['real_model'] is True
    assert modules['detection']['implementation']['category'] != 'requires_external_weights'
    assert modules['semantic']['implementation']['real_model'] is True
    assert modules['semantic']['implementation']['category'] != 'requires_external_weights'
    assert modules['instance']['implementation']['real_model'] is True
    assert modules['instance']['implementation']['category'] != 'requires_external_weights'
    assert modules['diffusion']['implementation']['category'] == 'pretrained_model'
    assert modules['gan']['implementation']['category'] == 'teaching_simulation'


def test_external_weight_modules_return_controlled_error():
    app = create_app()
    client = app.test_client()

    resp = client.post('/api/demo/vit', data={}, content_type='multipart/form-data')
    assert resp.status_code == 503
    payload = resp.get_json()
    assert payload['implementation']['category'] == 'requires_external_weights'
    assert payload['metrics']['status'] == 'requires_external_weights'
    assert payload['steps'] == []


def test_demo_requires_upload_and_returns_controlled_error():
    app = create_app()
    client = app.test_client()

    for module_id in ['threshold', 'detr', 'clip', 'sam']:
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
            elif module_id in OFFLINE_TEACHING_MODULES:
                assert resp.status_code == 200, module_id
                assert payload['implementation']['category'] == 'teaching_simulation'
                assert payload['steps'], module_id
    finally:
        os.remove(path)
