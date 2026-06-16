import os
import tempfile

import numpy as np
from imageio.v3 import imwrite

from app import create_app


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
        assert isinstance(payload['steps'], list) and payload['steps']
        assert all('id' in s and 'name' in s for s in payload['steps'])
    finally:
        os.remove(path)
