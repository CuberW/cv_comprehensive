import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image

from app import create_app


def test_cv_legacy_files_are_present():
    required = [
        'app/modules/phase1_fundamentals/convolution/backprop.py',
        'app/modules/phase1_fundamentals/convolution/lenet.py',
        'app/modules/phase1_fundamentals/convolution/processor.py',
        'app/modules/phase2_classical/edge/edge_numba.py',
        'app/modules/phase2_classical/edge/edge_naive.py',
        'static/data/edge_naive_benchmark.json',
        'static/data/edge_optimization_benchmark.json',
        'templates/pages/lenet_backprop.html',
    ]
    for path in required:
        assert Path(path).exists(), path


def test_cv_legacy_conv_routes_are_wired():
    app = create_app()
    client = app.test_client()

    assert client.get('/conv/backprop').status_code == 200
    assert client.get('/static/data/edge_naive_benchmark.json').status_code == 200
    assert client.get('/static/data/edge_optimization_benchmark.json').status_code == 200

    assert client.post('/conv/api/generate', json={'h': 3, 'w': 4, 'seed': 1}).status_code == 200
    assert client.post('/conv/api/kernel', json={'size': 3, 'preset': 'edge_detect'}).status_code == 200
    assert client.post(
        '/conv/api/compute',
        json={
            'mode': 'basic',
            'input': [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            'kernel': [[1, 0], [0, -1]],
            'stride': 1,
            'padding': 0,
        },
    ).status_code == 200

    img = np.zeros((28, 28), dtype=np.uint8)
    img[8:20, 12:16] = 255
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format='PNG')
    encoded = base64.b64encode(buf.getvalue()).decode('ascii')
    assert client.post('/conv/api/forward_trace', json={'image': encoded}).status_code == 200
