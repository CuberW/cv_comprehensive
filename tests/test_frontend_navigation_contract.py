import json
import re
import subprocess
from pathlib import Path

from app import create_app
from app.routes import MODULE_ALIASES


DEBUG_OR_EXPERIMENTAL_PAGES = {
    'upload_test.html',
    'gauss_test.html',
}


def _formal_static_pages():
    return sorted(
        page
        for page in Path('static/pages').glob('*.html')
        if page.name not in DEBUG_OR_EXPERIMENTAL_PAGES
    )


def _algorithm_content():
    script = (
        "const fs=require('fs'),vm=require('vm');"
        "const code=fs.readFileSync('static/js/algorithm-content.js','utf8');"
        "const ctx={window:{},console};"
        "vm.createContext(ctx);vm.runInContext(code,ctx);"
        "console.log(JSON.stringify(ctx.window.AlgorithmContent));"
    )
    result = subprocess.run(
        ['node', '-e', script],
        cwd='.',
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout.decode('utf-8'))


def _registered_module_ids():
    app = create_app()
    client = app.test_client()
    payload = client.get('/api/modules').get_json()
    return {
        mod['id']
        for phase in payload['phases']
        for mod in phase['modules']
    }


def test_app_shell_shows_topics_not_individual_algorithm_cards():
    script = Path('static/js/app.js').read_text(encoding='utf-8')

    assert 'function _card(a)' not in script
    assert 'phase.algo.map(a=>_card(a))' not in script
    assert 'grp.algo.map(a=>_card(a))' not in script
    assert 'line.querySelectorAll(\'.algo-card\')' not in script
    assert "line.className='phase-line topic-rail'" in script
    assert 'topic-row' in script
    assert 'TOPIC_CARDS.filter' in script
    assert 'bridge-algo-tag' in script


def test_topic_cards_are_vertical_without_horizontal_scroll():
    css = Path('static/css/main.css').read_text(encoding='utf-8')
    topic_row = css.split('.topic-row{', 1)[1].split('}', 1)[0]
    topic_card = css.split('.topic-row .topic-card{', 1)[1].split('}', 1)[0]

    assert 'flex-direction:column' in topic_row
    assert 'overflow-x:auto' not in topic_row
    assert 'scroll-snap-type:x' not in topic_row
    assert 'scroll-snap-align' not in topic_card
    assert 'width:100%' in topic_card


def test_implemented_modules_keep_their_existing_subpages():
    app = create_app()
    client = app.test_client()
    payload = client.get('/api/modules').get_json()
    modules = {
        mod['id']: mod
        for phase in payload['phases']
        for mod in phase['modules']
    }

    expected_pages = {
        'grayscale': 'grayscale.html',
        'convolution': 'conv_basic.html',
        'edge': 'edge.html',
        'corner': 'corner.html',
        'sift': 'sift.html',
        'match': 'match.html',
        'detection': 'detection.html',
        'semantic': 'semantic.html',
        'instance': 'instance.html',
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

    for module_id, page in expected_pages.items():
        assert modules[module_id]['page'] == page, module_id


def test_formal_pages_load_unified_iframe_theme_assets():
    for page in _formal_static_pages():
        html = page.read_text(encoding='utf-8')
        assert '../css/iframe-theme.css' in html, page.name
        assert '../js/theme-sync.js' in html, page.name


def test_theme_sync_uses_parent_messages_and_standalone_local_storage():
    script = Path('static/js/theme-sync.js').read_text(encoding='utf-8')

    assert 'window.parent.document.documentElement' in script
    assert "localStorage.getItem('theme')" in script
    assert "type==='theme'" in script or 'type === \'theme\'' in script
    assert "setAttribute('data-theme'" in script


def test_algorithm_content_entries_target_registered_demo_modules():
    allowed_ids = _registered_module_ids() | set(MODULE_ALIASES)
    content = _algorithm_content()

    for module_id, cfg in content.items():
        if module_id == 'common':
            continue
        endpoint = cfg.get('endpoint')
        assert endpoint and endpoint.startswith('/api/demo/'), module_id
        endpoint_id = endpoint.rsplit('/', 1)[-1]
        assert endpoint_id in allowed_ids, f'{module_id} -> {endpoint_id}'


def test_nn_interactive_referenced_modules_have_demo_backends():
    allowed_ids = _registered_module_ids() | set(MODULE_ALIASES)
    html = Path('static/pages/nn_interactive.html').read_text(encoding='utf-8')
    data_keys = set(re.findall(r'([A-Za-z0-9_]+):\{cat:', html))

    assert data_keys
    assert data_keys <= allowed_ids


def test_formal_visible_files_do_not_contain_common_mojibake():
    patterns = ('鐪', '闃', '鍥', '绠', '鈥', '鈫', '脳', '�')
    files = _formal_static_pages() + [
        Path('static/js/app.js'),
        Path('static/js/algorithm-content.js'),
        Path('static/js/teaching-page.js'),
        Path('static/js/detection-segmentation.js'),
    ]

    for path in files:
        text = path.read_text(encoding='utf-8')
        hits = [pat for pat in patterns if pat in text]
        assert not hits, f'{path}: {hits}'


def test_demo_step_renderers_include_formula_output():
    files = _formal_static_pages() + [
        Path('static/js/detection-segmentation.js'),
    ]

    for path in files:
        text = path.read_text(encoding='utf-8')
        if '/api/demo/' not in text or ('steps' not in text and 'image_base64' not in text):
            continue
        assert '.formula' in text or "['formula']" in text or '["formula"]' in text, path


def test_formal_inline_scripts_parse_with_node(tmp_path):
    for page in _formal_static_pages():
        html = page.read_text(encoding='utf-8')
        scripts = re.findall(r'<script(?:\s[^>]*)?>(.*?)</script>', html, flags=re.S | re.I)
        for index, code in enumerate(scripts):
            if not code.strip() or ('location.replace' in code and len(code.strip()) < 120):
                continue
            script_path = tmp_path / f'{page.stem}_{index}.js'
            script_path.write_text(code, encoding='utf-8')
            result = subprocess.run(
                ['node', '--check', str(script_path)],
                text=True,
                capture_output=True,
                timeout=20,
            )
            assert result.returncode == 0, f'{page.name} script {index}\n{result.stderr or result.stdout}'
