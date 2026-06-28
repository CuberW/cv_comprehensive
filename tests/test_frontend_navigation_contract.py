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

HIDDEN_COMPAT_MODULES = {
    'ai_eye',
    'nms',
    'template_match',
    'tpl_match',
    'hough',
    'contour',
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


def _homepage_algorithm_ids():
    script = Path('static/js/app.js').read_text(encoding='utf-8')
    topic_block = script.split('const TOPIC_CARDS =', 1)[1].split('function _renderTopicCard', 1)[0]
    ids = set(re.findall(r"\{id:'([^']+)'", topic_block))
    ids.update(re.findall(r"mid:'([^']+)'", topic_block))
    return ids


def _nn_interactive_ids():
    html = Path('static/pages/nn_interactive.html').read_text(encoding='utf-8')
    return set(re.findall(r'([A-Za-z0-9_]+):\{cat:', html))


def _html_query_module_refs():
    refs = set()
    for page in _formal_static_pages():
        text = page.read_text(encoding='utf-8')
        refs.update(re.findall(r'[?&](?:id|module)=([A-Za-z0-9_]+)', text))
        refs.update(re.findall(r"(?:id|module)[:=]'([A-Za-z0-9_]+)'", text))
    return refs


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
        'colorspace': 'colorspace.html',
        'grayscale': 'grayscale.html',
        'convolution': 'conv_basic.html',
        'edge': 'edge.html',
        'corner': 'corner.html',
        'sift': 'sift.html',
        'match': 'match.html',
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

    for module_id, page in expected_pages.items():
        assert modules[module_id]['page'] == page, module_id


def test_formal_pages_load_unified_iframe_theme_assets():
    for page in _formal_static_pages():
        html = page.read_text(encoding='utf-8')
        assert '../css/iframe-theme.css' in html, page.name
        assert '../js/theme-sync.js' in html, page.name


def test_formal_pages_using_esc_load_or_define_escape_helper():
    helper_tag = '<script src="../js/page-utils.js"></script>'
    uses_esc = re.compile(r'\besc\s*\(')
    defines_esc = re.compile(r'function\s+esc\b|(?:var|let|const)\s+esc\s*=|window\.esc\s*=')

    for page in _formal_static_pages():
        html = page.read_text(encoding='utf-8')
        helper_pos = html.find(helper_tag)
        scripts = list(re.finditer(r'<script(?:\s[^>]*)?>(.*?)</script>', html, flags=re.S | re.I))

        for index, match in enumerate(scripts):
            code = match.group(1)
            if not uses_esc.search(code) or defines_esc.search(code):
                continue
            assert helper_pos != -1, f'{page.name} script {index} uses esc() without a helper'
            assert helper_pos < match.start(), f'{page.name} loads page-utils.js after script {index}'


def test_histogram_threshold_noise_pages_have_rich_visual_teaching_surfaces():
    histogram = Path('static/pages/histogram_new.html').read_text(encoding='utf-8')
    threshold = Path('static/pages/threshold_new.html').read_text(encoding='utf-8')
    noise = Path('static/pages/noise.html').read_text(encoding='utf-8')

    for html in [histogram, threshold, noise]:
        assert '../css/iframe-theme.css' in html
        assert '../js/theme-sync.js' in html
        assert '../js/page-utils.js' in html

    for marker in [
        'mappingDemo',
        'compareSlider',
        'histCanvas',
        'cdfCanvas',
        'mapCanvas',
        'eqHistCanvas',
        '低光照照片增强',
    ]:
        assert marker in histogram

    for marker in [
        'decisionCanvas',
        'scoreCanvas',
        'stripCanvas',
        'manualThreshold',
        'Otsu 搜索分数',
        '文档二值化',
    ]:
        assert marker in threshold

    for marker in [
        'noiseScope',
        'imageGrid',
        'model-row',
        'strategy-grid',
        '椒盐噪声',
        '泊松噪声',
    ]:
        assert marker in noise


def test_smoothing_page_is_the_single_filter_frontend():
    app = create_app()
    client = app.test_client()
    payload = client.get('/api/modules').get_json()
    modules = {
        mod['id']: mod
        for phase in payload['phases']
        for mod in phase['modules']
    }

    for module_id in ['smoothing', 'gaussian', 'median', 'bilateral']:
        assert modules[module_id]['page'] == 'smoothing.html', module_id

    app_js = Path('static/js/app.js').read_text(encoding='utf-8')
    topic_block = app_js.split('const TOPIC_CARDS =', 1)[1].split('function _renderTopicCard', 1)[0]
    assert "{id:'smoothing'" in topic_block
    assert "{id:'gaussian'" not in topic_block
    assert "{id:'median'" not in topic_block
    assert "{id:'bilateral'" not in topic_block

    html = Path('static/pages/smoothing.html').read_text(encoding='utf-8')
    for marker in [
        '../css/iframe-theme.css',
        '../js/theme-sync.js',
        '../js/page-utils.js',
        '../js/card-anims.js',
        'data-step="gaussian"',
        'initCannyGaussianAnim',
        'median-sort',
        'bilateral-anim',
        'compareTable',
        '/api/demo/smoothing',
    ]:
        assert marker in html


def test_sam_page_is_interactive_prompt_teaching_stage():
    html = Path('static/pages/sam.html').read_text(encoding='utf-8')
    script = Path('static/js/sam-page.js').read_text(encoding='utf-8')

    for marker in [
        '../css/iframe-theme.css',
        '../js/theme-sync.js',
        '../js/page-utils.js',
        '../js/sam-page.js',
        'data-mode="positive"',
        'data-mode="negative"',
        'data-mode="box"',
        'candidateGrid',
        'promptCanvas',
        'frameSlider',
    ]:
        assert marker in html

    for marker in [
        "/api/demo/sam",
        "points",
        "labels",
        "box",
        "selected_mask",
        "candidate_masks",
        "composeMask",
    ]:
        assert marker in script


def test_model_demo_page_keeps_real_interaction_parameter_wiring():
    script = Path('static/js/model-demo-page.js').read_text(encoding='utf-8')

    for marker in [
        'onStageImageClick',
        'state.dynamicParams.selected_patch',
        "fetch('/api/demo/' + encodeURIComponent(moduleId)",
        'delete state.dynamicParams[p.name]',
        '正在用真实 ViT 注意力重新计算',
    ]:
        assert marker in script


def test_hidden_compat_modules_are_not_standalone_frontend_but_demos_stay_compatible():
    app = create_app()
    client = app.test_client()
    payload = client.get('/api/modules').get_json()
    registered_visible_ids = {
        mod['id']
        for phase in payload['phases']
        for mod in phase['modules']
    }
    homepage_ids = _homepage_algorithm_ids()

    for module_id in ['nms', 'template_match', 'hough', 'contour']:
        assert module_id not in registered_visible_ids
    for module_id in ['nms', 'tpl_match', 'hough', 'contour']:
        assert module_id not in homepage_ids

    for module_id in ['nms', 'template_match', 'hough', 'contour']:
        resp = client.post(f'/api/demo/{module_id}', data={}, content_type='multipart/form-data')
        demo = resp.get_json()
        assert resp.status_code == 200, module_id
        assert demo['module_id'] == module_id
        assert demo['steps'], module_id


def test_hidden_compat_modules_are_removed_from_algorithm_network():
    network = Path('static/algorithm_network.html').read_text(encoding='utf-8')

    for marker in [
        "{id:'tpl_match'",
        "{id:'hough'",
        "{id:'contour'",
        "{id:'nms'",
        '模板匹配',
        'Hough变换',
        '轮廓查找',
    ]:
        assert marker not in network


def test_page_utils_provides_global_escape_helpers():
    script = Path('static/js/page-utils.js')
    assert script.exists()
    text = script.read_text(encoding='utf-8')

    assert 'window.esc' in text
    assert 'window.escapeHtml' in text
    assert '&amp;' in text
    assert '&#39;' in text


def test_theme_sync_uses_parent_messages_and_standalone_local_storage():
    script = Path('static/js/theme-sync.js').read_text(encoding='utf-8')

    assert 'window.parent.document.documentElement' in script
    assert "localStorage.getItem('theme')" in script
    assert "type==='theme'" in script or 'type === \'theme\'' in script
    assert "setAttribute('data-theme'" in script


def test_theme_sync_injects_final_light_mode_overrides():
    script = Path('static/js/theme-sync.js').read_text(encoding='utf-8')

    assert 'cv-theme-final-overrides' in script
    assert 'DOMContentLoaded' in script
    assert '--cv-page-gutter' in script
    assert '--cv-page-max' in script
    assert 'html body .ai-page' in script
    assert 'html body .sam-page' in script
    assert 'html[data-theme="dark"] .step-card' in script
    assert 'html[data-theme="dark"] .step-img' in script
    assert 'html[data-theme="light"] .step-card' in script
    assert 'html[data-theme="light"] .step-img' in script
    assert 'html[data-theme="light"] .teach-title-panel' in script
    assert 'html[data-theme="light"] .teach-step-card' in script
    assert 'html[data-theme="light"] .nn-step-card' in script
    assert 'html[data-theme="light"] .rc' in script
    assert 'html[data-theme="light"] input' in script
    assert 'var(--cv-media-bg)' in script
    assert 'var(--cv-control-bg)' in script
    assert 'button.primary' in script


def test_iframe_theme_has_light_safe_color_tokens():
    css = Path('static/css/iframe-theme.css').read_text(encoding='utf-8')
    light_block = css.split('[data-theme="light"] {', 1)[1].split('}', 1)[0]

    for token in [
        '--cv-page-max',
        '--cv-page-gutter',
        '--cv-card-gap',
        '--cv-card-pad',
    ]:
        assert token in css

    for token in [
        '--cv-title',
        '--cv-page-header-bg',
        '--cv-soft-surface',
        '--cv-control-bg',
        '--cv-media-bg',
        '--cv-code-bg',
        '--cv-danger-text',
    ]:
        assert token in css
        assert token in light_block

    assert 'color: var(--cv-title)' in css
    assert 'background: var(--cv-media-bg)' in css
    assert '.ai-page' in css
    assert '.sam-page' in css
    assert 'object-fit: contain !important' in css


def test_page_specific_css_uses_shared_layout_variables():
    css_files = [
        Path('static/css/model-demo-page.css'),
        Path('static/css/ai-eye-page.css'),
        Path('static/css/sam-page.css'),
    ]

    for path in css_files:
        css = path.read_text(encoding='utf-8')
        assert '--cv-page-max' in css, path
        assert '--cv-page-gutter' in css, path
        assert 'calc(100% - 48px)' not in css, path
        assert 'calc(100vw - 44px)' not in css, path


def test_main_gallery_spacing_is_wider_but_not_edge_to_edge():
    css = Path('static/css/main.css').read_text(encoding='utf-8')
    gallery_block = css.split('.gallery{', 1)[1].split('}', 1)[0]

    assert 'max-width:1580px' in gallery_block
    assert 'clamp(20px,2.2vw,30px)' in gallery_block
    assert 'padding:24px 32px 48px' not in gallery_block


def test_main_light_topic_cards_override_dark_topic_backgrounds():
    css = Path('static/css/main.css').read_text(encoding='utf-8')
    topic_classes = [
        'topic-color-and-contrast',
        'topic-filter-and-convolution',
        'topic-edges-to-corners',
        'topic-descriptors-and-shapes',
        'topic-feature-matching',
        'topic-ai-eye-bridge',
        'topic-ai-eye-detection-seg',
        'topic-cnn-foundations',
        'topic-generative-models',
        'topic-transformer-vision',
        'topic-foundation-models',
    ]

    for topic in topic_classes:
        marker = f'[data-theme="light"] .bridge-body.{topic}'
        assert marker in css
        block = css.split(marker, 1)[1].split('}', 1)[0]
        assert '!important' in block, topic
        assert 'rgba(15,23,42' not in block, topic
        assert '#080d16' not in block, topic

    assert '[data-theme="light"] .bridge-body.topic-ai-eye-bridge h3' in css
    assert '[data-theme="light"] .bridge-body.topic-ai-eye-bridge p' in css


def test_algorithm_content_entries_target_registered_demo_modules():
    allowed_ids = _registered_module_ids() | set(MODULE_ALIASES) | HIDDEN_COMPAT_MODULES
    content = _algorithm_content()

    for module_id, cfg in content.items():
        if module_id == 'common':
            continue
        endpoint = cfg.get('endpoint')
        assert endpoint and endpoint.startswith('/api/demo/'), module_id
        endpoint_id = endpoint.rsplit('/', 1)[-1]
        assert endpoint_id in allowed_ids, f'{module_id} -> {endpoint_id}'


def test_all_frontend_visible_algorithm_ids_are_registered_or_aliased():
    allowed_ids = _registered_module_ids() | set(MODULE_ALIASES) | HIDDEN_COMPAT_MODULES
    content_ids = set(_algorithm_content()) - {'common'}
    visible_ids = (content_ids | _homepage_algorithm_ids() | _nn_interactive_ids() | _html_query_module_refs()) - HIDDEN_COMPAT_MODULES

    assert visible_ids
    assert visible_ids <= allowed_ids, sorted(visible_ids - allowed_ids)


def test_registered_modules_have_openable_frontend_pages():
    app = create_app()
    client = app.test_client()
    payload = client.get('/api/modules').get_json()

    for phase in payload['phases']:
        for module in phase['modules']:
            page = module.get('page')
            assert page, module['id']
            page_name = page.split('?', 1)[0]
            assert Path('static/pages', page_name).exists(), f'{module["id"]}: {page}'


def test_static_demo_endpoint_references_are_known_modules_or_status_routes():
    allowed_ids = _registered_module_ids() | set(MODULE_ALIASES) | HIDDEN_COMPAT_MODULES
    status_routes = {'vision-real-status'}
    refs = set()
    for path in list(_formal_static_pages()) + list(Path('static/js').glob('*.js')):
        if path.name in DEBUG_OR_EXPERIMENTAL_PAGES:
            continue
        refs.update(re.findall(r"/api/demo/([A-Za-z0-9_-]+)", path.read_text(encoding='utf-8')))

    assert refs
    assert refs <= allowed_ids | status_routes, sorted(refs - allowed_ids - status_routes)

    app = create_app()
    client = app.test_client()
    resp = client.get('/api/demo/vision-real-status')
    payload = resp.get_json()
    assert resp.status_code == 200
    assert payload['ready'] is True
    assert {'detection', 'semantic', 'instance'} <= set(payload['modules'])


def test_nn_interactive_referenced_modules_have_demo_backends():
    allowed_ids = _registered_module_ids() | set(MODULE_ALIASES) | HIDDEN_COMPAT_MODULES
    data_keys = _nn_interactive_ids()

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


def test_formal_pages_load_shared_formula_renderer():
    for page in _formal_static_pages():
        text = page.read_text(encoding='utf-8')
        if 'theme-sync.js' not in text:
            continue
        assert 'page-utils.js' in text, page


def test_formal_pages_do_not_crop_step_images_with_cover():
    files = _formal_static_pages() + [
        Path('static/css/ai-eye-page.css'),
        Path('static/css/detection-segmentation.css'),
        Path('static/css/model-demo-page.css'),
        Path('static/css/iframe-theme.css'),
    ]
    for path in files:
        text = path.read_text(encoding='utf-8')
        assert 'object-fit: cover' not in text
        assert 'object-fit:cover' not in text


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
