(function () {
  'use strict';

  var cfg = window.ModelDemoConfig || {};
  var moduleId = cfg.moduleId || new URLSearchParams(location.search).get('id') || 'resnet';
  var state = {
    file: null,
    result: null,
    currentStep: 0,
    playing: false,
    playTimer: null,
    dynamicParams: {}
  };

  var $ = function (id) { return document.getElementById(id); };
  var els = {
    file: $('fileInput'),
    fileName: $('fileName'),
    choose: $('chooseBtn'),
    sample: $('sampleBtn'),
    run: $('runBtn'),
    loading: $('loading'),
    error: $('errorBox'),
    metrics: $('metricsGrid'),
    steps: $('stepsGrid'),
    status: $('statusPill'),
    params: $('paramsPanel'),
    intro: $('introText')
  };

  init();

  function init() {
    document.title = (cfg.title || moduleId) + ' - CV 通识教育';
    document.body.classList.add('model-' + moduleId);
    setText('pageTitle', cfg.title || moduleId);
    setText('subtitle', cfg.subtitle || '');
    if (els.intro) els.intro.textContent = cfg.intro || '';
    if (els.status) els.status.textContent = cfg.status || '后端真实计算';
    renderStaticSections();
    renderParams();
    ensureTeachingStage();
    bind();
    runDemo();
  }

  function bind() {
    if (els.choose && els.file) {
      els.choose.addEventListener('click', function () { els.file.click(); });
      els.file.addEventListener('change', function () {
        state.file = els.file.files && els.file.files[0] ? els.file.files[0] : null;
        if (els.fileName) {
          els.fileName.textContent = state.file ? state.file.name : '未选择文件，将使用内置样例';
        }
      });
    }
    if (els.sample) {
      els.sample.addEventListener('click', function () {
        state.file = null;
        if (els.file) els.file.value = '';
        if (els.fileName) els.fileName.textContent = '使用内置样例';
        runDemo();
      });
    }
    if (els.run) els.run.addEventListener('click', runDemo);
    $('stagePlayBtn').addEventListener('click', togglePlay);
    $('stagePrevBtn').addEventListener('click', function () { stopPlay(); showStep(state.currentStep - 1); });
    $('stageNextBtn').addEventListener('click', function () { stopPlay(); showStep(state.currentStep + 1); });
    $('stageStepSlider').addEventListener('input', function () {
      stopPlay();
      showStep(Number($('stageStepSlider').value || 0));
    });
    $('stageImage').addEventListener('click', onStageImageClick);
  }

  function ensureTeachingStage() {
    if ($('teachingStage')) return;
    if (!els.metrics || !els.metrics.parentNode) return;
    var section = document.createElement('section');
    section.id = 'teachingStage';
    section.className = 'panel teaching-stage';
    section.innerHTML = [
      '<div class="stage-layout">',
      '<aside class="stage-rail"><p class="rail-label">过程线</p><div id="stageRail"></div></aside>',
      '<div class="stage-main">',
      '<div class="stage-media-wrap">',
      '<img id="stageImage" class="stage-image" alt="算法主舞台图像">',
      '<div id="stageEmpty" class="stage-empty">等待后端真实结果</div>',
      '</div>',
      '<div class="stage-controls">',
      '<button id="stagePlayBtn" type="button">播放过程</button>',
      '<button id="stagePrevBtn" type="button">上一步</button>',
      '<input id="stageStepSlider" type="range" min="0" max="0" value="0">',
      '<button id="stageNextBtn" type="button">下一步</button>',
      '<span id="stageStepName">等待步骤</span>',
      '</div>',
      '<div id="interactionPanel" class="interaction-panel"></div>',
      '</div>',
      '<aside class="stage-explainer">',
      '<p class="eyebrow">当前步骤</p>',
      '<h2 id="stageTitle">等待运行</h2>',
      '<p id="stageText">上传图片或使用样例后，后端会返回真实中间步骤。点击步骤卡、拖动滑块或播放过程，可以看到算法如何一步步产生结果。</p>',
      '<code id="stageFormula">Y=f(X)</code>',
      '<dl id="stageData"><dt>状态</dt><dd>等待后端</dd></dl>',
      '</aside>',
      '</div>'
    ].join('');
    els.metrics.parentNode.insertBefore(section, els.metrics);
  }

  function renderParams() {
    if (!els.params || !cfg.params || !cfg.params.length) return;
    els.params.innerHTML = cfg.params.map(function (p) {
      var value = p.value == null ? '' : p.value;
      if (p.type === 'range') {
        return '<label class="param"><span>' + esc(p.label) + '</span><input id="param_' + esc(p.name) +
          '" type="range" min="' + esc(p.min) + '" max="' + esc(p.max) + '" step="' + esc(p.step || 1) +
          '" value="' + esc(value) + '"><strong id="param_' + esc(p.name) + '_value">' + esc(formatParam(p, value)) +
          '</strong></label>';
      }
      if (p.type === 'select') {
        return '<label class="param"><span>' + esc(p.label) + '</span><select id="param_' + esc(p.name) + '">' +
          (p.options || []).map(function (opt) {
            var val = typeof opt === 'string' ? opt : opt.value;
            var label = typeof opt === 'string' ? opt : opt.label;
            return '<option value="' + esc(val) + '"' + (String(val) === String(value) ? ' selected' : '') + '>' +
              esc(label) + '</option>';
          }).join('') + '</select></label>';
      }
      return '<label class="param text-param"><span>' + esc(p.label) + '</span><input id="param_' + esc(p.name) +
        '" type="text" value="' + esc(value) + '"></label>';
    }).join('');
    cfg.params.forEach(function (p) {
      var input = $('param_' + p.name);
      var value = $('param_' + p.name + '_value');
      if (!input) return;
      if (value) input.addEventListener('input', function () {
        delete state.dynamicParams[p.name];
        value.textContent = formatParam(p, input.value);
      });
      if (p.live) input.addEventListener('change', runDemo);
    });
  }

  function renderStaticSections() {
    renderList('pipelineList', cfg.pipeline);
    renderList('applicationsList', cfg.applications);
    renderList('limitationsList', cfg.limitations);
  }

  function renderList(id, items) {
    var node = $(id);
    if (node && items) node.innerHTML = items.map(function (item) { return '<li>' + esc(item) + '</li>'; }).join('');
  }

  function runDemo() {
    clearError();
    stopPlay();
    setBusy(true);
    var form = new FormData();
    if (state.file) form.append('file', state.file);
    (cfg.params || []).forEach(function (p) {
      if (Object.prototype.hasOwnProperty.call(state.dynamicParams, p.name)) return;
      var input = $('param_' + p.name);
      if (input) form.append(p.name, normalizeParam(p, input.value));
    });
    Object.keys(state.dynamicParams).forEach(function (key) {
      form.append(key, state.dynamicParams[key]);
    });
    fetch('/api/demo/' + encodeURIComponent(moduleId), { method: 'POST', body: form })
      .then(function (res) {
        return res.json().catch(function () { return {}; }).then(function (json) {
          json._status = res.status;
          if (!res.ok && json.error) throw new Error(json.error);
          return json;
        });
      })
      .then(function (json) {
        state.result = json;
        state.currentStep = Math.max(0, (json.steps || []).length - 1);
        setBusy(false);
        renderResult(json);
        if (json.error) showError(json.error);
      })
      .catch(function (err) {
        setBusy(false);
        showError('请求失败：' + (err.message || err));
      });
  }

  function renderResult(json) {
    renderMetrics(json.metrics || {}, json.implementation || {});
    renderInteractionPanel(json);
    renderSteps(json.steps || []);
    renderRail(json.steps || []);
    setupStage(json.steps || []);
  }

  function setupStage(steps) {
    var slider = $('stageStepSlider');
    slider.max = String(Math.max(0, steps.length - 1));
    slider.value = String(state.currentStep);
    if (steps.length) showStep(state.currentStep);
  }

  function showStep(index) {
    var steps = state.result && state.result.steps ? state.result.steps : [];
    if (!steps.length) return;
    index = Math.max(0, Math.min(index, steps.length - 1));
    state.currentStep = index;
    var step = steps[index];
    var image = $('stageImage');
    var empty = $('stageEmpty');
    if (step.image_base64) {
      image.src = 'data:image/png;base64,' + step.image_base64;
      empty.hidden = true;
    } else {
      image.removeAttribute('src');
      empty.hidden = false;
    }
    setText('stageTitle', step.name || step.id || '过程步骤');
    setText('stageText', step.explanation || '该步骤来自后端真实计算。');
    setFormula('stageFormula', step.formula || cfg.formula || 'Y=f(X)');
    setText('stageStepName', step.name || step.id || '过程步骤');
    $('stageStepSlider').value = String(index);
    renderStageData(step.data || {});
    updateActiveCards('.step-card', index);
    updateActiveCards('.rail-step', index);
  }

  function renderStageData(data) {
    var node = $('stageData');
    var keys = Object.keys(data || {}).slice(0, 8);
    if (!keys.length) {
      node.innerHTML = '<dt>数据</dt><dd>暂无结构化数据</dd>';
      return;
    }
    node.innerHTML = keys.map(function (key) {
      return '<dt>' + esc(key) + '</dt><dd>' + esc(formatValue(data[key])) + '</dd>';
    }).join('');
  }

  function togglePlay() {
    var steps = state.result && state.result.steps ? state.result.steps : [];
    if (!steps.length) return;
    if (state.playing) {
      stopPlay();
      return;
    }
    state.playing = true;
    $('stagePlayBtn').textContent = '暂停';
    showStep(0);
    state.playTimer = setInterval(function () {
      if (state.currentStep >= steps.length - 1) {
        stopPlay();
        return;
      }
      showStep(state.currentStep + 1);
    }, Number(cfg.frameDelay || 1050));
  }

  function stopPlay() {
    state.playing = false;
    if ($('stagePlayBtn')) $('stagePlayBtn').textContent = '播放过程';
    if (state.playTimer) clearInterval(state.playTimer);
    state.playTimer = null;
  }

  function renderInteractionPanel(json) {
    var panel = $('interactionPanel');
    if (!panel) return;
    var html = '';
    if (moduleId === 'resnet') {
      var top5 = findData(json.steps, 'top5') || [];
      if (top5.length) {
        html += '<div class="interaction-group"><strong>点击类别重新生成 Grad-CAM</strong><div class="chip-row">' +
          top5.map(function (item, index) {
            var rank = index + 1;
            var active = String(state.dynamicParams.target_rank || '1') === String(rank);
            return '<button class="chip' + (active ? ' active' : '') + '" type="button" data-param="target_rank" data-value="' +
              rank + '">' + esc(rank + '. ' + item.label + ' ' + pct(item.probability)) + '</button>';
          }).join('') + '</div></div>';
      }
    }
    if (moduleId === 'detr') {
      var queries = findData(json.steps, 'top_queries') || [];
      if (queries.length) {
        html += '<div class="interaction-group"><strong>选择 Object Query 查看注意力</strong><div class="chip-row">' +
          queries.slice(0, 12).map(function (q) {
            var active = String(state.dynamicParams.query_idx || '') === String(q.query);
            return '<button class="chip' + (active ? ' active' : '') + '" type="button" data-param="query_idx" data-value="' +
              esc(q.query) + '">q' + esc(q.query) + ' · ' + esc(q.label || 'object') + ' · ' + esc(formatNumber(q.score)) + '</button>';
          }).join('') + '</div></div>';
      }
    }
    if (moduleId === 'clip') {
      var preds = json.predictions || findData(json.steps, 'predictions') || [];
      if (preds.length) {
        html += '<div class="interaction-group"><strong>候选文本得分</strong><div class="score-stack">' +
          preds.map(function (p) {
            return '<button class="score-row" type="button" data-step-id="similarity_chart"><span>' +
              esc(p.label) + '</span><i style="--w:' + Math.max(4, Math.min(100, Number(p.probability || 0) * 100)).toFixed(1) +
              '%"></i><b>' + pct(p.probability) + '</b></button>';
          }).join('') + '</div></div>';
      }
    }
    if (moduleId === 'vit') {
      html += '<div class="interaction-group"><strong>点击图片选择 Patch</strong><p>也可以直接点击主舞台图像。页面会把点击位置换算成 patch 编号并重跑后端，查看该 patch token 的真实注意力热力图。</p></div>';
    }
    if (moduleId === 'nerf') {
      html += '<div class="interaction-group"><strong>交互说明</strong><p>拖动视角、分辨率和每条射线采样点数后重渲染，主舞台会显示真实 ray marching 和体渲染结果。</p></div>';
    }
    if (moduleId === 'gan' || moduleId === 'diffusion') {
      html += '<div class="interaction-group"><strong>时间线</strong><p>播放过程或拖动步骤滑块，观察后端真实训练/加噪轨迹如何变化。</p></div>';
    }
    panel.innerHTML = html;
    Array.prototype.forEach.call(panel.querySelectorAll('[data-param]'), function (button) {
      button.addEventListener('click', function () {
        state.dynamicParams[button.dataset.param] = button.dataset.value;
        runDemo();
      });
    });
    Array.prototype.forEach.call(panel.querySelectorAll('[data-step-id]'), function (button) {
      button.addEventListener('click', function () {
        var target = (state.result.steps || []).findIndex(function (step) { return step.id === button.dataset.stepId; });
        if (target >= 0) showStep(target);
      });
    });
  }

  function renderMetrics(metrics, impl) {
    if (!els.metrics) return;
    var rows = [];
    if (impl.status) rows.push(['实现状态', impl.status]);
    if (impl.category) rows.push(['类型', impl.category]);
    if (impl.model) rows.push(['模型/算法', impl.model]);
    if (impl.backend) rows.push(['后端', impl.backend]);
    [
      'status', 'backend', 'model', 'device', 'top1', 'top1_prob', 'top1_conf',
      'num_detections', 'num_queries', 'query_idx', 'head_idx', 'layer',
      'reconstruction_mse', 'samples_per_ray', 'azimuth', 'D_loss_final',
      'G_loss_final', 'iterations', 'resolution', 'steps'
    ].forEach(function (key) {
      if (metrics[key] !== undefined) rows.push([labelMetric(key), metrics[key]]);
    });
    els.metrics.innerHTML = rows.map(function (row) {
      return '<div class="metric"><span>' + esc(row[0]) + '</span><strong>' + esc(formatValue(row[1])) + '</strong></div>';
    }).join('');
  }

  function renderSteps(steps) {
    if (!els.steps) return;
    if (!steps.length) {
      els.steps.innerHTML = '<div class="empty">后端没有返回步骤。若这是预训练模型，请先准备权重或查看错误提示。</div>';
      return;
    }
    els.steps.innerHTML = steps.map(function (step, index) {
      var img = step.image_base64
        ? '<img src="data:image/png;base64,' + step.image_base64 + '" alt="' + esc(step.name || step.id || 'step') + '" loading="lazy">'
        : '<div class="no-image">无图像</div>';
      return '<button class="step-card" type="button" data-index="' + index + '">' +
        '<div class="step-index">' + String(index + 1).padStart(2, '0') + '</div>' +
        '<div class="step-media">' + img + '</div>' +
        '<div class="step-body"><h3>' + esc(step.name || step.id || '步骤') + '</h3>' +
        '<p>' + esc(step.explanation || '') + '</p>' +
        (step.formula ? '<code class="step-formula">' + esc(step.formula) + '</code>' : '') +
        renderDataDetails(step.data) + '</div></button>';
    }).join('');
    if (window.renderLatexIn) window.renderLatexIn(els.steps);
    Array.prototype.forEach.call(els.steps.querySelectorAll('.step-card'), function (card) {
      card.addEventListener('click', function () {
        stopPlay();
        showStep(Number(card.dataset.index || 0));
      });
    });
  }

  function renderRail(steps) {
    var rail = $('stageRail');
    if (!rail) return;
    rail.innerHTML = steps.map(function (step, index) {
      return '<button class="rail-step" type="button" data-index="' + index + '"><span>' +
        String(index + 1).padStart(2, '0') + '</span><b>' + esc(step.name || step.id || '步骤') + '</b></button>';
    }).join('');
    Array.prototype.forEach.call(rail.querySelectorAll('.rail-step'), function (button) {
      button.addEventListener('click', function () {
        stopPlay();
        showStep(Number(button.dataset.index || 0));
      });
    });
  }

  function renderDataDetails(data) {
    if (!data || typeof data !== 'object') return '';
    var keys = Object.keys(data).slice(0, 6);
    if (!keys.length) return '';
    return '<details><summary>中间数据</summary><dl>' + keys.map(function (key) {
      return '<dt>' + esc(key) + '</dt><dd>' + esc(formatValue(data[key])) + '</dd>';
    }).join('') + '</dl></details>';
  }

  function setBusy(flag) {
    if (els.loading) els.loading.hidden = !flag;
    if (els.run) els.run.disabled = flag;
    if (els.sample) els.sample.disabled = flag;
    if (els.choose) els.choose.disabled = flag;
  }

  function showError(msg) {
    if (!els.error) return;
    els.error.hidden = false;
    els.error.textContent = String(msg || '未知错误');
  }

  function clearError() {
    if (!els.error) return;
    els.error.hidden = true;
    els.error.textContent = '';
  }

  function normalizeParam(p, value) {
    if (p.scale) return Number(value) / p.scale;
    return value;
  }

  function formatParam(p, value) {
    if (p.scale) return (Number(value) / p.scale).toFixed(p.fixed == null ? 2 : p.fixed);
    if (p.suffix) return value + p.suffix;
    return value;
  }

  function labelMetric(key) {
    var map = {
      status: '状态',
      backend: '后端',
      model: '模型',
      device: '设备',
      top1: 'Top-1',
      top1_prob: 'Top-1 概率',
      top1_conf: 'Top-1 置信度',
      num_detections: '检测数',
      num_queries: 'Query 数',
      query_idx: 'Query',
      head_idx: '注意力头',
      layer: '层',
      reconstruction_mse: '重建 MSE',
      samples_per_ray: '每射线采样',
      azimuth: '视角',
      D_loss_final: '最终 D loss',
      G_loss_final: '最终 G loss',
      iterations: '迭代轮数',
      resolution: '分辨率',
      steps: '时间步'
    };
    return map[key] || key;
  }

  function findData(steps, key) {
    steps = steps || [];
    for (var i = 0; i < steps.length; i += 1) {
      if (steps[i].data && steps[i].data[key] !== undefined) return steps[i].data[key];
    }
    return null;
  }

  function updateActiveCards(selector, index) {
    Array.prototype.forEach.call(document.querySelectorAll(selector), function (card) {
      card.classList.toggle('active', Number(card.dataset.index || 0) === index);
    });
  }

  function onStageImageClick(event) {
    if (moduleId !== 'vit' || !state.result) return;
    var img = $('stageImage');
    if (!img || !img.naturalWidth || !img.naturalHeight) return;
    var patches = Number((state.result.metrics || {}).patches || 196);
    var grid = Math.max(1, Math.round(Math.sqrt(patches)));
    var rect = img.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    var x = Math.max(0, Math.min(rect.width - 1, event.clientX - rect.left));
    var y = Math.max(0, Math.min(rect.height - 1, event.clientY - rect.top));
    var col = Math.min(grid - 1, Math.floor(x / rect.width * grid));
    var row = Math.min(grid - 1, Math.floor(y / rect.height * grid));
    var patch = row * grid + col;
    state.dynamicParams.selected_patch = String(patch);
    var input = $('param_selected_patch');
    var value = $('param_selected_patch_value');
    if (input) input.value = String(patch);
    if (value) value.textContent = String(patch);
    setText('stageText', '已选择 Patch ' + patch + '，正在用真实 ViT 注意力重新计算。');
    runDemo();
  }

  function setText(id, value) {
    var node = $(id);
    if (node) node.textContent = value;
  }

  function setFormula(id, value) {
    var node = $(id);
    if (!node) return;
    if (window.renderLatex) window.renderLatex(node, value, { display: true });
    else node.textContent = value;
  }

  function pct(value) {
    var n = Number(value || 0);
    return (n * 100).toFixed(n >= 0.1 ? 1 : 2) + '%';
  }

  function formatNumber(value) {
    var n = Number(value || 0);
    return n >= 10 ? n.toFixed(1) : n.toFixed(3);
  }

  function formatValue(value) {
    if (Array.isArray(value)) {
      if (value.length > 6) return 'len=' + value.length + ' ' + JSON.stringify(value.slice(0, 3));
      return JSON.stringify(value);
    }
    if (value && typeof value === 'object') return JSON.stringify(value).slice(0, 180);
    if (typeof value === 'number') return Math.round(value * 10000) / 10000;
    return value == null ? '' : String(value);
  }

  function esc(value) {
    if (window.esc) return window.esc(value);
    return String(value == null ? '' : value).replace(/[&<>"']/g, function (ch) {
      return ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[ch];
    });
  }
}());
