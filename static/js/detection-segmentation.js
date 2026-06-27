(function () {
  'use strict';

  var state = {
    task: initialTask(),
    model: initialModel(),
    file: null,
    models: null,
    result: null,
    filter: 'all'
  };

  var el = {
    taskBtns: Array.prototype.slice.call(document.querySelectorAll('.task-btn')),
    filterBtns: Array.prototype.slice.call(document.querySelectorAll('.filter-btn')),
    modelSelect: document.getElementById('modelSelect'),
    modelHint: document.getElementById('modelHint'),
    fileInput: document.getElementById('fileInput'),
    uploadBtn: document.getElementById('uploadBtn'),
    fileLine: document.getElementById('fileLine'),
    sampleBtn: document.getElementById('sampleBtn'),
    modelsBtn: document.getElementById('modelsBtn'),
    runBtn: document.getElementById('runBtn'),
    score: document.getElementById('scoreThreshold'),
    scoreValue: document.getElementById('scoreValue'),
    mask: document.getElementById('maskThreshold'),
    maskValue: document.getElementById('maskValue'),
    opacity: document.getElementById('overlayOpacity'),
    opacityValue: document.getElementById('opacityValue'),
    loading: document.getElementById('loading'),
    error: document.getElementById('errorBox'),
    empty: document.getElementById('emptyBox'),
    summary: document.getElementById('summaryGrid'),
    steps: document.getElementById('stepsGrid'),
    metrics: document.getElementById('metrics'),
    modelStrip: document.getElementById('modelStrip')
  };

  bind();
  loadModels().then(function () {
    applyTask(state.task);
    runAiEye();
  });

  function initialTask() {
    var params = new URLSearchParams(location.search);
    var raw = params.get('module') || params.get('task') || params.get('tab') || location.hash.replace('#', '');
    if (['detection', 'semantic', 'instance', 'yolo', 'unet'].indexOf(raw) >= 0) return raw;
    return 'all';
  }

  function initialModel() {
    var params = new URLSearchParams(location.search);
    return params.get('model') || '';
  }

  function bind() {
    el.taskBtns.forEach(function (btn) {
      btn.addEventListener('click', function () {
        applyTask(btn.getAttribute('data-task'));
      });
    });
    el.filterBtns.forEach(function (btn) {
      btn.addEventListener('click', function () {
        state.filter = btn.getAttribute('data-filter');
        renderSteps();
        el.filterBtns.forEach(function (item) {
          item.classList.toggle('active', item === btn);
        });
      });
    });
    el.modelSelect.addEventListener('change', function () {
      state.model = el.modelSelect.value;
      renderModelHint();
    });
    el.uploadBtn.addEventListener('click', function () { el.fileInput.click(); });
    el.fileInput.addEventListener('change', function () {
      state.file = el.fileInput.files && el.fileInput.files[0] ? el.fileInput.files[0] : null;
      el.fileLine.textContent = state.file ? state.file.name : '未选择图片，将使用内置示例图。';
    });
    el.sampleBtn.addEventListener('click', function () {
      state.file = null;
      el.fileInput.value = '';
      el.fileLine.textContent = '使用内置示例图。';
      runAiEye();
    });
    el.modelsBtn.addEventListener('click', function () {
      loadModels(true).then(function () {
        applyTask(state.task);
      });
    });
    el.runBtn.addEventListener('click', runAiEye);
    el.score.addEventListener('input', function () {
      el.scoreValue.textContent = value01(el.score);
    });
    el.mask.addEventListener('input', function () {
      el.maskValue.textContent = value01(el.mask);
    });
    el.opacity.addEventListener('input', function () {
      el.opacityValue.textContent = value01(el.opacity);
      renderSummary();
    });
  }

  function loadModels(force) {
    if (state.models && !force) return Promise.resolve(state.models);
    return fetch('/api/ai-eye/models')
      .then(function (res) {
        return res.json().then(function (json) {
          if (!res.ok) throw new Error(json.error || ('HTTP ' + res.status));
          state.models = json;
          renderModelStrip();
          return json;
        });
      })
      .catch(function (err) {
        showError('模型清单读取失败：' + (err.message || err));
      });
  }

  function applyTask(task) {
    state.task = task || 'all';
    el.taskBtns.forEach(function (btn) {
      btn.classList.toggle('active', btn.getAttribute('data-task') === state.task);
    });
    renderModelOptions();
    renderModelHint();
    var urlTask = state.task === 'all' ? 'ai_eye' : state.task;
    history.replaceState(null, '', '?module=' + encodeURIComponent(urlTask));
  }

  function renderModelOptions() {
    if (!state.models) return;
    if (state.task === 'yolo' || state.task === 'unet') {
      var label = state.task === 'yolo' ? 'YOLO-style NumPy grid detector' : 'U-Net-style NumPy encoder-decoder';
      el.modelSelect.innerHTML = '<option value="local_mechanism">' + esc(label) + '</option>';
      el.modelSelect.disabled = true;
      el.modelHint.textContent = state.task === 'yolo'
        ? 'YOLO 页面运行后端单阶段网格机制，展示真实中间结果；不冒充官方 YOLO 预训练权重。'
        : 'U-Net 页面运行后端编解码与跳跃连接机制，展示真实中间结果；不冒充训练好的 U-Net 权重。';
      return;
    }
    var task = state.task === 'all' ? 'detection' : state.task;
    var ids = state.models.tasks[task] || [];
    var defaultId = state.models.defaults[task] || ids[0] || '';
    if (!ids.includes(state.model)) state.model = defaultId;
    el.modelSelect.innerHTML = ids.map(function (id) {
      var model = state.models.models[id];
      var cached = model.cached ? '已缓存' : '待下载';
      return '<option value="' + esc(id) + '"' + (id === state.model ? ' selected' : '') + '>' +
        esc(model.name + ' · ' + cached) + '</option>';
    }).join('');
    el.modelSelect.disabled = state.task === 'all';
    if (state.task === 'all') {
      el.modelHint.textContent = '三视角总览会使用每个任务的默认模型；切到单任务后可切换该任务模型。';
    }
  }

  function renderModelHint() {
    if (!state.models || state.task === 'all' || state.task === 'yolo' || state.task === 'unet') return;
    var model = state.models.models[state.model];
    if (!model) return;
    el.modelHint.textContent = model.description + ' 优势：' + model.strengths + ' 局限：' + model.limitations;
  }

  function runAiEye() {
    clearError();
    setBusy(true);
    var form = new FormData();
    var localMechanism = state.task === 'yolo' || state.task === 'unet';
    var endpoint = localMechanism ? ('/api/demo/' + state.task) : '/api/demo/ai_eye';
    if (!localMechanism) form.append('task', state.task);
    if (!localMechanism && state.task !== 'all' && state.model) form.append('model', state.model);
    form.append('score_threshold', value01(el.score));
    form.append('threshold', value01(el.score));
    form.append('mask_threshold', value01(el.mask));
    form.append('return_steps', '1');
    if (state.file) form.append('file', state.file);
    fetch(endpoint, { method: 'POST', body: form })
      .then(function (res) {
        return res.json().catch(function () { return {}; }).then(function (json) {
          if (!res.ok) throw new Error(json.error || (json.metrics && json.metrics.error) || ('HTTP ' + res.status));
          return json;
        });
      })
      .then(function (json) {
        state.result = json;
        if (json.models) state.models = json.models;
        setBusy(false);
        renderAll();
      })
      .catch(function (err) {
        setBusy(false);
        showError('运行失败：' + (err.message || err));
      });
  }

  function renderAll() {
    el.empty.style.display = 'none';
    renderMetrics();
    renderSummary();
    renderSteps();
    renderModelStrip();
  }

  function renderMetrics() {
    var metrics = state.result && state.result.metrics ? state.result.metrics : {};
    var keys = ['status', 'tasks_succeeded', 'tasks_failed', 'elapsed_ms', 'score_threshold', 'mask_threshold'];
    el.metrics.innerHTML = keys.filter(function (key) { return metrics[key] !== undefined; }).map(function (key) {
      return '<span class="metric">' + esc(labelMetric(key)) + ': <strong>' + esc(formatMetric(metrics[key])) + '</strong></span>';
    }).join('');
  }

  function renderSummary() {
    var result = state.result || {};
    var steps = result.steps || [];
    if (result.module_id === 'yolo' || result.module_id === 'unet') {
      el.summary.classList.add('single');
      var finalIds = result.module_id === 'yolo' ? ['detections', 'score_filter'] : ['mask', 'skip_comparison'];
      var title = result.module_id === 'yolo' ? 'YOLO范式：一阶段检测' : 'U-Net机制：跳跃连接分割';
      var image = findStepImage(steps, finalIds);
      el.summary.innerHTML = '<article class="summary-card"><header><strong>' + esc(title) + '</strong><span>' +
        esc(result.metrics && result.metrics.status || 'local_mechanism') + '</span></header><div class="media">' +
        (image ? '<img src="data:image/png;base64,' + image + '" alt="' + esc(title) + '">' : '<span class="empty">等待后端结果</span>') +
        '</div><p>' + esc(result.metrics && result.metrics.note || '结果来自后端真实计算的机制管线。') + '</p></article>';
      return;
    }
    el.summary.classList.remove('single');
    var cards = [
      {
        task: 'detection',
        title: '目标检测',
        image: findStepImage(steps, ['detection_detections']),
        text: summarizeTask('detection')
      },
      {
        task: 'semantic',
        title: '语义分割',
        image: findStepImage(steps, ['semantic_overlay', 'semantic_label_map']),
        text: summarizeTask('semantic')
      },
      {
        task: 'instance',
        title: '实例分割',
        image: findStepImage(steps, ['instance_masks']),
        text: summarizeTask('instance')
      }
    ];
    var opacity = Number(el.opacity.value) / 100;
    el.summary.innerHTML = cards.map(function (card) {
      var image = card.image
        ? '<img src="data:image/png;base64,' + card.image + '" alt="' + esc(card.title) + '" style="opacity:' + opacity.toFixed(2) + '">'
        : '<span class="empty">尚未运行该视角</span>';
      return '<article class="summary-card"><header><strong>' + esc(card.title) + '</strong><span>' +
        esc(taskOutput(card.task)) + '</span></header><div class="media">' + image + '</div><p>' +
        esc(card.text) + '</p></article>';
    }).join('');
  }

  function summarizeTask(task) {
    var alg = state.result && state.result.algorithms ? state.result.algorithms[task] : null;
    if (!alg) return '当前运行范围未包含该任务。';
    if (alg.error) return alg.error;
    var out = alg.output || {};
    if (task === 'detection') {
      var detections = out.detections || [];
      return '检测到 ' + detections.length + ' 个高于阈值的目标，输出类别、置信度和边界框。';
    }
    if (task === 'semantic') {
      var labels = out.labels || [];
      return '分割出 ' + labels.length + ' 个主要语义类别，每个像素都有类别预测。';
    }
    var instances = out.instances || [];
    return '分离出 ' + instances.length + ' 个实例，每个实例都有独立 mask、框和面积。';
  }

  function renderSteps() {
    if (!state.result) return;
    var all = state.result.steps || [];
    var filtered = all.filter(function (step) {
      if (state.filter === 'all') return true;
      if ((state.result.module_id === 'yolo' || state.result.module_id === 'unet') && state.filter === state.result.module_id) return true;
      return String(step.id || '').indexOf(state.filter + '_') === 0;
    });
    el.steps.innerHTML = filtered.map(function (step, index) {
      var image = step.image_base64
        ? '<img src="data:image/png;base64,' + step.image_base64 + '" alt="' + esc(step.name || step.id || 'step') + '" loading="lazy">'
        : '<span>无图像</span>';
      return '<article class="step-card"><div class="img-area">' + image + '</div><div class="body"><strong>' +
        esc((index + 1) + '. ' + (step.name || step.id || 'Step')) + '</strong><p>' +
        esc(step.explanation || '') + '</p>' + (step.formula ? '<code class="step-formula">' + esc(step.formula) + '</code>' : '') +
        renderMiniData(step.data) + '</div></article>';
    }).join('');
  }

  function renderMiniData(data) {
    if (!data || typeof data !== 'object') return '';
    var parts = Object.keys(data).slice(0, 4).map(function (key) {
      var value = data[key];
      if (Array.isArray(value)) return key + ': ' + value.length;
      if (value && typeof value === 'object') return key + ': ' + Object.keys(value).length + '项';
      return key + ': ' + formatMetric(value);
    });
    return parts.length ? '<div class="data-mini">' + esc(parts.join(' · ')) + '</div>' : '';
  }

  function renderModelStrip() {
    if (!state.models) return;
    var ids = [];
    ['detection', 'semantic', 'instance'].forEach(function (task) {
      var def = state.models.defaults[task];
      if (def) ids.push(def);
    });
    if (state.task !== 'all' && state.model && ids.indexOf(state.model) < 0) ids.unshift(state.model);
    var html = ids.map(function (id) {
      var model = state.models.models[id];
      if (!model) return '';
      return '<article class="model-card"><strong>' + esc(model.name) + '</strong><span>' +
        esc(taskName(model.task) + ' · ' + (model.cached ? '权重已缓存' : '首次运行会下载')) + '</span><p>' +
        esc(model.description) + '</p><p>权重：' + esc(model.weights) + '</p><p>准备命令：<code>' +
        esc(model.download_command) + '</code></p></article>';
    }).join('');
    html += '<article class="model-card"><strong>YOLO范式机制版</strong><span>本地机制 · 无外部权重</span><p>后端真实计算网格、目标性、候选框和重叠抑制，用于解释单阶段检测，不冒充官方 YOLO 权重。</p><p>接口：<code>/api/demo/yolo</code></p></article>';
    html += '<article class="model-card"><strong>U-Net机制版</strong><span>本地机制 · 无外部权重</span><p>后端真实计算编码器、瓶颈、解码器、跳跃融合和 mask，用于解释 U-Net 结构，不冒充训练权重。</p><p>接口：<code>/api/demo/unet</code></p></article>';
    el.modelStrip.innerHTML = html;
  }

  function findStepImage(steps, ids) {
    for (var i = 0; i < ids.length; i += 1) {
      var hit = steps.find(function (step) { return step.id === ids[i] && step.image_base64; });
      if (hit) return hit.image_base64;
    }
    return '';
  }

  function setBusy(busy) {
    el.loading.style.display = busy ? 'block' : 'none';
    el.runBtn.disabled = busy;
    el.sampleBtn.disabled = busy;
  }

  function showError(message) {
    el.error.textContent = message;
    el.error.style.display = 'block';
  }

  function clearError() {
    el.error.style.display = 'none';
    el.error.textContent = '';
  }

  function value01(input) {
    return (Number(input.value) / 100).toFixed(2);
  }

  function formatMetric(value) {
    if (typeof value === 'number') return Math.abs(value) >= 10 ? value.toFixed(1) : value.toFixed(3);
    if (value === null || value === undefined) return '';
    if (typeof value === 'object') return JSON.stringify(value);
    return String(value);
  }

  function labelMetric(key) {
    return {
      status: '状态',
      tasks_succeeded: '成功任务',
      tasks_failed: '失败任务',
      elapsed_ms: '耗时 ms',
      score_threshold: '置信度阈值',
      mask_threshold: 'Mask阈值'
    }[key] || key;
  }

  function taskName(task) {
    return { detection: '目标检测', semantic: '语义分割', instance: '实例分割', yolo: 'YOLO范式', unet: 'U-Net机制' }[task] || task;
  }

  function taskOutput(task) {
    return {
      detection: 'box + class + score',
      semantic: 'pixel class map',
      instance: 'box + class + mask'
    }[task] || '';
  }

  function esc(value) {
    return String(value).replace(/[&<>"']/g, function (ch) {
      return ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[ch];
    });
  }
})();
