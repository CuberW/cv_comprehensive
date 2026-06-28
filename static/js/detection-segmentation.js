(function () {
  'use strict';

  var TASK_TEXT = {
    all: '三视角总览',
    detection: '目标检测',
    semantic: '语义分割',
    instance: '实例分割',
    yolo: 'YOLO范式',
    unet: 'U-Net机制'
  };
  var TASK_STEPS = {
    detection: ['detection_detections', 'detection_score_filter', 'detection_objectness'],
    semantic: ['semantic_overlay', 'semantic_label_map', 'semantic_confidence'],
    instance: ['instance_masks', 'instance_mask_threshold', 'instance_detector_stage'],
    yolo: ['detections', 'score_filter', 'objectness'],
    unet: ['mask', 'probability', 'skip_fusion']
  };
  var COLORS = [
    '#22c55e', '#3b82f6', '#a855f7', '#f59e0b', '#14b8a6',
    '#ec4899', '#eab308', '#38bdf8', '#84cc16', '#fb7185'
  ];

  var state = {
    task: initialTask(),
    model: initialModel(),
    file: null,
    models: null,
    result: null,
    view: 'result',
    filter: 'all',
    selectedTask: 'detection',
    selectedObject: null,
    selectedStepIndex: 0,
    playing: false,
    playTimer: null
  };

  var $ = function (id) { return document.getElementById(id); };
  var el = {
    taskBtns: Array.prototype.slice.call(document.querySelectorAll('.task-btn')),
    viewBtns: Array.prototype.slice.call(document.querySelectorAll('.view-btn')),
    filterBtns: Array.prototype.slice.call(document.querySelectorAll('.filter-btn')),
    modelSelect: $('modelSelect'),
    modelHint: $('modelHint'),
    fileInput: $('fileInput'),
    uploadBtn: $('uploadBtn'),
    sampleBtn: $('sampleBtn'),
    modelsBtn: $('modelsBtn'),
    runBtn: $('runBtn'),
    fileLine: $('fileLine'),
    score: $('scoreThreshold'),
    scoreValue: $('scoreValue'),
    mask: $('maskThreshold'),
    maskValue: $('maskValue'),
    opacity: $('overlayOpacity'),
    opacityValue: $('opacityValue'),
    stageTitle: $('stageTitle'),
    stage: $('stage'),
    stageImage: $('stageImage'),
    canvas: $('stageCanvas'),
    stageEmpty: $('stageEmpty'),
    playBtn: $('playBtn'),
    stepSlider: $('stepSlider'),
    stepName: $('stepName'),
    inspectTitle: $('inspectTitle'),
    inspectText: $('inspectText'),
    formula: $('formulaBox'),
    inspectData: $('inspectData'),
    overview: $('overviewGrid'),
    objectList: $('objectList'),
    steps: $('stepsGrid'),
    modelStrip: $('modelStrip'),
    loading: $('loading'),
    error: $('errorBox')
  };
  var ctx = el.canvas.getContext('2d');

  init();

  function init() {
    bind();
    loadModels().then(function () {
      applyTask(state.task);
      runCurrentTask();
    });
  }

  function initialTask() {
    var params = new URLSearchParams(location.search);
    var raw = params.get('module') || params.get('task') || params.get('tab') || location.hash.replace('#', '');
    if (['detection', 'semantic', 'instance', 'yolo', 'unet'].indexOf(raw) >= 0) return raw;
    return 'all';
  }

  function initialModel() {
    return new URLSearchParams(location.search).get('model') || '';
  }

  function bind() {
    el.taskBtns.forEach(function (btn) {
      btn.addEventListener('click', function () {
        applyTask(btn.dataset.task);
      });
    });
    el.viewBtns.forEach(function (btn) {
      btn.addEventListener('click', function () {
        setView(btn.dataset.view);
      });
    });
    el.filterBtns.forEach(function (btn) {
      btn.addEventListener('click', function () {
        state.filter = btn.dataset.filter;
        el.filterBtns.forEach(function (item) {
          item.classList.toggle('active', item === btn);
        });
        renderSteps();
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
      runCurrentTask();
    });
    el.modelsBtn.addEventListener('click', function () {
      loadModels(true).then(function () {
        applyTask(state.task);
      });
    });
    el.runBtn.addEventListener('click', runCurrentTask);
    el.score.addEventListener('input', function () {
      el.scoreValue.textContent = value01(el.score);
      updateThresholdHint();
    });
    el.mask.addEventListener('input', function () {
      el.maskValue.textContent = value01(el.mask);
      updateThresholdHint();
    });
    el.opacity.addEventListener('input', function () {
      el.opacityValue.textContent = value01(el.opacity);
      renderStageResult();
    });
    el.playBtn.addEventListener('click', togglePlayback);
    el.stepSlider.addEventListener('input', function () {
      stopPlayback();
      showStep(Number(el.stepSlider.value));
    });
    el.stageImage.addEventListener('load', syncCanvasToImage);
    window.addEventListener('resize', syncCanvasToImage);
    el.canvas.addEventListener('click', onCanvasClick);
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
    state.selectedTask = state.task === 'all' ? 'detection' : state.task;
    state.selectedObject = null;
    el.taskBtns.forEach(function (btn) {
      btn.classList.toggle('active', btn.dataset.task === state.task);
    });
    renderModelOptions();
    renderModelHint();
    history.replaceState(null, '', '?module=' + encodeURIComponent(state.task === 'all' ? 'ai_eye' : state.task));
    updateThresholdHint();
  }

  function renderModelOptions() {
    if (!state.models) return;
    if (state.task === 'yolo' || state.task === 'unet') {
      var localLabel = state.task === 'yolo' ? 'YOLO-style 本地机制实现' : 'U-Net-style 本地机制实现';
      el.modelSelect.innerHTML = '<option value="local_mechanism">' + esc(localLabel) + '</option>';
      el.modelSelect.disabled = true;
      return;
    }
    var task = state.task === 'all' ? 'detection' : state.task;
    var ids = state.models.tasks[task] || [];
    var defaultId = state.models.defaults[task] || ids[0] || '';
    if (ids.indexOf(state.model) < 0) state.model = defaultId;
    el.modelSelect.innerHTML = ids.map(function (id) {
      var model = state.models.models[id] || {};
      var cached = model.cached ? '已缓存' : '待下载';
      return '<option value="' + esc(id) + '"' + (id === state.model ? ' selected' : '') + '>' +
        esc((model.name || id) + ' · ' + cached) + '</option>';
    }).join('');
    el.modelSelect.disabled = state.task === 'all';
  }

  function renderModelHint() {
    if (!state.models) return;
    if (state.task === 'all') {
      el.modelHint.textContent = '总览会同时运行检测、语义分割、实例分割的默认模型；切到单任务后可切换该任务模型。';
      return;
    }
    if (state.task === 'yolo') {
      el.modelHint.textContent = 'YOLO范式是后端 NumPy/Pillow 本地机制实现：真实计算网格、目标性、候选框和过滤，但不是官方预训练 YOLO 权重。';
      return;
    }
    if (state.task === 'unet') {
      el.modelHint.textContent = 'U-Net机制是后端本地编码器、瓶颈、解码器、跳跃融合和 mask 计算，用来讲清结构，不冒充训练好权重。';
      return;
    }
    var model = state.models.models[state.model];
    if (!model) return;
    el.modelHint.textContent = model.description + ' 优势：' + model.strengths + ' 局限：' + model.limitations;
  }

  function updateThresholdHint() {
    el.scoreValue.textContent = value01(el.score);
    el.maskValue.textContent = value01(el.mask);
    el.opacityValue.textContent = value01(el.opacity);
  }

  function runCurrentTask() {
    clearError();
    stopPlayback();
    setBusy(true);
    var local = state.task === 'yolo' || state.task === 'unet';
    var endpoint = local ? '/api/demo/' + state.task : '/api/demo/ai_eye';
    var form = new FormData();
    if (!local) form.append('task', state.task);
    if (!local && state.task !== 'all' && state.model) form.append('model', state.model);
    form.append('score_threshold', value01(el.score));
    form.append('threshold', value01(el.score));
    form.append('mask_threshold', value01(el.mask));
    form.append('return_steps', '1');
    if (state.file) form.append('file', state.file);

    fetch(endpoint, { method: 'POST', body: form })
      .then(function (res) {
        return res.json().catch(function () { return {}; }).then(function (json) {
          json._status = res.status;
          if (!res.ok) {
            var error = new Error(json.error || (json.metrics && json.metrics.error) || ('HTTP ' + res.status));
            error.payload = json;
            throw error;
          }
          return json;
        });
      })
      .then(function (json) {
        setBusy(false);
        state.result = json;
        if (json.models) state.models = json.models;
        state.selectedTask = defaultSelectedTask(json);
        state.selectedObject = null;
        state.selectedStepIndex = Math.max(0, (json.steps || []).length - 1);
        renderAll();
      })
      .catch(function (err) {
        setBusy(false);
        renderFailurePayload(err.payload);
        showError('运行失败：' + (err.message || err));
      });
  }

  function renderFailurePayload(payload) {
    if (!payload || typeof payload !== 'object') return;
    state.result = payload;
    if (payload.models) state.models = payload.models;
    state.selectedObject = null;
    renderModelOptions();
    renderModelHint();
    renderModelStrip();
    renderOverview();
    renderObjects();
    renderSteps();
    setupTimeline();
    updateInspectorForTask(state.selectedTask);
    el.stageImage.removeAttribute('src');
    el.stageEmpty.classList.remove('is-hidden');
  }

  function defaultSelectedTask(json) {
    if (json.module_id === 'yolo') return 'yolo';
    if (json.module_id === 'unet') return 'unet';
    if (state.task !== 'all') return state.task;
    var outputs = json.outputs || {};
    if (outputs.detection) return 'detection';
    if (outputs.semantic) return 'semantic';
    if (outputs.instance) return 'instance';
    return 'detection';
  }

  function renderAll() {
    el.stageEmpty.classList.add('is-hidden');
    renderOverview();
    renderObjects();
    renderSteps();
    renderModelStrip();
    setupTimeline();
    setView('result');
    renderStageResult();
    updateInspectorForTask(state.selectedTask);
  }

  function setView(view) {
    state.view = view;
    el.viewBtns.forEach(function (btn) {
      btn.classList.toggle('active', btn.dataset.view === view);
    });
    if (view === 'process') {
      showStep(state.selectedStepIndex);
    } else {
      renderStageResult();
      updateInspectorForTask(state.selectedTask);
    }
  }

  function renderOverview() {
    var cards = overviewCards();
    if (!cards.length) {
      el.overview.innerHTML = '<div class="empty-card">当前结果没有可展示的总览图。</div>';
      return;
    }
    el.overview.innerHTML = cards.map(function (card) {
      return '<button class="overview-card' + (card.task === state.selectedTask ? ' active' : '') +
        '" type="button" data-task="' + esc(card.task) + '">' +
        '<header><strong>' + esc(card.title) + '</strong><span>' + esc(card.badge) + '</span></header>' +
        '<div class="media">' + (card.image ? '<img src="data:image/png;base64,' + card.image + '" alt="' + esc(card.title) + '">' : '<span class="empty-card">无图</span>') + '</div>' +
        '<p>' + esc(card.text) + '</p></button>';
    }).join('');
    Array.prototype.forEach.call(el.overview.querySelectorAll('.overview-card'), function (button) {
      button.addEventListener('click', function () {
        state.selectedTask = button.dataset.task;
        state.selectedObject = null;
        setView('result');
        renderOverview();
        renderObjects();
      });
    });
  }

  function overviewCards() {
    if (!state.result) return [];
    if (state.result.module_id === 'yolo') {
      return [{
        task: 'yolo',
        title: 'YOLO范式：网格一次性预测',
        badge: 'local mechanism',
        image: findStepImage(['detections', 'score_filter', 'raw_boxes']),
        text: '后端真实计算网格目标性、候选框、过滤和最终框。'
      }];
    }
    if (state.result.module_id === 'unet') {
      return [{
        task: 'unet',
        title: 'U-Net机制：编码解码与跳跃连接',
        badge: 'local mechanism',
        image: findStepImage(['mask', 'probability', 'skip_fusion']),
        text: '后端真实计算编码器细节、瓶颈、跳跃融合、概率图和最终mask。'
      }];
    }
    return [
      {
        task: 'detection',
        title: '目标检测',
        badge: 'box + class + score',
        image: findStepImage(['detection_detections', 'detection_score_filter']),
        text: summarizeTask('detection')
      },
      {
        task: 'semantic',
        title: '语义分割',
        badge: 'pixel class map',
        image: findStepImage(['semantic_overlay', 'semantic_label_map']),
        text: summarizeTask('semantic')
      },
      {
        task: 'instance',
        title: '实例分割',
        badge: 'box + mask',
        image: findStepImage(['instance_masks', 'instance_mask_threshold']),
        text: summarizeTask('instance')
      }
    ].filter(function (card) {
      return state.task === 'all' || state.task === card.task || (state.result.outputs || {})[card.task];
    });
  }

  function summarizeTask(task) {
    var alg = (state.result.algorithms || {})[task];
    if (!alg) return '当前运行范围未包含该任务。';
    if (alg.error) return alg.error;
    var out = ((state.result.outputs || {})[task]) || alg.output || {};
    if (task === 'detection') return '检测到 ' + ((out.detections || []).length) + ' 个高于阈值的目标。';
    if (task === 'semantic') return '得到 ' + ((out.labels || []).length) + ' 个主要语义类别及其像素占比。';
    return '分离出 ' + ((out.instances || []).length) + ' 个实例，每个实例有独立框和mask统计。';
  }

  function renderObjects() {
    var objects = currentObjects();
    if (!objects.length) {
      el.objectList.innerHTML = '<div class="empty-card">当前视角没有可点击的结构化对象；请查看过程步骤。</div>';
      return;
    }
    el.objectList.innerHTML = objects.map(function (obj, index) {
      var score = obj.score != null ? obj.score : obj.ratio != null ? obj.ratio : obj.area_ratio != null ? obj.area_ratio : 0;
      var active = state.selectedObject && state.selectedObject.key === obj.key ? ' active' : '';
      return '<button class="object-card' + active + '" type="button" data-index="' + index + '">' +
        '<header><strong>' + esc(obj.title) + '</strong><span>' + esc(obj.badge) + '</span></header>' +
        '<p>' + esc(obj.text) + '</p><div class="bars"><div class="mini-bar"><i style="--w:' +
        Math.max(3, Math.min(100, score * 100)).toFixed(1) + '%"></i></div></div></button>';
    }).join('');
    Array.prototype.forEach.call(el.objectList.querySelectorAll('.object-card'), function (button) {
      button.addEventListener('click', function () {
        state.selectedObject = objects[Number(button.dataset.index || 0)];
        setView('result');
        renderObjects();
        renderStageResult();
        updateInspectorForObject(state.selectedObject);
      });
    });
  }

  function currentObjects() {
    if (!state.result) return [];
    if (state.selectedTask === 'detection') {
      return getOutput('detection', 'detections').map(function (det, index) {
        return {
          key: 'detection-' + index,
          type: 'box',
          task: 'detection',
          box: det.box,
          title: det.label || 'object',
          badge: 'score ' + formatNumber(det.score),
          text: '边界框 ' + formatBox(det.box),
          data: det,
          score: Number(det.score || 0)
        };
      });
    }
    if (state.selectedTask === 'instance') {
      return getOutput('instance', 'instances').map(function (inst, index) {
        return {
          key: 'instance-' + index,
          type: 'box',
          task: 'instance',
          box: inst.box,
          title: inst.label || 'instance',
          badge: 'score ' + formatNumber(inst.score),
          text: '面积 ' + (inst.area || 0) + ' 像素，框 ' + formatBox(inst.box),
          data: inst,
          score: Number(inst.score || 0)
        };
      });
    }
    if (state.selectedTask === 'semantic') {
      return getOutput('semantic', 'labels').map(function (label, index) {
        return {
          key: 'semantic-' + index,
          type: 'semantic',
          task: 'semantic',
          title: label.label || ('class ' + label.label_id),
          badge: Math.round((label.ratio || 0) * 1000) / 10 + '%',
          text: '像素数 ' + label.pixels + '，类别 ID ' + label.label_id,
          data: label,
          ratio: Number(label.ratio || 0)
        };
      });
    }
    if (state.selectedTask === 'yolo') {
      return extractStepData('detections', 'detections').map(function (det, index) {
        return {
          key: 'yolo-' + index,
          type: 'box',
          task: 'yolo',
          box: det.box,
          title: det.label || ('候选 ' + (index + 1)),
          badge: 'score ' + formatNumber(det.score),
          text: 'YOLO机制候选框 ' + formatBox(det.box),
          data: det,
          score: Number(det.score || det.objectness || 0)
        };
      });
    }
    return [];
  }

  function renderStageResult() {
    if (!state.result) return;
    state.view = 'result';
    el.canvas.style.pointerEvents = 'auto';
    var focus = selectedFocusImage();
    var image = focus || stageImageForTask(state.selectedTask);
    if (!image) {
      el.stageImage.removeAttribute('src');
      el.stageEmpty.classList.remove('is-hidden');
      return;
    }
    el.stageEmpty.classList.add('is-hidden');
    el.stageImage.style.opacity = focus ? '1' : String(Number(el.opacity.value) / 100);
    el.stageImage.src = 'data:image/png;base64,' + image;
    el.stageTitle.textContent = focus && state.selectedObject
      ? '聚焦查看：' + state.selectedObject.title
      : (TASK_TEXT[state.selectedTask] || '结果视图');
    syncCanvasToImage();
  }

  function selectedFocusImage() {
    if (!state.selectedObject || !state.selectedObject.data) return '';
    return state.selectedObject.data.focus_image_base64 || state.selectedObject.data.image_base64 || '';
  }

  function stageImageForTask(task) {
    if (task === 'detection') return findStepImage(['detection_detections', 'detection_score_filter']);
    if (task === 'semantic') return findStepImage(['semantic_overlay', 'semantic_label_map']);
    if (task === 'instance') return findStepImage(['instance_masks', 'instance_mask_threshold']);
    if (task === 'yolo') return findStepImage(['detections', 'score_filter']);
    if (task === 'unet') return findStepImage(['mask', 'probability', 'skip_fusion']);
    return findStepImage(['detection_detections', 'semantic_overlay', 'instance_masks', 'detections', 'mask']);
  }

  function syncCanvasToImage() {
    var image = el.stageImage;
    if (!image.complete || !image.naturalWidth || !image.naturalHeight) return;
    var rect = image.getBoundingClientRect();
    var stageRect = el.stage.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    el.canvas.width = Math.max(1, Math.round(rect.width));
    el.canvas.height = Math.max(1, Math.round(rect.height));
    el.canvas.style.width = rect.width + 'px';
    el.canvas.style.height = rect.height + 'px';
    el.canvas.style.left = (rect.left - stageRect.left) + 'px';
    el.canvas.style.top = (rect.top - stageRect.top) + 'px';
    drawStageOverlay();
  }

  function drawStageOverlay() {
    ctx.clearRect(0, 0, el.canvas.width, el.canvas.height);
    var focus = selectedFocusImage();
    var objects = focus && state.selectedObject && state.selectedObject.box
      ? [state.selectedObject]
      : currentObjects().filter(function (obj) { return obj.box; });
    if (!objects.length || !el.stageImage.naturalWidth) return;
    var sx = el.canvas.width / el.stageImage.naturalWidth;
    var sy = el.canvas.height / el.stageImage.naturalHeight;
    objects.forEach(function (obj, index) {
      var box = obj.box;
      var active = state.selectedObject && state.selectedObject.key === obj.key;
      var color = COLORS[index % COLORS.length];
      ctx.lineWidth = active ? 4 : 2;
      ctx.strokeStyle = color;
      ctx.fillStyle = active ? hexToRgba(color, 0.18) : hexToRgba(color, 0.06);
      var x = box[0] * sx;
      var y = box[1] * sy;
      var w = (box[2] - box[0]) * sx;
      var h = (box[3] - box[1]) * sy;
      ctx.fillRect(x, y, w, h);
      ctx.strokeRect(x, y, w, h);
      ctx.fillStyle = color;
      ctx.fillRect(x, Math.max(0, y - 24), Math.min(180, Math.max(70, String(obj.title).length * 8 + 46)), 22);
      ctx.fillStyle = '#fff';
      ctx.font = 'bold 12px sans-serif';
      ctx.fillText(obj.title + ' ' + (obj.data.score ? Number(obj.data.score).toFixed(2) : ''), x + 6, Math.max(14, y - 8));
    });
  }

  function onCanvasClick(event) {
    if (state.view !== 'result' || !el.stageImage.naturalWidth) return;
    var rect = el.canvas.getBoundingClientRect();
    var x = (event.clientX - rect.left) / rect.width * el.stageImage.naturalWidth;
    var y = (event.clientY - rect.top) / rect.height * el.stageImage.naturalHeight;
    var hit = currentObjects().find(function (obj) {
      if (!obj.box) return false;
      return x >= obj.box[0] && x <= obj.box[2] && y >= obj.box[1] && y <= obj.box[3];
    });
    if (hit) {
      state.selectedObject = hit;
      renderObjects();
      drawStageOverlay();
      updateInspectorForObject(hit);
    }
  }

  function setupTimeline() {
    var steps = state.result && state.result.steps ? state.result.steps : [];
    el.stepSlider.max = String(Math.max(0, steps.length - 1));
    el.stepSlider.value = String(Math.max(0, state.selectedStepIndex));
    el.stepName.textContent = steps[state.selectedStepIndex] ? steps[state.selectedStepIndex].name : '等待步骤';
  }

  function showStep(index) {
    var steps = state.result && state.result.steps ? state.result.steps : [];
    if (!steps.length) return;
    index = Math.max(0, Math.min(steps.length - 1, index));
    state.view = 'process';
    state.selectedStepIndex = index;
    var step = steps[index];
    el.stageTitle.textContent = step.name || step.id || '过程步骤';
    el.stepSlider.value = String(index);
    el.stepName.textContent = step.name || step.id || '过程步骤';
    if (step.image_base64) {
      el.stageImage.style.opacity = '1';
      el.stageImage.src = 'data:image/png;base64,' + step.image_base64;
      el.stageEmpty.classList.add('is-hidden');
    }
    ctx.clearRect(0, 0, el.canvas.width, el.canvas.height);
    updateInspectorForStep(step);
  }

  function togglePlayback() {
    var steps = state.result && state.result.steps ? state.result.steps : [];
    if (!steps.length) return;
    if (state.playing) {
      stopPlayback();
      return;
    }
    state.playing = true;
    el.playBtn.textContent = '暂停';
    setView('process');
    var index = 0;
    showStep(index);
    state.playTimer = setInterval(function () {
      index += 1;
      if (index >= steps.length) {
        stopPlayback();
        setView('result');
        return;
      }
      showStep(index);
    }, 1050);
  }

  function stopPlayback() {
    state.playing = false;
    el.playBtn.textContent = '播放过程';
    if (state.playTimer) clearInterval(state.playTimer);
    state.playTimer = null;
  }

  function renderSteps() {
    if (!state.result) return;
    var steps = state.result.steps || [];
    var filtered = steps.filter(function (step) {
      if (state.filter === 'all') return true;
      if ((state.result.module_id === 'yolo' || state.result.module_id === 'unet') && state.filter === state.result.module_id) return true;
      return String(step.id || '').indexOf(state.filter + '_') === 0;
    });
    if (!filtered.length) {
      el.steps.innerHTML = '<div class="empty-card">当前筛选没有步骤。</div>';
      return;
    }
    el.steps.innerHTML = filtered.map(function (step, index) {
      var realIndex = steps.indexOf(step);
      return '<button class="step-card" type="button" data-index="' + realIndex + '">' +
        '<div class="media">' + (step.image_base64 ? '<img src="data:image/png;base64,' + step.image_base64 + '" alt="' + esc(step.name || step.id || 'step') + '">' : '<span class="empty-card">无图像</span>') + '</div>' +
        '<div class="body"><strong>' + esc((index + 1) + '. ' + (step.name || step.id || '步骤')) + '</strong>' +
        '<p>' + esc(step.explanation || '') + '</p>' +
        (step.formula ? '<code class="step-formula">' + esc(step.formula) + '</code>' : '') +
        renderMiniData(step.data) + '</div></button>';
    }).join('');
    if (window.renderLatexIn) window.renderLatexIn(el.steps);
    Array.prototype.forEach.call(el.steps.querySelectorAll('.step-card'), function (button) {
      button.addEventListener('click', function () {
        stopPlayback();
        setView('process');
        showStep(Number(button.dataset.index || 0));
      });
    });
  }

  function renderMiniData(data) {
    if (!data || typeof data !== 'object') return '';
    var parts = Object.keys(data).filter(function (key) {
      return key.indexOf('base64') < 0 && key !== 'mask';
    }).slice(0, 4).map(function (key) {
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
      var id = state.models.defaults && state.models.defaults[task];
      if (id) ids.push(id);
    });
    if (state.task !== 'all' && state.model && ids.indexOf(state.model) < 0) ids.unshift(state.model);
    var html = ids.map(function (id) {
      var model = state.models.models[id];
      if (!model) return '';
      return '<article class="model-card"><strong>' + esc(model.name) + '</strong><span>' +
        esc(taskName(model.task) + ' · ' + (model.cached ? '权重已缓存' : '首次运行会下载')) + '</span><p>' +
        esc(model.description || '') + '</p><p>准备命令：<code>' + esc(model.download_command || '') + '</code></p></article>';
    }).join('');
    html += '<article class="model-card"><strong>YOLO范式机制版</strong><span>本地机制 · 非官方权重</span><p>用后端真实计算解释单阶段网格检测，不冒充预训练 YOLO。</p></article>';
    html += '<article class="model-card"><strong>U-Net机制版</strong><span>本地机制 · 非训练权重</span><p>用后端真实计算解释编码器、解码器和跳跃连接如何形成 mask。</p></article>';
    el.modelStrip.innerHTML = html;
  }

  function updateInspectorForTask(task) {
    var map = {
      detection: {
        title: '目标检测：有什么，在哪里',
        text: '检测模型输出矩形框、类别和置信度。点击某个框，可以把结构化数据和图像位置对应起来。',
        formula: 'D={(box_i, class_i, score_i)}'
      },
      semantic: {
        title: '语义分割：每个像素属于哪一类',
        text: '语义分割把整张图变成类别地图。点击类别条目，可以看到该类在图中占多少像素。',
        formula: 'label(x,y)=argmax_c p(c|x,y,I)'
      },
      instance: {
        title: '实例分割：同类目标逐个分开',
        text: '实例分割在检测框基础上为每个目标生成独立 mask，因此同类物体不会混在一起。',
        formula: 'M_i(x,y)=1[P_i(x,y)>=tau_m]'
      },
      yolo: {
        title: 'YOLO范式：一次前向完成检测',
        text: '后端把图像划为网格，计算目标性和候选框，展示单阶段检测为什么快。',
        formula: 'box, obj, class = head(grid(I))'
      },
      unet: {
        title: 'U-Net机制：语义和边界汇合',
        text: '编码器负责语义，解码器恢复分辨率，跳跃连接把浅层边界细节送回高分辨率输出。',
        formula: 'D_l=concat(up(D_{l+1}), E_l)'
      }
    };
    var item = map[task] || map.detection;
    el.inspectTitle.textContent = item.title;
    el.inspectText.textContent = item.text;
    setFormula(item.formula);
    renderData({
      task: TASK_TEXT[task] || task,
      model: modelForTask(task),
      threshold: value01(el.score),
      mask_threshold: value01(el.mask)
    });
  }

  function updateInspectorForObject(obj) {
    if (!obj) return;
    el.inspectTitle.textContent = obj.title;
    el.inspectText.textContent = obj.data && obj.data.focus_note
      ? obj.text + ' ' + obj.data.focus_note
      : obj.text;
    setFormula(obj.task === 'semantic'
      ? 'area_c = sum_{x,y} 1[label(x,y)=c]'
      : obj.task === 'instance'
        ? 'instance_i=(box_i, class_i, score_i, mask_i)'
        : 'det_i=(box_i, class_i, score_i)');
    renderData(obj.data || {});
  }

  function updateInspectorForStep(step) {
    el.inspectTitle.textContent = step.name || step.id || '过程步骤';
    el.inspectText.textContent = step.explanation || '该步骤来自后端真实计算。';
    setFormula(step.formula || 'Y=f(X)');
    renderData(step.data || {});
  }

  function renderData(data) {
    var keys = Object.keys(data || {}).filter(function (key) {
      return key.indexOf('base64') < 0 && key !== 'mask';
    }).slice(0, 8);
    if (!keys.length) {
      el.inspectData.innerHTML = '<dt>数据</dt><dd>暂无结构化数据</dd>';
      return;
    }
    el.inspectData.innerHTML = keys.map(function (key) {
      return '<dt>' + esc(key) + '</dt><dd>' + esc(formatValue(data[key])) + '</dd>';
    }).join('');
  }

  function setFormula(value) {
    if (!el.formula) return;
    if (window.renderLatex) window.renderLatex(el.formula, value, { display: true });
    else el.formula.textContent = value;
  }

  function getOutput(task, key) {
    return (((state.result || {}).outputs || {})[task] || {})[key] || [];
  }

  function extractStepData(stepId, key) {
    var steps = (state.result || {}).steps || [];
    var step = steps.find(function (item) { return item.id === stepId; });
    return ((step || {}).data || {})[key] || [];
  }

  function findStepImage(ids) {
    var steps = (state.result || {}).steps || [];
    for (var i = 0; i < ids.length; i += 1) {
      var hit = steps.find(function (step) { return step.id === ids[i] && step.image_base64; });
      if (hit) return hit.image_base64;
    }
    return '';
  }

  function modelForTask(task) {
    if (task === 'yolo' || task === 'unet') return 'local mechanism';
    var alg = (state.result && state.result.algorithms || {})[task];
    if (alg) return alg.model_name || alg.model_id || '';
    if (state.models && state.models.defaults) return state.models.defaults[task] || '';
    return '';
  }

  function setBusy(flag) {
    el.loading.hidden = !flag;
    el.runBtn.disabled = flag;
    el.sampleBtn.disabled = flag;
    el.uploadBtn.disabled = flag;
  }

  function showError(message) {
    el.error.hidden = false;
    el.error.textContent = String(message || '未知错误');
  }

  function clearError() {
    el.error.hidden = true;
    el.error.textContent = '';
  }

  function value01(input) {
    return (Number(input.value) / 100).toFixed(2);
  }

  function taskName(task) {
    return TASK_TEXT[task] || task;
  }

  function formatNumber(value) {
    var n = Number(value || 0);
    return n >= 10 ? n.toFixed(1) : n.toFixed(3);
  }

  function formatBox(box) {
    if (!Array.isArray(box)) return '';
    return '[' + box.map(function (v) { return Math.round(Number(v)); }).join(', ') + ']';
  }

  function formatMetric(value) {
    if (typeof value === 'number') return formatNumber(value);
    if (value == null) return '';
    if (typeof value === 'object') return JSON.stringify(value).slice(0, 90);
    return String(value);
  }

  function formatValue(value) {
    if (Array.isArray(value)) {
      if (value.length > 8) return 'len=' + value.length + ' · ' + JSON.stringify(value.slice(0, 3));
      return JSON.stringify(value);
    }
    if (value && typeof value === 'object') return JSON.stringify(value).slice(0, 180);
    return value == null ? '' : String(value);
  }

  function hexToRgba(hex, alpha) {
    var clean = hex.replace('#', '');
    var r = parseInt(clean.slice(0, 2), 16);
    var g = parseInt(clean.slice(2, 4), 16);
    var b = parseInt(clean.slice(4, 6), 16);
    return 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')';
  }

  function esc(value) {
    if (window.esc) return window.esc(value);
    return String(value == null ? '' : value).replace(/[&<>"']/g, function (ch) {
      return ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[ch];
    });
  }
}());
