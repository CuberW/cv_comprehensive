(function () {
  'use strict';

  var cfg = window.ModelDemoConfig || {};
  var moduleId = cfg.moduleId || new URLSearchParams(location.search).get('id') || 'resnet';
  var state = { file: null, result: null };

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
    if ($('pageTitle')) $('pageTitle').textContent = cfg.title || moduleId;
    if ($('subtitle')) $('subtitle').textContent = cfg.subtitle || '';
    if (els.intro) els.intro.textContent = cfg.intro || '';
    if (els.status) els.status.textContent = cfg.status || '后端真实计算';
    renderStaticSections();
    renderParams();
    bind();
    runDemo();
  }

  function bind() {
    if (els.choose && els.file) {
      els.choose.addEventListener('click', function () { els.file.click(); });
      els.file.addEventListener('change', function () {
        state.file = els.file.files && els.file.files[0] ? els.file.files[0] : null;
        els.fileName.textContent = state.file ? state.file.name : '未选择文件，将使用内置样例';
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
      return '<label class="param"><span>' + esc(p.label) + '</span><input id="param_' + esc(p.name) +
        '" type="text" value="' + esc(value) + '"></label>';
    }).join('');
    cfg.params.forEach(function (p) {
      var input = $('param_' + p.name);
      var value = $('param_' + p.name + '_value');
      if (input && value) {
        input.addEventListener('input', function () { value.textContent = formatParam(p, input.value); });
      }
    });
  }

  function renderStaticSections() {
    var pipeline = $('pipelineList');
    if (pipeline && cfg.pipeline) {
      pipeline.innerHTML = cfg.pipeline.map(function (item) { return '<li>' + esc(item) + '</li>'; }).join('');
    }
    var applications = $('applicationsList');
    if (applications && cfg.applications) {
      applications.innerHTML = cfg.applications.map(function (item) { return '<li>' + esc(item) + '</li>'; }).join('');
    }
    var limitations = $('limitationsList');
    if (limitations && cfg.limitations) {
      limitations.innerHTML = cfg.limitations.map(function (item) { return '<li>' + esc(item) + '</li>'; }).join('');
    }
  }

  function runDemo() {
    clearError();
    setBusy(true);
    var form = new FormData();
    if (state.file) form.append('file', state.file);
    if (cfg.params) {
      cfg.params.forEach(function (p) {
        var input = $('param_' + p.name);
        if (input) form.append(p.name, normalizeParam(p, input.value));
      });
    }
    fetch('/api/demo/' + encodeURIComponent(moduleId), { method: 'POST', body: form })
      .then(function (res) {
        return res.json().catch(function () { return {}; }).then(function (json) {
          if (!res.ok) {
            json._httpError = res.status;
            return json;
          }
          return json;
        });
      })
      .then(function (json) {
        state.result = json;
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
    renderSteps(json.steps || []);
  }

  function renderMetrics(metrics, impl) {
    if (!els.metrics) return;
    var rows = [];
    if (impl.status) rows.push(['实现状态', impl.status]);
    if (impl.model) rows.push(['模型/算法', impl.model]);
    if (impl.backend) rows.push(['后端', impl.backend]);
    ['status', 'backend', 'model', 'top1', 'top1_prob', 'num_detections', 'best_iou_score',
      'reconstruction_mse', 'samples_per_ray', 'D_loss_final', 'G_loss_final'].forEach(function (key) {
      if (metrics[key] !== undefined) rows.push([labelMetric(key), metrics[key]]);
    });
    els.metrics.innerHTML = rows.map(function (row) {
      return '<div class="metric"><span>' + esc(row[0]) + '</span><strong>' + esc(formatMetric(row[1])) + '</strong></div>';
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
      var data = renderData(step.data);
      return '<article class="step-card">' +
        '<div class="step-index">' + String(index + 1).padStart(2, '0') + '</div>' +
        '<div class="step-media">' + img + '</div>' +
        '<div class="step-body"><h3>' + esc(step.name || step.id || '步骤') + '</h3>' +
        '<p>' + esc(step.explanation || '') + '</p>' +
        (step.formula ? '<code>' + esc(step.formula) + '</code>' : '') +
        data + '</div></article>';
    }).join('');
  }

  function renderData(data) {
    if (!data || typeof data !== 'object') return '';
    var keys = Object.keys(data).slice(0, 6);
    if (!keys.length) return '';
    return '<details><summary>中间数据</summary><dl>' + keys.map(function (key) {
      return '<dt>' + esc(key) + '</dt><dd>' + esc(formatMetric(data[key])) + '</dd>';
    }).join('') + '</dl></details>';
  }

  function setBusy(flag) {
    if (els.loading) els.loading.hidden = !flag;
    if (els.run) els.run.disabled = flag;
    if (els.sample) els.sample.disabled = flag;
  }

  function showError(msg) {
    if (!els.error) return;
    els.error.hidden = false;
    els.error.textContent = msg;
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
    return value;
  }

  function labelMetric(key) {
    var map = {
      status: '状态',
      backend: '后端',
      model: '模型',
      top1: 'Top-1',
      top1_prob: 'Top-1 概率',
      num_detections: '检测数',
      best_iou_score: '最佳 IoU',
      reconstruction_mse: '重建 MSE',
      samples_per_ray: '每射线采样',
      D_loss_final: '最终 D loss',
      G_loss_final: '最终 G loss'
    };
    return map[key] || key;
  }

  function formatMetric(value) {
    if (Array.isArray(value)) return 'len=' + value.length;
    if (value && typeof value === 'object') return JSON.stringify(value).slice(0, 160);
    if (typeof value === 'number') return Math.round(value * 10000) / 10000;
    return value == null ? '' : String(value);
  }

  function esc(value) {
    if (window.PageUtils && window.PageUtils.esc) return window.PageUtils.esc(value);
    return String(value == null ? '' : value).replace(/[&<>"']/g, function (ch) {
      return ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[ch];
    });
  }
}());
