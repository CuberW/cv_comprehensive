(function() {
  function qs(sel, root) {
    return (root || document).querySelector(sel);
  }

  function qsa(sel, root) {
    return Array.prototype.slice.call((root || document).querySelectorAll(sel));
  }

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
  }

  function esc(value) {
    return String(value == null ? '' : value).replace(/[&<>"']/g, function(ch) {
      return ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[ch];
    });
  }

  function paramId() {
    var params = new URLSearchParams(window.location.search);
    return params.get('id') || window.ALGORITHM_ID || 'grayscale';
  }

  function stepImage(step) {
    if (!step) return '';
    if (step.image && /^data:|^\//.test(step.image)) return step.image;
    if (step.image_base64) return 'data:image/png;base64,' + step.image_base64;
    if (step.image_b64) return 'data:image/png;base64,' + step.image_b64;
    return step.image || '';
  }

  function normalizeStep(step, cfg) {
    var id = step.id || step.step || 'result';
    var meta = (cfg.stepMeta && cfg.stepMeta[id]) || [];
    return {
      id: id,
      name: meta[0] || step.name || step.title || id,
      explanation: step.explanation || step.description || meta[1] || '',
      formula: step.formula || meta[2] || '',
      image: stepImage(step)
    };
  }

  function renderStatic(cfg) {
    document.title = cfg.title + ' - Cvtoolkits';
    qs('[data-title]').textContent = cfg.title;
    qs('[data-english]').textContent = cfg.english || '';
    qs('[data-phase]').textContent = cfg.phase || '';
    qs('[data-status]').textContent = cfg.status || '';
    qs('[data-difficulty]').textContent = cfg.difficulty || '';
    qs('[data-tagline]').textContent = cfg.tagline || '';
    qs('[data-formula]').textContent = cfg.formula || '';

    var core = qs('#coreGrid');
    core.innerHTML = (cfg.core || []).map(function(item) {
      return '<div class="teach-core-item"><strong>' + esc(item[0]) + '</strong><span>' + esc(item[1]) + '</span></div>';
    }).join('');

    var copy = qs('#principles');
    copy.innerHTML = (cfg.principles || []).map(function(text) {
      return '<p>' + esc(text) + '</p>';
    }).join('');

    var pipeline = qs('#pipeline');
    pipeline.innerHTML = (cfg.pipeline || []).map(function(item, idx) {
      return '<article class="teach-stage" data-index="' + String(idx + 1).padStart(2, '0') + '">' +
        '<strong>' + esc(item[0]) + '</strong><span>' + esc(item[1]) + '</span></article>';
    }).join('');

    var controls = qs('#controls');
    controls.innerHTML = '';
    (cfg.controls || []).forEach(function(control) {
      var wrap = el('label', 'teach-control');
      wrap.appendChild(el('span', null, control.label));
      var input;
      if (control.type === 'select') {
        input = document.createElement('select');
        (control.options || []).forEach(function(opt) {
          var option = document.createElement('option');
          option.value = opt[0];
          option.textContent = opt[1];
          input.appendChild(option);
        });
        input.value = control.value;
      } else {
        input = document.createElement('input');
        input.type = control.type || 'range';
        ['min', 'max', 'step', 'value'].forEach(function(key) {
          if (control[key] != null) input.setAttribute(key, control[key]);
        });
      }
      input.name = control.name;
      wrap.appendChild(input);
      var out = document.createElement('output');
      out.textContent = input.value;
      if (control.type === 'range') {
        input.addEventListener('input', function() { out.textContent = input.value; });
      } else {
        input.addEventListener('change', function() { out.textContent = input.options[input.selectedIndex].textContent; });
        out.textContent = input.options[input.selectedIndex] ? input.options[input.selectedIndex].textContent : input.value;
      }
      wrap.appendChild(out);
      controls.appendChild(wrap);
    });
    controls.style.display = (cfg.controls || []).length ? 'grid' : 'none';
  }

  function renderMetrics(metrics) {
    var root = qs('#metrics');
    var keys = Object.keys(metrics || {});
    if (!keys.length) {
      root.innerHTML = '<span class="teach-metric">运行后显示指标</span>';
      return;
    }
    root.innerHTML = keys.map(function(key) {
      var value = metrics[key];
      if (Array.isArray(value)) value = value.join(', ');
      return '<span class="teach-metric">' + esc(key) + '<strong>' + esc(value) + '</strong></span>';
    }).join('');
  }

  function renderSteps(rawSteps, cfg) {
    var root = qs('#results');
    var steps = (rawSteps || []).map(function(step) { return normalizeStep(step, cfg); });
    if (!steps.length) {
      root.innerHTML = '<div class="teach-empty">运行算法后，这里会显示每一步的真实中间结果。</div>';
      return;
    }
    root.innerHTML = steps.map(function(step) {
      var img = step.image ? '<img src="' + esc(step.image) + '" alt="' + esc(step.name) + '">' : '<div class="teach-empty">无图像输出</div>';
      return '<article class="teach-step-card">' +
        '<div class="teach-step-image">' + img + '</div>' +
        '<div class="teach-step-body">' +
        '<strong>' + esc(step.name) + '</strong>' +
        (step.explanation ? '<p>' + esc(step.explanation) + '</p>' : '') +
        (step.formula ? '<div class="teach-step-formula">' + esc(step.formula) + '</div>' : '') +
        '</div></article>';
    }).join('');
  }

  function createDemoFile() {
    return new Promise(function(resolve) {
      var canvas = document.createElement('canvas');
      canvas.width = 192;
      canvas.height = 144;
      var ctx = canvas.getContext('2d');
      var grad = ctx.createLinearGradient(0, 0, 192, 144);
      grad.addColorStop(0, '#eef2ff');
      grad.addColorStop(0.5, '#8dd7c6');
      grad.addColorStop(1, '#f7b267');
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, 192, 144);
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(24, 24, 54, 54);
      ctx.fillStyle = '#f8fafc';
      ctx.fillRect(36, 36, 30, 30);
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 7;
      ctx.beginPath();
      ctx.moveTo(118, 20);
      ctx.lineTo(166, 70);
      ctx.lineTo(106, 118);
      ctx.stroke();
      ctx.strokeStyle = '#2563eb';
      ctx.lineWidth = 5;
      ctx.beginPath();
      ctx.arc(118, 64, 24, 0, Math.PI * 2);
      ctx.stroke();
      canvas.toBlob(function(blob) {
        resolve(new File([blob], 'teaching-demo.png', { type: 'image/png' }));
      }, 'image/png');
    });
  }

  function collectFormData(file, cfg) {
    var fd = new FormData();
    fd.set('file', file);
    qsa('#controls input, #controls select').forEach(function(input) {
      fd.set(input.name, input.value);
    });
    return fd;
  }

  function run(cfg, file) {
    var status = qs('#status');
    var button = qs('#runButton');
    if (!file) {
      status.textContent = '请先选择图片。';
      return;
    }
    status.textContent = '正在运行 NumPy 算法...';
    button.disabled = true;
    fetch(cfg.endpoint, { method: 'POST', body: collectFormData(file, cfg), headers: { Accept: 'application/json' } })
      .then(function(res) {
        if (res.ok) return res.json();
        return res.text().then(function(text) { throw new Error(text || ('HTTP ' + res.status)); });
      })
      .then(function(json) {
        renderSteps(json.steps || [], cfg);
        renderMetrics(json.metrics || {});
        status.textContent = '已生成真实中间结果。';
      })
      .catch(function(err) {
        status.textContent = '运行失败：' + (err.message || err);
      })
      .finally(function() {
        button.disabled = false;
      });
  }

  function init() {
    var id = paramId();
    var cfg = window.AlgorithmContent && window.AlgorithmContent[id];
    if (!cfg) {
      document.body.innerHTML = '<main class="teach-page"><div class="teach-empty">未找到算法内容配置：' + esc(id) + '</div></main>';
      return;
    }
    renderStatic(cfg);
    renderSteps([], cfg);
    renderMetrics({});

    var selectedFile = null;
    var upload = qs('#upload');
    var input = qs('#fileInput');
    var filename = qs('#filename');

    function setFile(file) {
      selectedFile = file;
      filename.textContent = file ? file.name : '未选择文件';
      qs('#runButton').disabled = !file;
    }

    upload.addEventListener('click', function() { input.click(); });
    upload.addEventListener('dragover', function(ev) {
      ev.preventDefault();
      upload.classList.add('dragging');
    });
    upload.addEventListener('dragleave', function() { upload.classList.remove('dragging'); });
    upload.addEventListener('drop', function(ev) {
      ev.preventDefault();
      upload.classList.remove('dragging');
      if (ev.dataTransfer.files && ev.dataTransfer.files[0]) setFile(ev.dataTransfer.files[0]);
    });
    input.addEventListener('change', function() {
      setFile(input.files && input.files[0] ? input.files[0] : null);
    });
    qs('#runButton').addEventListener('click', function() { run(cfg, selectedFile); });
    qs('#demoButton').addEventListener('click', function() {
      createDemoFile().then(function(file) {
        setFile(file);
        run(cfg, file);
      });
    });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
