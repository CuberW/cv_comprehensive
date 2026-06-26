(function () {
  'use strict';

  var DEMO_SRC = '../images/demo-street.jpg';
  var conceptQueue = Promise.resolve();
  var realStatusPromise = null;
  var conceptResults = {
    detection: { status: 'idle', data: null, image: null, boxes: [] },
    semantic: { status: 'idle', data: null, image: null, boxes: [] },
    instance: { status: 'idle', data: null, image: null, boxes: [] }
  };

  var demoImage = new Image();
  demoImage.src = DEMO_SRC;

  function setupCanvas(canvas) {
    var rect = canvas.getBoundingClientRect();
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.max(1, Math.floor(rect.width * dpr));
    canvas.height = Math.max(1, Math.floor(rect.height * dpr));
    var ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx: ctx, rect: rect };
  }

  function coverSize(containerW, containerH, imageW, imageH) {
    var scale = Math.max(containerW / imageW, containerH / imageH);
    var w = imageW * scale;
    var h = imageH * scale;
    return { x: (containerW - w) / 2, y: (containerH - h) / 2, w: w, h: h, scale: scale };
  }

  function containSize(containerW, containerH, imageW, imageH) {
    var scale = Math.min(containerW / imageW, containerH / imageH, 1.25);
    var w = imageW * scale;
    var h = imageH * scale;
    return { x: (containerW - w) / 2, y: (containerH - h) / 2, w: w, h: h, scale: scale };
  }

  function resultStage(rect, imageW, imageH) {
    var isMobile = rect.width < 900;
    var reservedLeft = isMobile ? 18 : Math.min(560, Math.max(430, rect.width * 0.36));
    var reservedBottom = isMobile ? 250 : 56;
    var stage = {
      x: isMobile ? 16 : reservedLeft,
      y: isMobile ? 28 : Math.max(28, rect.height * 0.09),
      w: isMobile ? rect.width - 32 : rect.width - reservedLeft - 56,
      h: isMobile ? rect.height - reservedBottom - 48 : rect.height * 0.82
    };
    stage.w = Math.max(260, stage.w);
    stage.h = Math.max(220, stage.h);
    var fit = containSize(stage.w, stage.h, imageW, imageH);
    fit.x += stage.x;
    fit.y += stage.y;
    return fit;
  }

  function drawStatus(canvas, message) {
    var c = setupCanvas(canvas);
    c.ctx.clearRect(0, 0, c.rect.width, c.rect.height);
    drawStatusOn(c.ctx, c.rect, message);
  }

  function drawStatusOn(ctx, rect, message) {
    ctx.save();
    ctx.fillStyle = 'rgba(2, 6, 23, 0.72)';
    roundRect(ctx, 18, rect.height - 68, Math.min(460, rect.width - 36), 46, 8);
    ctx.fill();
    ctx.strokeStyle = 'rgba(226, 232, 240, 0.18)';
    ctx.stroke();
    ctx.fillStyle = '#dbeafe';
    ctx.font = '800 14px "Noto Sans SC", "Microsoft YaHei", sans-serif';
    ctx.fillText(message, 34, rect.height - 39);
    ctx.restore();
  }

  function drawImageCover(canvas, image, alpha) {
    var c = setupCanvas(canvas);
    var ctx = c.ctx;
    var rect = c.rect;
    ctx.clearRect(0, 0, rect.width, rect.height);
    if (!image || !image.naturalWidth) return null;
    var box = coverSize(rect.width, rect.height, image.naturalWidth, image.naturalHeight);
    ctx.save();
    ctx.globalAlpha = alpha == null ? 1 : alpha;
    ctx.drawImage(image, box.x, box.y, box.w, box.h);
    ctx.restore();
    return box;
  }

  function roundRect(ctx, x, y, w, h, r) {
    var rr = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + rr, y);
    ctx.lineTo(x + w - rr, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + rr);
    ctx.lineTo(x + w, y + h - rr);
    ctx.quadraticCurveTo(x + w, y + h, x + w - rr, y + h);
    ctx.lineTo(x + rr, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - rr);
    ctx.lineTo(x, y + rr);
    ctx.quadraticCurveTo(x, y, x + rr, y);
    ctx.closePath();
  }

  function drawLabel(ctx, text, x, y, color) {
    ctx.save();
    ctx.font = '800 13px "Noto Sans SC", "Microsoft YaHei", sans-serif';
    var width = Math.min(ctx.measureText(text).width + 20, 260);
    var top = Math.max(8, y - 30);
    roundRect(ctx, x, top, width, 24, 7);
    ctx.fillStyle = 'rgba(2, 6, 23, 0.9)';
    ctx.fill();
    ctx.strokeStyle = color;
    ctx.stroke();
    ctx.fillStyle = '#ffffff';
    ctx.fillText(text, x + 10, top + 16, width - 18);
    ctx.restore();
  }

  function placeTooltip(el, text, x, y) {
    if (!el) return;
    el.textContent = text;
    el.style.left = x + 'px';
    el.style.top = y + 'px';
    el.classList.add('show');
  }

  function hideTooltip(el) {
    if (el) el.classList.remove('show');
  }

  function esc(value) {
    return String(value).replace(/[&<>"']/g, function (ch) {
      return ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[ch];
    });
  }

  function fetchDemoBlob() {
    return fetch(DEMO_SRC)
      .then(function (res) { return res.blob(); })
      .then(function (blob) { return { blob: blob, name: 'demo-street.jpg' }; });
  }

  function ensureRealReady() {
    if (!realStatusPromise) {
      realStatusPromise = fetch('/api/demo/vision-real-status')
        .then(function (res) {
          return res.json().catch(function () { return {}; }).then(function (json) {
            if (!res.ok) throw new Error(json.error || ('HTTP ' + res.status));
            if (!json.ready) throw new Error('真实视觉模型不可用，请确认 PyTorch、torchvision 和模型权重已正确安装。');
            return json;
          });
        });
    }
    return realStatusPromise;
  }

  function runDemo(mode) {
    if (conceptResults[mode].status === 'loading' || conceptResults[mode].status === 'ready') {
      return conceptResults[mode].promise || Promise.resolve(conceptResults[mode]);
    }
    conceptResults[mode].status = 'loading';
    conceptResults[mode].promise = enqueueConceptRequest(function () {
      return ensureRealReady()
        .then(fetchDemoBlob)
        .then(function (payload) {
          return postAlgorithm(mode, payload, {
            score_threshold: '0.30',
            threshold: '0.30',
            num_classes: '6',
            num_instances: '6',
            require_real: '1'
          });
        });
    })
      .then(function (res) {
        return res.json().catch(function () { return {}; }).then(function (json) {
          if (!res.ok) throw new Error(json.error || ('HTTP ' + res.status));
          return json;
        });
      })
      .then(function (data) {
        return hydrateConceptResult(mode, data);
      })
      .catch(function (err) {
        conceptResults[mode].status = 'error';
        conceptResults[mode].error = err.message || String(err);
        return conceptResults[mode];
      });
    return conceptResults[mode].promise;
  }

  function enqueueConceptRequest(factory) {
    var task = conceptQueue.then(factory, factory);
    conceptQueue = task.catch(function () {});
    return task;
  }

  function postAlgorithm(mode, payload, fields) {
    var form = new FormData();
    form.append('file', payload.blob, payload.name);
    Object.keys(fields || {}).forEach(function (key) {
      form.append(key, fields[key]);
    });
    return fetch('/api/demo/' + mode, { method: 'POST', body: form });
  }

  function hydrateConceptResult(mode, data) {
    var preferred = {
      detection: ['detections', 'result', 'output'],
      semantic: ['overlay', 'segmentation', 'result'],
      instance: ['masks', 'result', 'output']
    }[mode];
    var step = pickStep(data.steps || [], preferred);
    var image = new Image();
    var dataUrl = step && step.image_base64 ? 'data:image/png;base64,' + step.image_base64 : '';
    conceptResults[mode].data = data;
    conceptResults[mode].dataUrl = dataUrl;
    conceptResults[mode].boxes = extractBoxes(mode, data);
    return new Promise(function (resolve) {
      image.onload = function () {
        conceptResults[mode].status = 'ready';
        conceptResults[mode].image = image;
        resolve(conceptResults[mode]);
      };
      image.onerror = function () {
        conceptResults[mode].status = 'error';
        conceptResults[mode].error = 'Result image failed to load';
        resolve(conceptResults[mode]);
      };
      if (dataUrl) {
        image.src = dataUrl;
      } else {
        conceptResults[mode].status = 'error';
        conceptResults[mode].error = 'No result image returned by API';
        resolve(conceptResults[mode]);
      }
    });
  }

  function pickStep(steps, ids) {
    for (var i = 0; i < ids.length; i += 1) {
      var found = steps.find(function (step) { return step.id === ids[i]; });
      if (found && found.image_base64) return found;
    }
    for (var j = steps.length - 1; j >= 0; j -= 1) {
      if (steps[j].image_base64) return steps[j];
    }
    return null;
  }

  function extractBoxes(mode, data) {
    var items = mode === 'instance' ? (data.instances || []) : (data.detections || []);
    return items
      .filter(function (item) { return item.box && item.box.length === 4; })
      .map(function (item, index) {
        return {
          box: item.box.map(Number),
          label: item.label || (mode === 'instance' ? 'instance' : 'object'),
          score: typeof item.score === 'number' ? item.score : null,
          color: ['#22d3ee', '#f472b6', '#facc15', '#34d399', '#fb7185', '#a78bfa'][index % 6],
          id: mode === 'instance' ? 'ID-' + String(index + 1).padStart(2, '0') : ''
        };
      });
  }

  function renderRealResult(mode, canvas, hoverIndex) {
    var result = conceptResults[mode];
    if (result.status === 'loading' || result.status === 'idle') {
      drawStatus(canvas, '正在运行真实 ' + mode + ' 结果...');
      return [];
    }
      if (result.status === 'error') {
      drawStatus(canvas, '真实模型结果加载失败：' + (result.error || 'unknown error'));
      return [];
    }
    var c = setupCanvas(canvas);
    var ctx = c.ctx;
    var rect = c.rect;
    ctx.clearRect(0, 0, rect.width, rect.height);
    if (!result.image || !result.image.naturalWidth) return [];
    var imageBox = resultStage(rect, result.image.naturalWidth, result.image.naturalHeight);
    drawStageFrame(ctx, imageBox, mode);
    ctx.save();
    ctx.globalAlpha = mode === 'instance' ? 0.92 : 1;
    ctx.drawImage(result.image, imageBox.x, imageBox.y, imageBox.w, imageBox.h);
    ctx.restore();
    if (!imageBox) return [];
    if (mode === 'semantic') {
      drawStatusOn(ctx, rect, '语义分割画面来自后端真实输出，不使用前端假 mask。');
      return [];
    }
    var sx = imageBox.w / result.image.naturalWidth;
    var sy = imageBox.h / result.image.naturalHeight;
    return result.boxes.map(function (item, index) {
      var x1 = imageBox.x + item.box[0] * sx;
      var y1 = imageBox.y + item.box[1] * sy;
      var x2 = imageBox.x + item.box[2] * sx;
      var y2 = imageBox.y + item.box[3] * sy;
      var active = hoverIndex === index;
      ctx.save();
      ctx.strokeStyle = item.color;
      ctx.lineWidth = active ? 4 : 2;
      ctx.shadowColor = item.color;
      ctx.shadowBlur = active ? 24 : 10;
      roundRect(ctx, x1, y1, x2 - x1, y2 - y1, 8);
      ctx.stroke();
      if (active) {
        var text = item.label + (item.id ? ' ' + item.id : '') +
          (item.score == null ? '' : ' ' + Math.round(item.score * 100) + '%');
        drawLabel(ctx, text, x1, y1, item.color);
      }
      ctx.restore();
      return { x: x1, y: y1, w: x2 - x1, h: y2 - y1, data: item, rect: rect };
    });
  }

  function drawStageFrame(ctx, box, mode) {
    ctx.save();
    var pad = 10;
    ctx.fillStyle = 'rgba(2, 6, 23, 0.7)';
    ctx.strokeStyle = mode === 'semantic'
      ? 'rgba(167, 139, 250, 0.35)'
      : mode === 'instance'
        ? 'rgba(244, 114, 182, 0.35)'
        : 'rgba(34, 211, 238, 0.35)';
    ctx.lineWidth = 1.5;
    ctx.shadowColor = 'rgba(0, 0, 0, 0.45)';
    ctx.shadowBlur = 28;
    roundRect(ctx, box.x - pad, box.y - pad, box.w + pad * 2, box.h + pad * 2, 8);
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }

  function initHero() {
    var canvas = document.getElementById('flashCanvas');
    var scene = document.querySelector('[data-scene="hero"]');
    var hint = document.getElementById('scrollHint');
    if (!canvas || !scene) return;
    var pointer = { x: 0.5, y: 0.5 };

    function draw() {
      var c = setupCanvas(canvas);
      var ctx = c.ctx;
      var rect = c.rect;
      ctx.clearRect(0, 0, rect.width, rect.height);
      var gx = pointer.x * rect.width;
      var gy = pointer.y * rect.height;
      var grad = ctx.createRadialGradient(gx, gy, 20, gx, gy, 190);
      grad.addColorStop(0, 'rgba(34, 211, 238, 0.18)');
      grad.addColorStop(0.45, 'rgba(34, 211, 238, 0.06)');
      grad.addColorStop(1, 'rgba(34, 211, 238, 0)');
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, rect.width, rect.height);

      var result = conceptResults.detection;
      if (result.status !== 'ready') return;
      var imageBox = coverSize(rect.width, rect.height, demoImage.naturalWidth || 1, demoImage.naturalHeight || 1);
      var sx = imageBox.w / (demoImage.naturalWidth || 1);
      var sy = imageBox.h / (demoImage.naturalHeight || 1);
      result.boxes.forEach(function (item) {
        var x1 = imageBox.x + item.box[0] * sx;
        var y1 = imageBox.y + item.box[1] * sy;
        var x2 = imageBox.x + item.box[2] * sx;
        var y2 = imageBox.y + item.box[3] * sy;
        var cx = (x1 + x2) / 2;
        var cy = (y1 + y2) / 2;
        var d = Math.hypot(cx - gx, cy - gy);
        if (d < 190) {
          ctx.save();
          ctx.globalAlpha = 1 - d / 230;
          ctx.strokeStyle = item.color;
          ctx.lineWidth = 2;
          ctx.shadowColor = item.color;
          ctx.shadowBlur = 18;
          roundRect(ctx, x1, y1, x2 - x1, y2 - y1, 8);
          ctx.stroke();
          ctx.restore();
        }
      });
    }

    runDemo('detection').then(draw);
    scene.addEventListener('mousemove', function (event) {
      var rect = scene.getBoundingClientRect();
      pointer.x = (event.clientX - rect.left) / rect.width;
      pointer.y = (event.clientY - rect.top) / rect.height;
      draw();
    });
    hint.addEventListener('click', function () {
      var target = document.querySelector('[data-scene="detection"]');
      if (target) target.scrollIntoView({ behavior: 'smooth' });
    });
    window.addEventListener('resize', draw);
    draw();
  }

  function initDetection() {
    var canvas = document.getElementById('detCanvas');
    var tooltip = document.getElementById('detTooltip');
    var annotation = document.getElementById('detAnnotation');
    if (!canvas) return;
    var rendered = [];

    function redraw(hoverIndex) {
      rendered = renderRealResult('detection', canvas, hoverIndex);
    }

    runDemo('detection').then(function () { redraw(-1); });
    attachBoxHover(canvas, tooltip, function () { return rendered; }, redraw, function (hit) {
      if (!hit) {
        if (annotation) annotation.classList.remove('visible');
        return '';
      }
      if (annotation) annotation.classList.add('visible');
      return hit.data.label + (hit.data.score == null ? '' : ' ' + Math.round(hit.data.score * 100) + '%');
    });
    observeScene(canvas.parentElement, function () {
      runDemo('detection').then(function () { redraw(-1); });
    });
    window.addEventListener('resize', function () { redraw(-1); });
    redraw(-1);
  }

  function initSemantic() {
    var canvas = document.getElementById('semCanvas');
    var tooltip = document.getElementById('semTooltip');
    var annotation = document.getElementById('semAnnotation');
    if (!canvas) return;

    function redraw() {
      renderRealResult('semantic', canvas, -1);
    }

    runDemo('semantic').then(redraw);
    canvas.addEventListener('mousemove', function (event) {
      var rect = canvas.getBoundingClientRect();
      placeTooltip(tooltip, '真实语义分割输出：颜色来自后端模型/算法', event.clientX - rect.left + 72, event.clientY - rect.top - 8);
      if (annotation) annotation.classList.add('visible');
    });
    canvas.addEventListener('mouseleave', function () {
      hideTooltip(tooltip);
      if (annotation) annotation.classList.remove('visible');
    });
    observeScene(canvas.parentElement, function () {
      runDemo('semantic').then(redraw);
      if (annotation) annotation.classList.add('visible');
    }, function () {
      if (annotation) annotation.classList.remove('visible');
    });
    window.addEventListener('resize', redraw);
    redraw();
  }

  function initInstance() {
    var canvas = document.getElementById('insCanvas');
    var tooltip = document.getElementById('insTooltip');
    if (!canvas) return;
    var rendered = [];

    function redraw(hoverIndex) {
      rendered = renderRealResult('instance', canvas, hoverIndex);
    }

    runDemo('instance').then(function () { redraw(-1); });
    attachBoxHover(canvas, tooltip, function () { return rendered; }, redraw, function (hit) {
      if (!hit) return '';
      return hit.data.label + ' ' + hit.data.id +
        (hit.data.score == null ? '' : ' ' + Math.round(hit.data.score * 100) + '%');
    });
    observeScene(canvas.parentElement, function () {
      runDemo('instance').then(function () { redraw(-1); });
    });
    window.addEventListener('resize', function () { redraw(-1); });
    redraw(-1);
  }

  function attachBoxHover(canvas, tooltip, getRendered, redraw, textForHit) {
    canvas.addEventListener('mousemove', function (event) {
      var rect = canvas.getBoundingClientRect();
      var mx = event.clientX - rect.left;
      var my = event.clientY - rect.top;
      var items = getRendered();
      var found = -1;
      for (var i = items.length - 1; i >= 0; i -= 1) {
        var item = items[i];
        if (mx >= item.x && mx <= item.x + item.w && my >= item.y && my <= item.y + item.h) {
          found = i;
          break;
        }
      }
      redraw(found);
      if (found >= 0) {
        var hit = getRendered()[found] || items[found];
        placeTooltip(tooltip, textForHit(hit), hit.x + hit.w / 2, hit.y - 10);
      } else {
        hideTooltip(tooltip);
        textForHit(null);
      }
    });
    canvas.addEventListener('mouseleave', function () {
      redraw(-1);
      hideTooltip(tooltip);
      textForHit(null);
    });
  }

  function observeScene(scene, onEnter, onLeave) {
    if (!scene || !('IntersectionObserver' in window)) {
      if (onEnter) onEnter();
      return;
    }
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          if (onEnter) onEnter();
        } else if (onLeave) {
          onLeave();
        }
      });
    }, { threshold: 0.35 });
    observer.observe(scene);
  }

  function initPlayground() {
    var mode = 'original';
    var uploadedFile = null;
    var currentObjectUrl = '';
    var latestResultImage = null;

    var fileInput = document.getElementById('fileInput');
    var uploadBtn = document.getElementById('uploadBtn');
    var presetThumb = document.getElementById('presetThumb');
    var fileName = document.getElementById('fileName');
    var modeBtns = Array.prototype.slice.call(document.querySelectorAll('.mode-btn'));
    var confidence = document.getElementById('confidence');
    var confidenceValue = document.getElementById('confidenceValue');
    var opacity = document.getElementById('opacity');
    var opacityValue = document.getElementById('opacityValue');
    var runBtn = document.getElementById('runBtn');
    var canvas = document.getElementById('playCanvas');
    var viewport = document.getElementById('viewport');
    var status = document.getElementById('viewportStatus');
    var stepGrid = document.getElementById('stepGrid');
    var metrics = document.getElementById('metrics');
    var errorBox = document.getElementById('errorBox');

    function sourceUrl() {
      return currentObjectUrl || DEMO_SRC;
    }

    function renderImage(url, alpha) {
      var img = new Image();
      img.onload = function () {
        var rect = viewport.getBoundingClientRect();
        var maxW = Math.max(280, rect.width - 24);
        var maxH = Math.max(220, rect.height - 24);
        var scale = Math.min(maxW / img.width, maxH / img.height, 1);
        var dpr = Math.min(window.devicePixelRatio || 1, 2);
        canvas.width = Math.max(1, Math.floor(img.width * scale * dpr));
        canvas.height = Math.max(1, Math.floor(img.height * scale * dpr));
        canvas.style.width = Math.floor(img.width * scale) + 'px';
        canvas.style.height = Math.floor(img.height * scale) + 'px';
        var ctx = canvas.getContext('2d');
        ctx.setTransform(dpr * scale, 0, 0, dpr * scale, 0, 0);
        ctx.clearRect(0, 0, img.width, img.height);
        ctx.globalAlpha = alpha == null ? 1 : alpha;
        ctx.drawImage(img, 0, 0);
        ctx.globalAlpha = 1;
      };
      img.src = url;
    }

    function updateControls() {
      confidenceValue.textContent = mode === 'semantic' || mode === 'original'
        ? '-'
        : (confidence.value / 100).toFixed(2);
      confidence.disabled = mode === 'semantic' || mode === 'original';
      opacityValue.textContent = (opacity.value / 100).toFixed(2);
      runBtn.textContent = mode === 'original' ? '显示原图' : '运行当前模式';
    }

    function setMode(next) {
      mode = next;
      modeBtns.forEach(function (btn) {
        btn.classList.toggle('active', btn.getAttribute('data-mode') === mode);
      });
      updateControls();
      clearError();
      if (mode === 'original') {
        latestResultImage = null;
        stepGrid.innerHTML = '';
        metrics.innerHTML = '';
        status.textContent = '原图模式：无需运行算法。';
        renderImage(sourceUrl());
      }
    }

    function clearError() {
      errorBox.style.display = 'none';
      errorBox.textContent = '';
    }

    function showError(message) {
      errorBox.textContent = message;
      errorBox.style.display = 'block';
    }

    function setBusy(busy) {
      runBtn.disabled = busy;
      if (busy) status.textContent = '算法运行中，请稍等...';
    }

    function getImageBlob() {
      if (uploadedFile) return Promise.resolve({ blob: uploadedFile, name: uploadedFile.name || 'upload.png' });
      return fetchDemoBlob();
    }

    function runCurrent() {
      clearError();
      if (mode === 'original') {
        latestResultImage = null;
        stepGrid.innerHTML = '';
        metrics.innerHTML = '';
        status.textContent = '原图模式：无需运行算法。';
        renderImage(sourceUrl());
        return;
      }

      setBusy(true);
      stepGrid.innerHTML = '';
      metrics.innerHTML = '';
      ensureRealReady()
        .then(getImageBlob)
        .then(function (payload) {
          return postAlgorithm(mode, payload, {
            score_threshold: (confidence.value / 100).toFixed(2),
            threshold: (confidence.value / 100).toFixed(2),
            num_classes: '6',
            num_instances: '6',
            require_real: '1'
          });
        })
        .then(function (res) {
          return res.json().catch(function () { return {}; }).then(function (json) {
            if (!res.ok) throw new Error(json.error || ('HTTP ' + res.status));
            return json;
          });
        })
        .then(function (data) {
          setBusy(false);
          renderPlaygroundResult(data);
        })
        .catch(function (err) {
          setBusy(false);
          status.textContent = '运行失败。';
          showError('运行失败：' + (err.message || '未知错误'));
        });
    }

    function renderPlaygroundResult(data) {
      var steps = data.steps || [];
      var last = pickStep(steps, {
        detection: ['detections', 'result', 'output'],
        semantic: ['overlay', 'segmentation', 'result'],
        instance: ['masks', 'result', 'output']
      }[mode] || ['result']);
      if (last && last.image_base64) {
        latestResultImage = 'data:image/png;base64,' + last.image_base64;
        status.textContent = modeName(mode) + '：显示后端真实输出，透明度可继续调节。';
        renderImage(latestResultImage, opacity.value / 100);
      } else {
        latestResultImage = null;
        status.textContent = '接口返回成功，但没有可显示的图像步骤。';
      }

      var metricObj = data.metrics || {};
      metrics.innerHTML = Object.keys(metricObj).slice(0, 8).map(function (key) {
        return '<span class="metric">' + esc(key) + ': ' + esc(formatMetric(metricObj[key])) + '</span>';
      }).join('');

      stepGrid.innerHTML = steps.map(function (step, index) {
        var image = step.image_base64
          ? '<img src="data:image/png;base64,' + step.image_base64 + '" alt="' + esc(step.name || step.id || '步骤') + '">'
          : '';
        return '<article class="step-card">' + image + '<div><strong>' +
          esc((index + 1) + '. ' + (step.name || step.id || 'Step')) +
          '</strong><span>' + esc(step.explanation || '') + '</span>' + (step.formula ? '<code class="step-formula">' + esc(step.formula) + '</code>' : '') + '</div></article>';
      }).join('');
    }

    function modeName(value) {
      return {
        original: '原图',
        detection: '目标检测',
        semantic: '语义分割',
        instance: '实例分割'
      }[value] || value;
    }

    function formatMetric(value) {
      if (typeof value === 'number') return Math.abs(value) > 10 ? value.toFixed(1) : value.toFixed(3);
      if (value === null || value === undefined) return '';
      return String(value);
    }

    uploadBtn.addEventListener('click', function () { fileInput.click(); });
    fileInput.addEventListener('change', function () {
      uploadedFile = fileInput.files && fileInput.files[0] ? fileInput.files[0] : null;
      if (currentObjectUrl) URL.revokeObjectURL(currentObjectUrl);
      currentObjectUrl = uploadedFile ? URL.createObjectURL(uploadedFile) : '';
      fileName.textContent = uploadedFile ? uploadedFile.name : '未选择文件';
      presetThumb.classList.toggle('active', !uploadedFile);
      latestResultImage = null;
      stepGrid.innerHTML = '';
      metrics.innerHTML = '';
      renderImage(sourceUrl());
    });
    presetThumb.addEventListener('click', function () {
      uploadedFile = null;
      fileInput.value = '';
      if (currentObjectUrl) URL.revokeObjectURL(currentObjectUrl);
      currentObjectUrl = '';
      fileName.textContent = '未选择文件';
      presetThumb.classList.add('active');
      latestResultImage = null;
      stepGrid.innerHTML = '';
      metrics.innerHTML = '';
      renderImage(sourceUrl());
    });
    modeBtns.forEach(function (btn) {
      btn.addEventListener('click', function () {
        setMode(btn.getAttribute('data-mode'));
      });
    });
    confidence.addEventListener('input', updateControls);
    opacity.addEventListener('input', function () {
      updateControls();
      if (latestResultImage) renderImage(latestResultImage, opacity.value / 100);
    });
    runBtn.addEventListener('click', runCurrent);
    window.addEventListener('resize', function () {
      renderImage(latestResultImage || sourceUrl(), latestResultImage ? opacity.value / 100 : undefined);
    });

    updateControls();
    renderImage(sourceUrl());
  }

  function jumpFromUrl() {
    var params = new URLSearchParams(window.location.search);
    var key = params.get('module') || params.get('tab') || window.location.hash.replace('#', '');
    var target = key && document.querySelector('[data-scene="' + key + '"]');
    if (target) {
      setTimeout(function () {
        target.scrollIntoView({ behavior: 'auto' });
      }, 120);
    }
  }

  function init() {
    initHero();
    initDetection();
    initSemantic();
    initInstance();
    initPlayground();
    jumpFromUrl();
  }

  if (demoImage.complete) {
    init();
  } else {
    demoImage.addEventListener('load', init, { once: true });
    demoImage.addEventListener('error', init, { once: true });
  }
})();
