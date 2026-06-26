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

  var visualAnimRaf = null;
  var visualAnimStates = [];
  var visualAnimLastTime = 0;

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

  function normalizeFormula(formula) {
    if (!formula) return '';
    return String(formula)
      .replace(/\$\$/g, '')
      .replace(/^\s*\\\[\s*|\s*\\\]\s*$/g, '')
      .replace(/^\s*\\\(\s*|\s*\\\)\s*$/g, '')
      .trim();
  }

  function renderFormula(elm, formula) {
    var latex = normalizeFormula(formula);
    if (!latex) {
      elm.innerHTML = '';
      return;
    }
    if (window.katex && window.katex.render) {
      try {
        elm.innerHTML = '';
        window.katex.render(latex, elm, { throwOnError: false, displayMode: true });
        return;
      } catch (err) {}
    }
    elm.textContent = latex;
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
    renderFormula(qs('[data-formula]'), cfg.formula || '');

    var core = qs('#coreGrid');
    core.innerHTML = (cfg.core || []).map(function(item) {
      return '<div class="teach-core-item"><strong>' + esc(item[0]) + '</strong><span>' + esc(item[1]) + '</span></div>';
    }).join('');

    var copy = qs('#principles');
    copy.innerHTML = (cfg.principles || []).map(function(text) {
      return '<p>' + esc(text) + '</p>';
    }).join('');
    renderVisualStory(cfg);

    var referenceSection = qs('#referenceSection');
    var references = qs('#references');
    if (cfg.references && cfg.references.length) {
      referenceSection.style.display = '';
      references.innerHTML = cfg.references.map(function(ref) {
        var title = ref.title || ref.name || '参考资料';
        var desc = ref.description || '';
        var url = ref.url || '';
        return '<article class="teach-ref"><strong>' + esc(title) + '</strong>' +
          (desc ? '<span>' + esc(desc) + '</span>' : '') +
          (url ? '<a href="' + esc(url) + '" target="_blank" rel="noopener noreferrer">' + esc(url) + '</a>' : '') +
          '</article>';
      }).join('');
    } else {
      referenceSection.style.display = 'none';
      references.innerHTML = '';
    }

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

    // Hide rect sliders if we'll use canvas instead
    if (hasRectControls(cfg)) {
      qsa('#controls [name=\"x\"], #controls [name=\"y\"], #controls [name=\"w\"], #controls [name=\"h\"]')
        .forEach(function(el) {
          var wrap = el.closest('.teach-control');
          if (wrap) wrap.style.display = 'none';
        });
    }
  }

  // ── Interactive rectangle canvas ──
  var liveImg = null;
  var rectState = { dragging: false, handle: null, startX: 0, startY: 0, startRect: null };

  function hasRectControls(cfg) {
    var names = (cfg.controls || []).map(function(c) { return c.name; });
    return names.indexOf('x') >= 0 && names.indexOf('y') >= 0 &&
           names.indexOf('w') >= 0 && names.indexOf('h') >= 0;
  }

  function getControlValue(name) {
    var el = qs('#controls [name="' + name + '"]');
    return el ? Number(el.value) : 0;
  }

  function setControlValue(name, val) {
    var el = qs('#controls [name="' + name + '"]');
    if (el) { el.value = Math.round(val); el.dispatchEvent(new Event('input', {bubbles:true})); }
  }

  function initRectCanvas(cfg) {
    var wrap = qs('#livePreviewWrap');
    var canvas = qs('#livePreview');
    var hint = qs('#previewHint');
    if (!wrap || !canvas || !hasRectControls(cfg)) return;

    // Make canvas bigger for better quality
    canvas.width = 560;
    canvas.height = 420;
    wrap.style.display = '';
    var ctx = canvas.getContext('2d');

    function imgToCanvas() {
      if (!liveImg) return { dx: 0, dy: 0, scale: 1 };
      var cw = canvas.width, ch = canvas.height;
      var scale = Math.min(cw / liveImg.width, ch / liveImg.height);
      var dw = liveImg.width * scale, dh = liveImg.height * scale;
      return { dx: (cw - dw) / 2, dy: (ch - dh) / 2, scale: scale, dw: dw, dh: dh };
    }

    function redraw() {
      var cw = canvas.width, ch = canvas.height;
      ctx.clearRect(0, 0, cw, ch);
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, 0, cw, ch);

      if (!liveImg) {
        ctx.fillStyle = '#64748b';
        ctx.font = '15px system-ui';
        ctx.textAlign = 'center';
        ctx.fillText('上传图片后在此拖拽调整前景框', cw / 2, ch / 2);
        hint.textContent = '请先上传图片';
        return;
      }

      var t = imgToCanvas();
      ctx.drawImage(liveImg, t.dx, t.dy, t.dw, t.dh);

      // Get current rect in image coords, clamp to image bounds
      var rx = Math.max(0, getControlValue('x'));
      var ry = Math.max(0, getControlValue('y'));
      var rw = Math.min(getControlValue('w'), liveImg.width - rx);
      var rh = Math.min(getControlValue('h'), liveImg.height - ry);

      // Canvas coords
      var cx = t.dx + rx * t.scale, cy = t.dy + ry * t.scale;
      var cw2 = rw * t.scale, ch2 = rh * t.scale;

      // Dim outside rect
      ctx.fillStyle = 'rgba(0,0,0,0.35)';
      ctx.fillRect(t.dx, t.dy, t.dw, t.dh);
      ctx.clearRect(cx, cy, cw2, ch2);
      ctx.drawImage(liveImg, t.dx, t.dy, t.dw, t.dh);
      ctx.globalAlpha = 0.35;
      ctx.fillStyle = '#000';
      ctx.fillRect(t.dx, t.dy, t.dw, t.dh);
      ctx.globalAlpha = 1;
      ctx.clearRect(cx, cy, cw2, ch2);
      ctx.drawImage(liveImg,
        rx, ry, rw, rh,
        cx, cy, cw2, ch2);

      // Draw rectangle border
      ctx.strokeStyle = '#22c55e';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 3]);
      ctx.strokeRect(cx, cy, cw2, ch2);
      ctx.setLineDash([]);

      // Draw resize handles
      var handles = [
        [cx, cy], [cx + cw2/2, cy], [cx + cw2, cy],
        [cx, cy + ch2/2], [cx + cw2, cy + ch2/2],
        [cx, cy + ch2], [cx + cw2/2, cy + ch2], [cx + cw2, cy + ch2]
      ];
      var hSize = 7;
      handles.forEach(function(h) {
        ctx.fillStyle = '#22c55e';
        ctx.fillRect(h[0] - hSize/2, h[1] - hSize/2, hSize, hSize);
      });

      hint.textContent = '框: (' + rx + ', ' + ry + ') ' + rw + 'x' + rh +
        ' — 拖拽框内移动，拖拽边角缩放';
    }

    function getHandle(px, py) {
      var t = imgToCanvas();
      if (!liveImg) return null;
      var rx = getControlValue('x'), ry = getControlValue('y');
      var rw = getControlValue('w'), rh = getControlValue('h');
      var cx = t.dx + rx * t.scale, cy = t.dy + ry * t.scale;
      var cw2 = rw * t.scale, ch2 = rh * t.scale;
      var hSize = 10;
      var handles = {
        'nw': [cx, cy], 'n': [cx + cw2/2, cy], 'ne': [cx + cw2, cy],
        'w': [cx, cy + ch2/2], 'e': [cx + cw2, cy + ch2/2],
        'sw': [cx, cy + ch2], 's': [cx + cw2/2, cy + ch2], 'se': [cx + cw2, cy + ch2]
      };
      for (var name in handles) {
        var h = handles[name];
        if (Math.abs(px - h[0]) < hSize && Math.abs(py - h[1]) < hSize) return name;
      }
      if (px >= cx && px <= cx + cw2 && py >= cy && py <= cy + ch2) return 'move';
      return null;
    }

    canvas.addEventListener('mousedown', function(e) {
      var rect = canvas.getBoundingClientRect();
      var scaleX = canvas.width / rect.width;
      var scaleY = canvas.height / rect.height;
      var px = (e.clientX - rect.left) * scaleX;
      var py = (e.clientY - rect.top) * scaleY;
      var handle = getHandle(px, py);
      if (handle) {
        rectState.dragging = true;
        rectState.handle = handle;
        rectState.startX = px;
        rectState.startY = py;
        rectState.startRect = {
          x: getControlValue('x'), y: getControlValue('y'),
          w: getControlValue('w'), h: getControlValue('h')
        };
        e.preventDefault();
      }
    });

    window.addEventListener('mouseup', function() {
      rectState.dragging = false;
    });

    // Touch support
    canvas.addEventListener('touchstart', function(e) {
      if (e.touches.length === 1) {
        var rect = canvas.getBoundingClientRect();
        var px = (e.touches[0].clientX - rect.left) * (canvas.width / rect.width);
        var py = (e.touches[0].clientY - rect.top) * (canvas.height / rect.height);
        var handle = getHandle(px, py);
        if (handle) {
          rectState.dragging = true;
          rectState.handle = handle;
          rectState.startX = px;
          rectState.startY = py;
          rectState.startRect = {x:getControlValue('x'), y:getControlValue('y'), w:getControlValue('w'), h:getControlValue('h')};
          e.preventDefault();
        }
      }
    }, {passive: false});

    function handleDrag(clientX, clientY) {
      if (!rectState.dragging || !liveImg) return;
      var rect = canvas.getBoundingClientRect();
      var scaleX = canvas.width / rect.width;
      var scaleY = canvas.height / rect.height;
      var px = (clientX - rect.left) * scaleX;
      var py = (clientY - rect.top) * scaleY;
      var t = imgToCanvas();
      if (!t.scale) return;
      var dx = (px - rectState.startX) / t.scale;
      var dy = (py - rectState.startY) / t.scale;
      var sr = rectState.startRect;
      var h = rectState.handle;
      var nx = sr.x, ny = sr.y, nw = sr.w, nh = sr.h;
      if (h === 'move') {
        nx = Math.max(0, Math.min(liveImg.width - nw, sr.x + dx));
        ny = Math.max(0, Math.min(liveImg.height - nh, sr.y + dy));
      } else {
        if (h.indexOf('n') >= 0) { ny = Math.max(0, Math.min(sr.y + sr.h - 10, sr.y + dy)); nh = sr.h + sr.y - ny; }
        if (h.indexOf('s') >= 0) { nh = Math.max(10, Math.min(liveImg.height - sr.y, sr.h + dy)); }
        if (h.indexOf('w') >= 0) { nx = Math.max(0, Math.min(sr.x + sr.w - 10, sr.x + dx)); nw = sr.w + sr.x - nx; }
        if (h.indexOf('e') >= 0) { nw = Math.max(10, Math.min(liveImg.width - sr.x, sr.w + dx)); }
      }
      setControlValue('x', Math.round(nx));
      setControlValue('y', Math.round(ny));
      setControlValue('w', Math.round(nw));
      setControlValue('h', Math.round(nh));
      redraw();
    }

    window.addEventListener('mousemove', function(e) { handleDrag(e.clientX, e.clientY); });
    canvas.addEventListener('touchmove', function(e) {
      if (e.touches.length === 1) { handleDrag(e.touches[0].clientX, e.touches[0].clientY); e.preventDefault(); }
    }, {passive: false});

    window.__redrawPreview = redraw;
    window.__getHandle = getHandle;
    redraw();
  }

  function renderVisualStory(cfg) {
    var section = qs('#visualStorySection');
    var root = qs('#visualStory');
    if (!section || !root) return;
    var story = cfg.visualStory || {};
    var cards = story.cards || [];
    if (!cards.length) {
      section.style.display = 'none';
      root.innerHTML = '';
      return;
    }
    section.style.display = '';
    var intro = qs('#visualStoryIntro');
    if (intro) intro.textContent = story.intro || '用图形先看懂像素和参数发生了什么。';
    root.innerHTML = cards.map(function(card) {
      return '<article class="teach-visual-card teach-visual-' + esc(card.type || 'bars') + '">' +
        '<div class="teach-visual-art">' + visualArt(card) + '</div>' +
        '<div class="teach-visual-copy"><strong>' + esc(card.title || '') + '</strong>' +
        '<p>' + esc(card.text || '') + '</p></div>' +
        '</article>';
    }).join('');
    initVisualAnimations(root);
  }

  function visualArt(card) {
    var type = card.type || 'bars';
    if (type === 'cardAnim') return cardAnimArt(card);
    if (type === 'semanticAnim') return semanticAnimArt(card);
    if (type === 'bars') return barsArt(card);
    if (type === 'histogram') return histogramArt(card);
    if (type === 'threshold') return thresholdArt(card);
    if (type === 'pixels') return pixelsArt(card);
    if (type === 'kernel') return kernelArt(card);
    if (type === 'arrows') return arrowsArt(card);
    if (type === 'window') return windowArt(card);
    if (type === 'bilateral') return bilateralArt(card);
    if (type === 'mix') return mixArt(card);
    if (type === 'gradientSet') return gradientSetArt(card);
    if (type === 'corner') return cornerArt(card);
    if (type === 'vote') return voteArt(card);
    if (type === 'morph') return morphArt(card);
    if (type === 'contour') return contourArt(card);
    if (type === 'nms') return nmsArt(card);
    if (type === 'template') return templateArt(card);
    return barsArt(card);
  }

  function cardAnimArt(card) {
    var anim = card.anim || card.step || 'gray';
    var caption = card.caption || '';
    return '<div class="teach-card-anim-wrap" data-anim-label="' + esc(caption) + '">' +
      '<canvas data-card-anim="' + esc(anim) + '" aria-label="' + esc(card.title || anim) + '"></canvas>' +
      (caption ? '<span>' + esc(caption) + '</span>' : '') +
      '</div>';
  }

  function semanticAnimArt(card) {
    var anim = card.anim || card.step || 'corner';
    var caption = card.caption || '';
    return '<div class="teach-card-anim-wrap teach-semantic-anim-wrap" data-anim-label="' + esc(caption) + '">' +
      '<canvas data-semantic-anim="' + esc(anim) + '" aria-label="' + esc(card.title || anim) + '"></canvas>' +
      (caption ? '<span>' + esc(caption) + '</span>' : '') +
      '</div>';
  }

  function stopVisualAnimations() {
    if (visualAnimRaf) {
      cancelAnimationFrame(visualAnimRaf);
      visualAnimRaf = null;
    }
    visualAnimStates = [];
    visualAnimLastTime = 0;
  }

  function initVisualAnimations(root) {
    stopVisualAnimations();
    if (window.CardAnims && window.CardAnims.init && window.CardAnims.tick) {
      qsa('[data-card-anim]', root || document).forEach(function(canvas) {
        var id = canvas.getAttribute('data-card-anim');
        var initFn = window.CardAnims.init[id];
        var tickFn = window.CardAnims.tick[id];
        if (!initFn || !tickFn) return;

        var ctx = window.CardAnims.setupCanvas(canvas);
        var state = {};
        initFn(state);
        tickFn(state, ctx, 0);
        visualAnimStates.push({ canvas: canvas, ctx: ctx, state: state, tick: tickFn, setup: window.CardAnims.setupCanvas });
      });
    }

    if (window.SemanticAnims && window.SemanticAnims.tick) {
      qsa('[data-semantic-anim]', root || document).forEach(function(canvas) {
        var id = canvas.getAttribute('data-semantic-anim');
        var ctx = window.SemanticAnims.setupCanvas(canvas);
        var state = {};
        window.SemanticAnims.init(state);
        window.SemanticAnims.tick(id, state, ctx, 0);
        visualAnimStates.push({
          canvas: canvas,
          ctx: ctx,
          state: state,
          setup: window.SemanticAnims.setupCanvas,
          tick: function(s, c, d) { window.SemanticAnims.tick(id, s, c, d); }
        });
      });
    }

    if (visualAnimStates.length) {
      visualAnimRaf = requestAnimationFrame(visualAnimationLoop);
    }
  }

  function visualAnimationLoop(timestamp) {
    if (!visualAnimLastTime) visualAnimLastTime = timestamp;
    var dt = Math.min(80, timestamp - visualAnimLastTime);
    visualAnimLastTime = timestamp;

    visualAnimStates.forEach(function(anim) {
      var parentW = anim.canvas.parentElement.getBoundingClientRect().width;
      var currentW = Math.round(anim.canvas.getBoundingClientRect().width);
      if (Math.abs(currentW - parentW) > 1) {
        anim.ctx = anim.setup(anim.canvas);
      }
      anim.tick(anim.state, anim.ctx, dt);
    });

    visualAnimRaf = requestAnimationFrame(visualAnimationLoop);
  }

  function barsArt(card) {
    var items = card.items || [];
    return '<div class="visual-bars">' + items.map(function(item) {
      var value = Math.max(0, Math.min(100, Number(item.value || 0)));
      return '<div class="visual-bar-row"><span>' + esc(item.label || '') + '</span>' +
        '<div class="visual-bar-track"><i style="width:' + value + '%;background:' + esc(item.color || '#66f2c2') + '"></i></div>' +
        '<b>' + esc(item.caption || value + '%') + '</b></div>';
    }).join('') + '</div>';
  }

  function histogramArt(card) {
    var values = card.values || [8, 12, 28, 44, 58, 48, 36, 24, 18, 34, 55, 72, 64, 42, 26, 12];
    var max = Math.max.apply(Math, values.concat([1]));
    return '<div class="visual-histogram">' + values.map(function(v, idx) {
      var h = Math.round(Number(v) / max * 100);
      return '<i style="height:' + h + '%" data-zone="' + (idx < values.length / 2 ? 'dark' : 'bright') + '"></i>';
    }).join('') + '</div><div class="visual-axis"><span>暗</span><span>亮</span></div>';
  }

  function thresholdArt(card) {
    var pos = Math.max(4, Math.min(96, Number(card.position || 52)));
    return '<div class="visual-threshold">' +
      '<div class="visual-gradient"></div><i style="left:' + pos + '%"></i>' +
      '<div class="visual-split"><span>背景</span><span>前景</span></div>' +
      '</div>';
  }

  function pixelsArt(card) {
    var rows = Number(card.rows || 8);
    var cols = Number(card.cols || 12);
    var cells = card.cells || imageCells(rows, cols, card.palette);
    return '<div class="visual-pixels visual-image-grid" style="--rows:' + rows + ';--cols:' + cols + '">' + cells.map(function(color, idx) {
      return '<i style="background:' + esc(color) + '" title="' + idx + '"></i>';
    }).join('') + '</div>';
  }

  function imageCells(rows, cols, palette) {
    var colors = palette || ['#0f172a', '#1d4ed8', '#38bdf8', '#22c55e', '#f97316', '#f8fafc'];
    var cells = [];
    for (var y = 0; y < rows; y += 1) {
      for (var x = 0; x < cols; x += 1) {
        var cx = (x - cols * 0.34) / cols;
        var cy = (y - rows * 0.46) / rows;
        var wave = Math.sin(x * 0.82) + Math.cos(y * 0.9) + (x / Math.max(cols - 1, 1)) * 1.4;
        var blob = (cx * cx + cy * cy) < 0.035 ? 2 : 0;
        var idx = Math.max(0, Math.min(colors.length - 1, Math.floor((wave + 2.2) / 4.6 * colors.length) + blob));
        cells.push(colors[idx]);
      }
    }
    return cells;
  }

  function kernelArt(card) {
    var values = card.values || [1, 2, 1, 2, 4, 2, 1, 2, 1];
    var max = Math.max.apply(Math, values.concat([1]));
    var patch = imageCells(8, 8, ['#111827', '#1e293b', '#334155', '#64748b', '#94a3b8', '#e2e8f0']);
    var kernel = '<div class="visual-kernel">' + values.map(function(v) {
      var alpha = 0.18 + Number(v) / max * 0.72;
      return '<i style="background:rgba(102,242,194,' + alpha.toFixed(2) + ')">' + esc(v) + '</i>';
    }).join('') + '</div>';
    return '<div class="visual-kernel-scene">' +
      '<div class="visual-pixels visual-image-grid visual-patch-grid" style="--rows:8;--cols:8">' +
      patch.map(function(color, idx) { return '<i style="background:' + esc(color) + '" title="' + idx + '"></i>'; }).join('') +
      '</div>' +
      '<div class="visual-kernel-overlay">' + kernel + '</div>' +
      '</div>';
  }

  function arrowsArt(card) {
    var labels = card.labels || ['Gx', 'Gy', '|G|'];
    return '<div class="visual-arrows">' +
      '<span>' + esc(labels[0]) + '</span><i class="arrow-x"></i>' +
      '<span>' + esc(labels[1]) + '</span><i class="arrow-y"></i>' +
      '<span>' + esc(labels[2]) + '</span><i class="arrow-mag"></i>' +
      '</div>';
  }

  function windowArt(card) {
    var values = card.values || [8, 10, 12, 13, 240, 14, 15, 16, 18];
    var sorted = values.slice().sort(function(a, b) { return a - b; });
    var median = sorted[Math.floor(sorted.length / 2)];
    // 3x3 grid positions for the scanning window
    var positions = [
      {col:3,row:3},{col:4,row:3},{col:5,row:3},
      {col:3,row:4},{col:4,row:4},{col:5,row:4},
      {col:3,row:5},{col:4,row:5},{col:5,row:5}
    ];
    return '<div class="visual-window">' +
      // Scene: 8x8 image with raster-scanning window
      '<div class="visual-window-scene">' +
        '<div class="visual-pixels visual-image-grid visual-patch-grid" style="--rows:8;--cols:8">' +
          imageCells(8, 8, ['#111827', '#1e293b', '#475569', '#64748b', '#cbd5e1', '#f8fafc']).map(function(color, idx) {
            var col = idx % 8, row = Math.floor(idx / 8);
            var isNoise = idx === 26 || idx === 36 || idx === 44;
            return '<i class="' + (isNoise ? 'is-noise' : '') + '" style="background:' + esc(isNoise ? '#f8fafc' : color) + '" title="' + idx + '"></i>';
          }).join('') +
        '</div>' +
        // Raster scan window — steps through 9 positions
        '<div class="visual-sample-window visual-raster-window">' +
          positions.map(function(p, i) {
            return '<span class="rw-step" style="--rw-col:' + p.col + ';--rw-row:' + p.row + ';animation-delay:' + (i * 0.44).toFixed(2) + 's"></span>';
          }).join('') +
        '</div>' +
      '</div>' +
      // Unsorted→Sorted transition stage
      '<div class="visual-window-stage">' +
        // Unsorted grid (visible first, fades out)
        '<div class="visual-window-unsorted">' +
          '<div class="visual-kernel visual-window-grid">' + values.map(function(v) {
            return '<i class="' + (v === median ? 'is-median' : '') + '">' + esc(v) + '</i>';
          }).join('') + '</div>' +
          '<span class="vw-label">滑动窗口取值</span>' +
        '</div>' +
        // Sorted row (fades in)
        '<div class="visual-window-sorted">' +
          '<div class="visual-sort">' + sorted.map(function(v) {
            return '<span class="' + (v === median ? 'is-median' : '') + '">' + esc(v) + '</span>';
          }).join('') + '</div>' +
          '<span class="vw-label">排序后取中值 = <strong>' + esc(median) + '</strong></span>' +
        '</div>' +
      '</div>' +
      '</div>';
  }

  function bilateralArt(card) {
    return '<div class="visual-bilateral">' +
      '<div class="bilateral-scene"><span></span><b></b><i></i></div>' +
      '<div class="bilateral-weights"><em>空间近</em><em>颜色像</em><strong>高权重</strong></div>' +
      '<div class="bilateral-weights muted"><em>空间近</em><em>颜色差</em><strong>低权重</strong></div>' +
      '</div>';
  }

  function mixArt(card) {
    var labels = card.labels || ['R', 'G', 'B'];
    return '<div class="visual-rgb-mix">' +
      '<span class="rgb-lobe red">' + esc(labels[0]) + '</span>' +
      '<span class="rgb-lobe green">' + esc(labels[1]) + '</span>' +
      '<span class="rgb-lobe blue">' + esc(labels[2]) + '</span>' +
      '<strong>RGB</strong>' +
      '</div>';
  }

  function gradientSetArt(card) {
    var rows = card.rows || [];
    return '<div class="visual-gradient-set">' + rows.map(function(row) {
      return '<div class="visual-gradient-row">' +
        '<span>' + esc(row.label || '') + '</span>' +
        '<i style="background:' + esc(row.gradient || 'linear-gradient(90deg,#020617,#f8fafc)') + '"></i>' +
        '<b>' + esc(row.caption || '') + '</b>' +
        '</div>';
    }).join('') + '</div>';
  }

  function cornerArt(card) {
    return '<div class="visual-corner-stage">' +
      '<div class="corner-patch">' +
        '<i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i>' +
        '<b class="corner-window"></b><b class="corner-dot"></b>' +
      '</div>' +
      '<div class="corner-eigen"><span>lambda1 high</span><span>lambda2 high</span><strong>corner</strong></div>' +
      '</div>';
  }

  function voteArt(card) {
    return '<div class="visual-vote-space visual-hough-stage">' +
      '<div class="hough-image">' +
        '<i class="hough-edge p1"></i><i class="hough-edge p2"></i><i class="hough-edge p3"></i><i class="hough-edge p4"></i>' +
        '<b class="hough-candidate a"></b><b class="hough-candidate b"></b><b class="hough-candidate c"></b>' +
      '</div>' +
      '<em class="hough-arrow">vote</em>' +
      '<div class="hough-accumulator">' +
        '<i class="vote-line" style="--top:36%;--rot:-18deg"></i>' +
        '<i class="vote-line" style="--top:50%;--rot:5deg"></i>' +
        '<i class="vote-line" style="--top:64%;--rot:22deg"></i>' +
        '<b class="vote-peak"></b><span>rho/theta peak</span>' +
      '</div>' +
      '</div>';
  }

  function morphArt(card) {
    var mode = card.mode || 'dilate';
    var after = mode === 'erode'
      ? ['', '', '', '', '', 'on', 'on', '', '', '', '', '', '', '', '', '']
      : ['', 'on', 'on', '', 'on', 'on', 'on', 'on', '', 'on', 'on', 'on', '', '', 'on', ''];
    return '<div class="visual-shape-stage visual-morph-stage ' + esc(mode) + '">' +
      '<div class="morph-grid before">' +
        '<i></i><i class="on"></i><i class="on"></i><i></i><i></i><i class="on"></i><i class="on"></i><i class="on"></i><i></i><i></i><i class="on"></i><i></i><i></i><i></i><i></i><i></i>' +
        '<b class="shape-kernel"></b>' +
      '</div>' +
      '<strong>' + (mode === 'erode' ? 'all hit -> keep' : 'any hit -> grow') + '</strong>' +
      '<div class="morph-grid after">' + after.map(function(cls) { return '<i class="' + cls + '"></i>'; }).join('') + '</div>' +
      '</div>';
  }

  function contourArt(card) {
    return '<div class="visual-contour-stage visual-contour-semantic">' +
      '<span class="contour-blob"></span>' +
      '<i class="contour-points"></i>' +
      '<b class="contour-cursor"></b>' +
      '<div class="contour-strip"><span>p1</span><span>p2</span><span>p3</span><span>...</span><span>pn</span></div>' +
      '</div>';
  }

  function nmsArt(card) {
    return '<div class="visual-nms-stage visual-nms-semantic">' +
      '<span class="nms-ridge"></span>' +
      '<i class="nms-fade left"></i>' +
      '<i class="nms-fade right"></i>' +
      '<b class="nms-needle"></b>' +
      '<em class="nms-direction">compare along gradient</em>' +
      '<strong>keep max</strong>' +
      '</div>';
  }

  function templateArt(card) {
    return '<div class="visual-template-stage visual-template-semantic">' +
      '<div class="template-image"><span class="template-target"></span><i class="template-window"></i></div>' +
      '<div class="template-patch">template</div>' +
      '<b class="template-heat"></b>' +
      '<em>response peak</em>' +
      '</div>';
  }

  function renderMetrics(metrics) {
    var root = qs('#metrics');
    var merged = Object.assign({}, metrics || {});
    var cfg = window.__TEACHING_CFG__;
    var impl = cfg && cfg.implementation;
    var isSim = impl && impl.category === 'teaching_simulation';
    var isReal = impl && (impl.category === 'numpy_algorithm' || impl.category === 'pretrained_model' || impl.category === 'numpy_model');

    if (impl) {
      merged['实现方式'] = impl.status || '';
      if (impl.model) merged['模型/后端'] = impl.model;
    }

    // Build a prominent badge
    var badge = '';
    if (metrics && metrics.status === 'offline_teaching') {
      badge = '<div class="teach-result-badge offline">离线教学演示 — 非真实模型推理结果</div>';
    } else if (isSim) {
      badge = '<div class="teach-result-badge offline">离线教学演示 — 下方为模拟可视化</div>';
    } else if (isReal) {
      badge = '<div class="teach-result-badge real">真实运算结果</div>';
    } else if (metrics && metrics.status === 'pretrained_model') {
      badge = '<div class="teach-result-badge real">真实预训练模型推理</div>';
    }

    var keys = Object.keys(merged || {});
    if (!keys.length && !badge) {
      root.innerHTML = '<span class="teach-metric">运行后显示指标</span>';
      return;
    }
    root.innerHTML = badge + keys.map(function(key) {
      var value = merged[key];
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
        (step.formula ? '<div class="teach-step-formula" data-step-formula="' + esc(step.id) + '">' + esc(step.formula) + '</div>' : '') +
        '</div></article>';
    }).join('');
    qsa('[data-step-formula]', root).forEach(function(node) {
      renderFormula(node, node.textContent || '');
    });
  }

  function renderConceptSteps(cfg) {
    renderSteps((cfg.conceptSteps || cfg.pipeline || []).map(function(item, idx) {
      if (Array.isArray(item)) {
        return { id: 'concept_' + idx, name: item[0], explanation: item[1], formula: item[2] || '' };
      }
      return item;
    }), cfg);
    renderMetrics(cfg.metrics || {
      '展示类型': cfg.status || '教学讲解',
      '本地推理': cfg.endpoint ? '可运行' : '未接入'
    });
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

  function syncBackendStatus() {
    var id = paramId();
    return fetch('/api/modules')
      .then(function(res) { return res.json(); })
      .then(function(payload) {
        var impl = null;
        (payload.phases || []).forEach(function(phase) {
          (phase.modules || []).forEach(function(mod) {
            if (mod.id === id && mod.implementation) impl = mod.implementation;
          });
        });
        if (!impl) return;
        var statusNode = qs('[data-status]');
        if (statusNode) statusNode.textContent = impl.status || statusNode.textContent;
        var progressNode = qs('#status');
        if (progressNode) {
          if (impl.category === 'not_implemented') {
            progressNode.textContent = '当前模块未接入真实实现。';
          } else if (impl.category === 'requires_external_weights') {
            progressNode.textContent = '需要本地权重文件，当前仅可查看原理讲解。';
          } else if (impl.category === 'model_not_available') {
            progressNode.textContent = '模型未配置。';
          } else if (impl.category === 'teaching_simulation') {
            progressNode.textContent = '离线教学演示 — 非真实运算，用于理解算法原理。';
          } else if (impl.category === 'numpy_algorithm') {
            progressNode.textContent = '真实 NumPy 算法 — 点击运行获得真实计算结果。';
          } else if (impl.category === 'pretrained_model') {
            progressNode.textContent = '真实预训练模型 — 点击运行加载模型推理。';
          }
        }
      })
      .catch(function() {});
  }

  function collectFormData(file, cfg) {
    var fd = new FormData();
    if (file) fd.set('file', file, file.name || 'upload.png');
    qsa('#controls input, #controls select').forEach(function(input) {
      fd.set(input.name, input.value);
    });
    return fd;
  }

  function run(cfg, file) {
    var status = qs('#status');
    var button = qs('#runButton');
    var impl = cfg.implementation || {};
    if (impl.category === 'requires_external_weights') {
      renderConceptSteps(cfg);
      status.textContent = impl.note || '该算法需要本地权重或远程模型，本轮离线模式暂不运行。';
      return;
    }
    if (!cfg.endpoint) {
      renderConceptSteps(cfg);
      status.textContent = '当前为论文/参考实现讲解页，未接入本地推理。';
      return;
    }
    if (!file && !(impl.category === 'teaching_simulation' || impl.requiresUpload === false || impl.requires_upload === false)) {
      status.textContent = '请先选择图片。';
      return;
    }
    status.textContent = impl.realModel
      ? '正在运行真实预训练模型...'
      : '正在运行算法...';
    button.disabled = true;
    var formBody = collectFormData(file, cfg);
    fetch(cfg.endpoint, { method: 'POST', body: formBody })
      .then(function(res) {
        if (!res.ok) {
          return res.text().then(function(txt) {
            var msg = txt;
            try { var j = JSON.parse(txt); msg = j.error || j.message || msg; } catch(e) {}
            throw new Error(msg || ('HTTP ' + res.status));
          });
        }
        return res.json();
      })
      .then(function(json) {
        if (json.implementation) {
          qs('[data-status]').textContent = json.implementation.status || cfg.status || '';
        }
        renderSteps(json.steps || [], cfg);
        renderMetrics(json.metrics || {});
        status.textContent = '已生成真实后端结果。';
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
    window.__TEACHING_CFG__ = cfg;
    renderStatic(cfg);
    initRectCanvas(cfg);
    if (cfg.endpoint) {
      renderSteps([], cfg);
      renderMetrics({});
    } else {
      renderConceptSteps(cfg);
    }

    var selectedFile = null;
    var upload = qs('#upload');
    var input = qs('#fileInput');
    var filename = qs('#filename');
    var runButton = qs('#runButton');
    var status = qs('#status');

    var impl = cfg.implementation || {};

    // Hide upload zone for modules that don't need image input
    // (external weights, GAN/DDPM/diffusion that generate from noise/text)
    var isExtWeight = impl.category === 'requires_external_weights';
    var isGenerative = ['gan','ddpm','diffusion'].indexOf(id) >= 0;
    var needsUpload = !isExtWeight && !isGenerative && cfg.endpoint;

    if (!cfg.endpoint) {
      qs('#runButton').disabled = false;
      qs('#runButton').textContent = '查看流程';
      qs('#demoButton').style.display = 'none';
      qs('#status').textContent = '论文/参考实现讲解模式。';
    } else if (isExtWeight) {
      qs('#runButton').disabled = false;
      qs('#runButton').textContent = '查看说明';
      qs('#demoButton').style.display = 'none';
      qs('#upload').style.display = 'none';
      qs('#status').textContent = '需要本地权重文件。当前页面讲解算法原理与网络结构。';
    } else {
      qs('#runButton').disabled = false;
      qs('#runButton').textContent = '运行算法';
      qs('#demoButton').style.display = '';
      if (isGenerative) {
        qs('#upload').style.display = 'none';
        qs('#status').textContent = '无需输入图片，直接从噪声/提示生成。';
      } else {
        qs('#status').textContent = '点击运行算法，或先上传图片。';
      }
    }

    function setFile(file) {
      selectedFile = file;
      filename.textContent = file ? file.name : '未选择文件';
      if (file && cfg.endpoint) {
        status.textContent = '已选择图片，点击”运行算法”开始处理。';
        runButton.disabled = false;
      }
      if (file) {
        var reader = new FileReader();
        reader.onload = function(e) {
          var img = new Image();
          img.onload = function() {
            liveImg = img;
            // Update rect slider ranges to cover full image
            var xEl = qs('#controls [name=\”x\”]');
            var yEl = qs('#controls [name=\”y\”]');
            var wEl = qs('#controls [name=\”w\”]');
            var hEl = qs('#controls [name=\”h\”]');
            if (xEl && yEl && wEl && hEl) {
              // Update slider ranges to cover full image, step=1 for smooth drag
              [xEl, yEl, wEl, hEl].forEach(function(el) {
                el.step = 1;
              });
              xEl.max = img.width;
              yEl.max = img.height;
              wEl.max = img.width;
              hEl.max = img.height;
              // Default rect: 30% of image, centered
              var dw = Math.round(img.width * 0.3);
              var dh = Math.round(img.height * 0.3);
              var dx = Math.round((img.width - dw) / 2);
              var dy = Math.round((img.height - dh) / 2);
              setControlValue('x', dx);
              setControlValue('y', dy);
              setControlValue('w', dw);
              setControlValue('h', dh);
            }
            if (window.__redrawPreview) window.__redrawPreview();
          };
          img.src = e.target.result;
        };
        reader.readAsDataURL(file);
      }
    }

    upload.addEventListener('click', function(ev) {
      if (ev.target !== input) {
        ev.preventDefault();
        input.click();
      }
    });
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
    input.addEventListener('click', function() {
      input.value = '';
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
  window.addEventListener('load', syncBackendStatus);
})();
