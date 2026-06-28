(function () {
  'use strict';

  function escapeHtml(value) {
    return String(value == null ? '' : value).replace(/[&<>"']/g, function (ch) {
      return {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;'
      }[ch];
    });
  }

  if (!window.escapeHtml) {
    window.escapeHtml = escapeHtml;
  }
  if (!window.esc) {
    window.esc = window.escapeHtml;
  }

  function normalizeLatex(value) {
    return String(value == null ? '' : value)
      .replace(/\$\$/g, '')
      .replace(/^\s*\\\[\s*|\s*\\\]\s*$/g, '')
      .replace(/^\s*\\\(\s*|\s*\\\)\s*$/g, '')
      .trim();
  }

  var mathJaxPromise = null;

  function ensureMathJax() {
    if (window.MathJax && window.MathJax.typesetPromise) {
      return Promise.resolve(window.MathJax);
    }
    if (mathJaxPromise) return mathJaxPromise;
    window.MathJax = window.MathJax || {
      tex: { inlineMath: [['\\(', '\\)']], displayMath: [['\\[', '\\]']] },
      svg: { fontCache: 'global' },
      options: { skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre'] }
    };
    mathJaxPromise = new Promise(function (resolve, reject) {
      var existing = document.getElementById('MathJax-script');
      if (existing) {
        existing.addEventListener('load', function () { resolve(window.MathJax); });
        existing.addEventListener('error', reject);
        return;
      }
      var script = document.createElement('script');
      script.id = 'MathJax-script';
      script.async = true;
      script.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js';
      script.onload = function () { resolve(window.MathJax); };
      script.onerror = reject;
      document.head.appendChild(script);
    });
    return mathJaxPromise;
  }

  function renderLatex(target, formula, options) {
    if (!target) return;
    var latex = normalizeLatex(formula);
    target.classList.add('latex-formula');
    target.dataset.latexSource = latex;
    if (!latex) {
      target.textContent = '';
      return;
    }
    var display = !options || options.display !== false;
    target.textContent = display ? '\\[' + latex + '\\]' : '\\(' + latex + '\\)';
    ensureMathJax().then(function () {
      if (!window.MathJax || !window.MathJax.typesetPromise) return;
      if (window.MathJax.typesetClear) window.MathJax.typesetClear([target]);
      return window.MathJax.typesetPromise([target]);
    }).catch(function () {
      target.textContent = latex;
      target.dataset.mathFallback = '1';
    });
  }

  function renderLatexIn(root) {
    root = root || document;
    var nodes = root.querySelectorAll(
      '[data-latex], .step-formula, .teach-step-formula, .nn-step-formula, #stageFormula, #formulaBox'
    );
    Array.prototype.forEach.call(nodes, function (node) {
      if (node.querySelector && node.querySelector('mjx-container') && !node.dataset.latex) return;
      var formula = node.dataset.latex || node.textContent;
      renderLatex(node, formula, { display: true });
    });
  }

  window.normalizeLatex = window.normalizeLatex || normalizeLatex;
  window.ensureMathJax = window.ensureMathJax || ensureMathJax;
  window.renderLatex = window.renderLatex || renderLatex;
  window.renderLatexIn = window.renderLatexIn || renderLatexIn;

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () {
      renderLatexIn(document);
      observeLatexMutations();
    });
  } else {
    setTimeout(function () {
      renderLatexIn(document);
      observeLatexMutations();
    }, 0);
  }

  function observeLatexMutations() {
    if (!window.MutationObserver || document.documentElement.dataset.latexObserver === '1') return;
    document.documentElement.dataset.latexObserver = '1';
    var timer = null;
    var observer = new MutationObserver(function () {
      clearTimeout(timer);
      timer = setTimeout(function () { renderLatexIn(document); }, 80);
    });
    observer.observe(document.body || document.documentElement, {
      childList: true,
      subtree: true
    });
  }
})();
