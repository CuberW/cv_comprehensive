// Sync algorithm subpages with the main shell theme.
(function(){
  var FINAL_THEME_CSS = [
    'html body .page,html body .match-page,html body .teach-page,html body .nn-page{width:min(1540px,calc(100% - 48px))!important;max-width:1540px!important;}@media(max-width:920px){html body .page,html body .match-page,html body .teach-page,html body .nn-page{width:min(100% - 20px,1540px)!important;}}',
    'html[data-theme="light"] body{background:var(--app-gradient)!important;color:var(--cv-text)!important;}',
    'html[data-theme="light"] .glass,html[data-theme="light"] .panel,html[data-theme="light"] .card,html[data-theme="light"] .hero-main,html[data-theme="light"] .relation-card,html[data-theme="light"] .step-card,html[data-theme="light"] .step,html[data-theme="light"] .concept,html[data-theme="light"] .analogy,html[data-theme="light"] .loading,html[data-theme="light"] .result-box,html[data-theme="light"] .preview-box,html[data-theme="light"] .upload-section,html[data-theme="light"] .teach-title-panel,html[data-theme="light"] .teach-io-panel,html[data-theme="light"] .teach-section,html[data-theme="light"] .teach-core-item,html[data-theme="light"] .teach-ref,html[data-theme="light"] .teach-visual-card,html[data-theme="light"] .teach-stage,html[data-theme="light"] .teach-step-card,html[data-theme="light"] .teach-metric,html[data-theme="light"] .teach-live-preview,html[data-theme="light"] .nn-story-card,html[data-theme="light"] .nn-formula-card,html[data-theme="light"] .nn-guide-item,html[data-theme="light"] .nn-meta-item,html[data-theme="light"] .nn-step-card,html[data-theme="light"] .nn-control,html[data-theme="light"] .nn-note,html[data-theme="light"] .rc{background:var(--cv-panel)!important;color:var(--cv-text)!important;border-color:var(--cv-border)!important;box-shadow:var(--cv-shadow)!important;}',
    'html[data-theme="light"] .page-header,html[data-theme="light"] .hero-copy{background:var(--cv-page-header-bg)!important;color:var(--cv-text)!important;border-color:var(--cv-border)!important;}',
    'html[data-theme="light"] h1,html[data-theme="light"] h2,html[data-theme="light"] h3,html[data-theme="light"] .page-title,html[data-theme="light"] .step-title,html[data-theme="light"] .step-name,html[data-theme="light"] .step-body strong,html[data-theme="light"] .metric-value,html[data-theme="light"] .filename,html[data-theme="light"] .file-picker-name,html[data-theme="light"] .file-meta strong,html[data-theme="light"] .concept strong,html[data-theme="light"] .teach-title,html[data-theme="light"] .teach-panel-title,html[data-theme="light"] .teach-section-head h2,html[data-theme="light"] .teach-core-item strong,html[data-theme="light"] .teach-visual-copy strong,html[data-theme="light"] .teach-stage strong,html[data-theme="light"] .teach-step-body strong,html[data-theme="light"] .nn-story-title,html[data-theme="light"] .nn-formula-name,html[data-theme="light"] .nn-step-title,html[data-theme="light"] .nn-meta-value,html[data-theme="light"] .rc .c{color:var(--cv-title)!important;}',
    'html[data-theme="light"] p,html[data-theme="light"] .subtitle,html[data-theme="light"] .page-subtitle,html[data-theme="light"] .step-desc,html[data-theme="light"] .step-body p,html[data-theme="light"] .metric-label,html[data-theme="light"] .file-meta,html[data-theme="light"] .upload-info,html[data-theme="light"] .concept span,html[data-theme="light"] .results-head p,html[data-theme="light"] .empty-hint,html[data-theme="light"] .no-img,html[data-theme="light"] .teach-subtitle,html[data-theme="light"] .teach-section-head p,html[data-theme="light"] .teach-core-item span,html[data-theme="light"] .teach-copy p,html[data-theme="light"] .teach-ref span,html[data-theme="light"] .teach-ref a,html[data-theme="light"] .teach-visual-copy p,html[data-theme="light"] .teach-stage span,html[data-theme="light"] .teach-step-body p,html[data-theme="light"] .teach-status,html[data-theme="light"] .nn-story-text,html[data-theme="light"] .nn-step-desc,html[data-theme="light"] .nn-meta-label,html[data-theme="light"] .nn-pipe-desc,html[data-theme="light"] .rc .e,html[data-theme="light"] .note{color:var(--cv-muted)!important;}',
    'html[data-theme="light"] .step-img,html[data-theme="light"] .img-area,html[data-theme="light"] .step-card .img-area,html[data-theme="light"] .image-frame,html[data-theme="light"] .image-panel,html[data-theme="light"] .match-result-image,html[data-theme="light"] .canvas-row,html[data-theme="light"] .anim-canvas-wrap,html[data-theme="light"] .viewport,html[data-theme="light"] .mock-photo,html[data-theme="light"] .thresh-viz,html[data-theme="light"] .teach-visual-art,html[data-theme="light"] .teach-card-anim-wrap,html[data-theme="light"] .teach-step-image,html[data-theme="light"] .nn-step-media,html[data-theme="light"] .rc img{background:var(--cv-media-bg)!important;color:var(--cv-muted)!important;border-color:var(--cv-border)!important;}',
    'html[data-theme="light"] input,html[data-theme="light"] select,html[data-theme="light"] textarea,html[data-theme="light"] .file-picker,html[data-theme="light"] .upload-box,html[data-theme="light"] .ghost-btn,html[data-theme="light"] .btn-secondary,html[data-theme="light"] .view-tab,html[data-theme="light"] .stage-tabs button,html[data-theme="light"] .algo-btn,html[data-theme="light"] .teach-upload,html[data-theme="light"] .teach-control,html[data-theme="light"] .teach-button,html[data-theme="light"] .nn-chip,html[data-theme="light"] .nn-control input,html[data-theme="light"] .bdemo{background:var(--cv-control-bg)!important;color:var(--cv-text)!important;border-color:var(--cv-border)!important;}',
    'html[data-theme="light"] pre,html[data-theme="light"] code,html[data-theme="light"] .step-formula,html[data-theme="light"] .formula-card,html[data-theme="light"] .math-card,html[data-theme="light"] .math-panel,html[data-theme="light"] .teach-formula,html[data-theme="light"] .teach-step-formula,html[data-theme="light"] .nn-formula-expr,html[data-theme="light"] .nn-step-formula,html[data-theme="light"] .nn-step-data{background:var(--cv-code-bg)!important;color:var(--cv-text)!important;border-color:var(--cv-border)!important;}',
    'html[data-theme="light"] .error,html[data-theme="light"] .error-box{background:rgba(220,38,38,.08)!important;color:var(--cv-danger-text)!important;border-color:rgba(220,38,38,.24)!important;}'
  ].join('\n');

  function normaliseTheme(theme){
    return theme === 'light' ? 'light' : 'dark';
  }

  function standaloneTheme(){
    try{
      return normaliseTheme(localStorage.getItem('theme') || 'dark');
    }catch(e){
      return 'dark';
    }
  }

  function parentTheme(){
    try{
      if(window.parent && window.parent !== window){
        var root = window.parent.document.documentElement;
        return normaliseTheme(root.getAttribute('data-theme') || standaloneTheme());
      }
    }catch(e){}
    return standaloneTheme();
  }

  function applyTheme(theme){
    document.documentElement.setAttribute('data-theme', normaliseTheme(theme));
    ensureFinalThemeOverrides();
  }

  function ensureFinalThemeOverrides(){
    if(!document.head) return;
    var style = document.getElementById('cv-theme-final-overrides');
    if(!style){
      style = document.createElement('style');
      style.id = 'cv-theme-final-overrides';
      style.textContent = FINAL_THEME_CSS;
    }
    document.head.appendChild(style);
  }

  applyTheme(parentTheme());

  if(document.readyState === 'loading'){
    document.addEventListener('DOMContentLoaded', ensureFinalThemeOverrides);
  }else{
    ensureFinalThemeOverrides();
  }

  window.addEventListener('message',function(e){
    if(e.data && e.data.type==='theme'){
      applyTheme(e.data.theme || standaloneTheme());
    }
  });

  window.addEventListener('storage',function(e){
    if(e.key === 'theme'){
      applyTheme(e.newValue || 'dark');
    }
  });
})();
