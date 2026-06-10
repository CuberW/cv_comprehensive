/**
 * Main application logic.
 * Manages sidebar navigation, gallery home page, module detail views.
 * Communicates with algorithm pages inside iframes via postMessage.
 */
const App = (() => {
  // ---- DOM refs (initialized on DOMContentLoaded) ----
  let $sidebar, $sidebarNav, $gallery, $detailView, $detailIframe, $detailTitle;
  let $categoryGrid, $ladderSteps;

  // ---- State ----
  let _phases = [];          // Phase list from API
  let _modules = [];         // Flat module list
  let _currentModule = null;

  // ================================================================
  //  Init
  // ================================================================

  function init() {
    $sidebar = document.getElementById('sidebar');
    $sidebarNav = document.getElementById('sidebar-nav');
    $gallery = document.getElementById('gallery');
    $detailView = document.getElementById('detail-view');
    $detailIframe = document.getElementById('detail-iframe');
    $detailTitle = document.getElementById('detail-title');
    $categoryGrid = document.getElementById('category-grid');
    $ladderSteps = document.getElementById('ladder-steps');

    document.getElementById('btn-back').addEventListener('click', _closeDetail);
    document.getElementById('btn-search').addEventListener('click', _toggleSearch);
    document.getElementById('search-input').addEventListener('input', Utils.debounce(_onSearch, 200));

    document.querySelectorAll('.ladder-step').forEach(el => {
      el.addEventListener('click', () => _filterByDifficulty(parseInt(el.dataset.level)));
    });

    _initSidebarResizer();
    _fetchModules();

    Router.on('/', () => _showGallery());
    Router.on('/module/:id', (params) => _openModule(params.id));

    window.addEventListener('message', _onIframeMessage);
  }

  // ================================================================
  //  Data fetching
  // ================================================================

  async function _fetchModules() {
    try {
      const res = await fetch('/api/modules');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      Utils.assertFields(data, ['phases', 'total'], '/api/modules response');

      _phases = data.phases || [];
      _modules = [];
      _phases.forEach(phase => {
        phase.modules.forEach(m => {
          _modules.push({ ...m, phase_id: phase.phase_id, phase_name: phase.phase_name, phase_color: phase.color });
        });
      });

      _renderSidebar(_phases);
      _renderGallery(_phases);
      _renderDifficultyLadder();

    } catch (err) {
      console.error('[App] Failed to load modules:', err);
      Utils.toast('加载模块列表失败，后端是否已启动？');
    }
  }

  // ================================================================
  //  Sidebar navigation
  // ================================================================

  function _renderSidebar(phases) {
    $sidebarNav.innerHTML = '';

    phases.forEach(phase => {
      const header = document.createElement('div');
      header.className = 'nav-category';
      header.dataset.phaseId = phase.phase_id;
      header.innerHTML = `<span class="phase-dot" style="background:${phase.color}"></span> ${phase.phase_name}`;
      $sidebarNav.appendChild(header);

      phase.modules.forEach(m => {
        const btn = document.createElement('button');
        btn.className = 'nav-item';
        btn.dataset.moduleId = m.id;
        btn.innerHTML = `
          <span>${m.name}${m.required ? '<i class="nav-req-dot"></i>' : ''}</span>
          <span class="nav-difficulty">${'★'.repeat(m.difficulty)}</span>
        `;
        if (m.required) btn.classList.add('nav-required');
        btn.addEventListener('click', () => Router.go('/module/' + m.id));
        $sidebarNav.appendChild(btn);
      });
    });
  }

  // ================================================================
  //  Gallery home page
  // ================================================================

  function _renderGallery(phases) {
    $categoryGrid.innerHTML = '';

    phases.forEach(phase => {
      const card = document.createElement('div');
      card.className = 'category-card';
      card.style.borderLeft = `3px solid ${phase.color}`;
      card.innerHTML = `
        <div class="cat-emoji" style="color:${phase.color}">阶段 ${phase.emoji}</div>
        <div class="cat-name">${phase.phase_name}</div>
        <div class="cat-name-en">${phase.phase_name_en}</div>
        <div class="cat-count">${phase.modules.length} 个算法模块</div>
        <div class="category-modules">
          ${phase.modules.map(m => `<span class="module-chip${m.required ? ' chip-required' : ''}" style="background:${phase.color}18;color:${phase.color}" data-module-id="${m.id}">${m.name}</span>`).join('')}
        </div>
      `;

      card.addEventListener('click', (e) => {
        if (e.target.classList.contains('module-chip')) {
          Router.go('/module/' + e.target.dataset.moduleId);
          return;
        }
        _scrollSidebarToPhase(phase.phase_id);
      });

      $categoryGrid.appendChild(card);
    });
  }

  function _renderDifficultyLadder() {
    const counts = { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 };
    _modules.forEach(m => {
      if (counts[m.difficulty] !== undefined) counts[m.difficulty]++;
    });

    document.querySelectorAll('.ladder-step').forEach(el => {
      const lv = parseInt(el.dataset.level);
      const hint = el.querySelector('.step-hint');
      if (hint && counts[lv] > 0) {
        hint.textContent = `${counts[lv]} 个模块`;
      }
    });
  }

  // ================================================================
  //  Module detail
  // ================================================================

  function _openModule(moduleId) {
    const mod = _modules.find(m => m.id === moduleId);
    if (!mod) {
      Utils.toast(`模块 "${moduleId}" 未找到`);
      Router.go('/');
      return;
    }

    _currentModule = mod;

    $gallery.classList.add('hidden');
    $detailView.classList.add('visible');

    $detailTitle.textContent = `${mod.name}  ·  ${mod.name_en}`;

    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
    const navItem = document.querySelector(`.nav-item[data-module-id="${moduleId}"]`);
    if (navItem) navItem.classList.add('active');

    $detailIframe.src = mod.page ? `/static/pages/${mod.page}` : '';
  }

  function _closeDetail() {
    $gallery.classList.remove('hidden');
    $detailView.classList.remove('visible');
    $detailIframe.src = '';
    _currentModule = null;
    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
  }

  function _showGallery() {
    if (_currentModule) _closeDetail();
  }

  // ================================================================
  //  iframe communication
  // ================================================================

  function _onIframeMessage(e) {
    const { type, payload } = e.data || {};
    if (!type) return;

    switch (type) {
      case 'toast':
        Utils.toast(payload?.message || '', payload?.duration);
        break;
      case 'navigate':
        Router.go('/module/' + (payload?.moduleId || ''));
        break;
    }
  }

  // ================================================================
  //  Helpers
  // ================================================================

  function _scrollSidebarToPhase(phaseId) {
    const header = $sidebarNav.querySelector(`.nav-category[data-phase-id="${phaseId}"]`);
    if (header) {
      header.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }

  function _filterByDifficulty(level) {
    const items = $sidebarNav.querySelectorAll('.nav-item');
    items.forEach(el => el.style.opacity = '1');
    _modules.forEach(m => {
      if (m.difficulty !== level) {
        const el = $sidebarNav.querySelector(`.nav-item[data-module-id="${m.id}"]`);
        if (el) el.style.opacity = '0.35';
      }
    });
    Utils.toast(`已筛选难度 ${level}（点击任意模块恢复）`);
    setTimeout(() => { items.forEach(el => el.style.opacity = '1'); }, 5000);
  }

  function _toggleSearch() {
    const box = document.getElementById('search-box');
    const input = document.getElementById('search-input');
    box.classList.toggle('hidden');
    if (!box.classList.contains('hidden')) {
      input.focus();
    } else {
      input.value = '';
      _onSearch({ target: input });
    }
  }

  function _onSearch(e) {
    const q = (e.target.value || '').toLowerCase().trim();
    const items = $sidebarNav.querySelectorAll('.nav-item');
    items.forEach(el => {
      const mid = el.dataset.moduleId || '';
      const mod = _modules.find(m => m.id === mid);
      if (!mod) { el.style.display = 'none'; return; }
      const match = !q
        || mod.name.toLowerCase().includes(q)
        || mod.name_en.toLowerCase().includes(q)
        || mod.description.toLowerCase().includes(q);
      el.style.display = match ? '' : 'none';
    });
  }

  // ================================================================
  //  Sidebar resizer
  // ================================================================

  function _initSidebarResizer() {
    const resizer = document.getElementById('sidebar-resizer');
    let dragging = false, startX = 0, startW = 0;

    resizer.addEventListener('mousedown', (e) => {
      dragging = true;
      startX = e.clientX;
      startW = $sidebar.getBoundingClientRect().width;
      resizer.classList.add('dragging');
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
    });

    document.addEventListener('mousemove', Utils.throttle((e) => {
      if (!dragging) return;
      const newW = Math.max(180, Math.min(420, startW + e.clientX - startX));
      document.documentElement.style.setProperty('--sidebar-w', newW + 'px');
    }, 16));

    document.addEventListener('mouseup', () => {
      if (!dragging) return;
      dragging = false;
      resizer.classList.remove('dragging');
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    });
  }

  // ---- Boot ----
  window.addEventListener('DOMContentLoaded', init);

  return { getModules: () => _modules };
})();
