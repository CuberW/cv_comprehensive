/**
 * 主应用逻辑。
 * 管理侧边栏导航、画廊首页、模块详情视图。
 * 通过 postMessage 与 iframe 内的算法页面通信。
 */

const App = (() => {
  // ---- DOM 引用（在 DOMContentLoaded 后初始化） ----
  let $sidebar, $sidebarNav, $gallery, $detailView, $detailIframe, $detailTitle;
  let $categoryGrid, $ladderSteps;

  // ---- 状态 ----
  let _modules = [];       // 所有模块元信息（从 API 获取）
  let _currentModule = null;

  // ================================================================
  //  初始化
  // ================================================================

  function init() {
    // 缓存 DOM 引用
    $sidebar = document.getElementById('sidebar');
    $sidebarNav = document.getElementById('sidebar-nav');
    $gallery = document.getElementById('gallery');
    $detailView = document.getElementById('detail-view');
    $detailIframe = document.getElementById('detail-iframe');
    $detailTitle = document.getElementById('detail-title');
    $categoryGrid = document.getElementById('category-grid');
    $ladderSteps = document.getElementById('ladder-steps');

    // 绑定事件
    document.getElementById('btn-back').addEventListener('click', _closeDetail);
    document.getElementById('btn-search').addEventListener('click', _toggleSearch);
    document.getElementById('search-input').addEventListener('input', Utils.debounce(_onSearch, 200));

    // 难度阶梯点击
    document.querySelectorAll('.ladder-step').forEach(el => {
      el.addEventListener('click', () => _filterByDifficulty(parseInt(el.dataset.level)));
    });

    // 侧边栏拖拽
    _initSidebarResizer();

    // 加载模块数据
    _fetchModules();

    // 注册路由
    Router.on('/', () => _showGallery());
    Router.on('/module/:id', (params) => _openModule(params.id));

    // 监听 iframe 消息
    window.addEventListener('message', _onIframeMessage);
  }

  // ================================================================
  //  数据获取
  // ================================================================

  async function _fetchModules() {
    try {
      const res = await fetch('/api/modules');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      // 防御：检查响应结构
      Utils.assertFields(data, ['categories', 'total'], '/api/modules 响应');

      // 展开为扁平的模块列表
      _modules = [];
      for (const [cat, mods] of Object.entries(data.categories)) {
        mods.forEach(m => _modules.push({ ...m, category: cat }));
      }

      _renderSidebar(data.categories);
      _renderGallery(data.categories);
      _renderDifficultyLadder();

    } catch (err) {
      console.error('[App] 加载模块列表失败:', err);
      Utils.toast('加载模块列表失败，请检查后端是否启动');
    }
  }

  // ================================================================
  //  侧边栏导航渲染
  // ================================================================

  function _renderSidebar(categories) {
    $sidebarNav.innerHTML = '';

    // 分类名 → emoji 映射
    const catMeta = {
      'seeing':        { emoji: '👁️', label: '看见 — 基础视觉' },
      'understanding': { emoji: '🧠', label: '理解 — 语义分析' },
      'transforming':  { emoji: '🔄', label: '变换 — 域与空间' },
      'generating':    { emoji: '🎨', label: '生成 — 创造图像' },
      'other':         { emoji: '📦', label: '其他' },
    };

    for (const [cat, mods] of Object.entries(categories)) {
      const meta = catMeta[cat] || catMeta['other'];

      // 分类标题
      const catHeader = document.createElement('div');
      catHeader.className = 'nav-category';
      catHeader.textContent = `${meta.emoji}  ${meta.label}`;
      $sidebarNav.appendChild(catHeader);

      // 模块列表
      mods.forEach(m => {
        const btn = document.createElement('button');
        btn.className = 'nav-item';
        btn.dataset.moduleId = m.id;
        btn.innerHTML = `
          <span>${m.name}</span>
          <span class="nav-difficulty">${'★'.repeat(m.difficulty)}</span>
        `;
        btn.addEventListener('click', () => Router.go('/module/' + m.id));
        $sidebarNav.appendChild(btn);
      });
    }
  }

  // ================================================================
  //  画廊首页渲染
  // ================================================================

  function _renderGallery(categories) {
    // 分类元信息
    const catMeta = {
      'seeing':        { emoji: '👁️', name: '看见', desc: '从像素中提取信息：边缘、角点、特征。CV 的第一步。' },
      'understanding': { emoji: '🧠', name: '理解', desc: '让机器看懂图像内容：检测、分割、分类。' },
      'transforming':  { emoji: '🔄', name: '变换', desc: '换个角度看图像：频域、色彩空间、几何变换。' },
      'generating':    { emoji: '🎨', name: '生成', desc: '从噪声中创造图像：扩散模型、GAN。' },
      'other':         { emoji: '📦', name: '其他', desc: '更多计算机视觉算法。' },
    };

    $categoryGrid.innerHTML = '';

    for (const [cat, mods] of Object.entries(categories)) {
      const meta = catMeta[cat] || catMeta['other'];

      const card = document.createElement('div');
      card.className = 'category-card';
      card.innerHTML = `
        <div class="cat-emoji">${meta.emoji}</div>
        <div class="cat-name">${meta.name}</div>
        <div class="cat-count">${mods.length} 个算法模块</div>
        <div class="cat-desc">${meta.desc}</div>
        <div class="category-modules">
          ${mods.map(m => `<span class="module-chip" data-module-id="${m.id}">${m.name}</span>`).join('')}
        </div>
      `;

      // 点击分类卡片 → 高亮该分类在侧边栏
      card.addEventListener('click', (e) => {
        // 如果点击的是具体模块 chip，直接打开
        if (e.target.classList.contains('module-chip')) {
          Router.go('/module/' + e.target.dataset.moduleId);
          return;
        }
        // 否则滚动侧边栏到该分类
        _scrollSidebarToCategory(cat);
      });

      $categoryGrid.appendChild(card);
    }
  }

  function _renderDifficultyLadder() {
    // 难度阶梯已在 HTML 中定义，此处可动态更新计数
    const counts = { 1: 0, 2: 0, 3: 0, 4: 0 };
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
  //  模块详情
  // ================================================================

  function _openModule(moduleId) {
    const mod = _modules.find(m => m.id === moduleId);
    if (!mod) {
      Utils.toast(`模块 "${moduleId}" 不存在`);
      Router.go('/');
      return;
    }

    _currentModule = mod;

    // 切换视图
    $gallery.classList.add('hidden');
    $detailView.classList.add('visible');

    // 标题
    $detailTitle.textContent = `${mod.name}  ·  ${mod.name_en}`;

    // 高亮侧边栏
    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
    const navItem = document.querySelector(`.nav-item[data-module-id="${moduleId}"]`);
    if (navItem) navItem.classList.add('active');

    // 加载 iframe
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
  //  iframe 通信
  // ================================================================

  /**
   * 接收算法子页面的消息。
   * 子页面通过 parent.postMessage({ type, payload }, '*') 发送。
   */
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
      default:
        break;
    }
  }

  // ================================================================
  //  辅助
  // ================================================================

  function _scrollSidebarToCategory(cat) {
    // 简单实现：找到该分类的第一个模块并滚动到可见
    const firstItem = $sidebarNav.querySelector(`.nav-item`);
    // 实际实现可根据 cat header 定位
    const headers = $sidebarNav.querySelectorAll('.nav-category');
    // 遍历找到匹配的分类标题
    const catMeta = {
      'seeing': '看见', 'understanding': '理解',
      'transforming': '变换', 'generating': '生成', 'other': '其他'
    };
    const label = catMeta[cat] || '';
    for (const h of headers) {
      if (h.textContent.includes(label)) {
        h.scrollIntoView({ behavior: 'smooth', block: 'start' });
        break;
      }
    }
  }

  function _filterByDifficulty(level) {
    // 简单实现：滚动侧边栏并闪烁对应难度的模块
    const items = $sidebarNav.querySelectorAll('.nav-item');
    items.forEach(el => el.style.opacity = '1');
    // 非目标难度的模块降低透明度
    _modules.forEach(m => {
      if (m.difficulty !== level) {
        const el = $sidebarNav.querySelector(`.nav-item[data-module-id="${m.id}"]`);
        if (el) el.style.opacity = '0.35';
      }
    });
    Utils.toast(`已筛选 ★ 难度=${level} 的模块（点击任意模块可恢复）`);
    // 点击任意导航项时恢复
    setTimeout(() => {
      items.forEach(el => el.style.opacity = '1');
    }, 5000);
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
  //  侧边栏拖拽
  // ================================================================

  function _initSidebarResizer() {
    const resizer = document.getElementById('sidebar-resizer');
    let dragging = false;
    let startX = 0;
    let startW = 0;

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

  // ---- 启动 ----
  window.addEventListener('DOMContentLoaded', init);

  // 暴露给外部（调试用）
  return { getModules: () => _modules };
})();
