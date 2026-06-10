/**
 * 轻量级 Hash 路由器。
 * 监听 hashchange 事件，将不同 hash 映射到对应的处理函数。
 *
 * 路由规则：
 *   #/              → 画廊首页
 *   #/module/<id>   → 打开某算法模块
 *   #/compare       → 对比模式（预留）
 */

const Router = (() => {
  const _routes = new Map();
  let _currentHash = '';

  /**
   * 注册一条路由。
   * @param {string} pattern - 支持 ':param' 动态段，如 '/module/:id'
   * @param {Function} handler - 接收 params 对象
   */
  function on(pattern, handler) {
    Utils.assertType(pattern, 'string', 'route pattern');
    Utils.assertType(handler, 'function', 'route handler');
    _routes.set(pattern, { handler, regex: _patternToRegex(pattern) });
  }

  /**
   * 导航到指定 hash。
   * @param {string} hash - 如 '/module/grayscale'
   */
  function go(hash) {
    if (!hash.startsWith('#')) hash = '#' + hash;
    if (!hash.startsWith('#/')) hash = '#/';
    window.location.hash = hash;
  }

  /**
   * 获取当前路径（不含 #）。
   */
  function currentPath() {
    const h = window.location.hash || '#/';
    return h.replace(/^#/, '') || '/';
  }

  /**
   * 将 '/module/:id' 转为匹配用的正则。
   */
  function _patternToRegex(pattern) {
    const escaped = pattern
      .replace(/[.+?^${}()|[\]\\]/g, '\\$&')
      .replace(/:(\w+)/g, '(?<$1>[^/]+)');
    return new RegExp('^' + escaped + '$');
  }

  /**
   * 匹配当前 hash 并调用对应 handler。
   */
  function _dispatch() {
    const path = currentPath();
    if (path === _currentHash) return; // 避免重复处理
    _currentHash = path;

    for (const [pattern, { handler, regex }] of _routes) {
      const m = path.match(regex);
      if (m) {
        handler(m.groups || {});
        return;
      }
    }

    // 无匹配 → 回首页
    console.warn('[Router] 未匹配的路由:', path, '→ 返回首页');
    go('/');
  }

  // 监听浏览器 hash 变化
  window.addEventListener('hashchange', _dispatch);
  // 页面加载时触发一次
  window.addEventListener('DOMContentLoaded', _dispatch);

  return { on, go, currentPath, dispatch: _dispatch };
})();
