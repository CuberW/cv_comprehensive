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
})();
