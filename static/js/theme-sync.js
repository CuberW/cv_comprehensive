// Sync algorithm subpages with the main shell theme.
(function(){
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
  }

  applyTheme(parentTheme());

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
