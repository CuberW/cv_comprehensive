// Sync theme with parent page (main site)
(function(){
  function sync(){
    try{
      var p=window.parent.document.documentElement;
      var t=p.getAttribute('data-theme')||'dark';
      document.documentElement.setAttribute('data-theme',t);
    }catch(e){}
  }
  sync();
  window.addEventListener('message',function(e){
    if(e.data&&e.data.type==='theme'){
      document.documentElement.setAttribute('data-theme',e.data.theme||'dark');
    }
  });
})();
