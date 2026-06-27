/**
 * CV Metro Map — Application Shell
 * Blueprint strictly mirrors ARCHITECTURE.md §六 (77 规划算法, 5 阶段).
 */
const App = (() => {
  let $metro, $overlay, $ifr, $dname, $den;
  let $searchInput, $searchClear;
  let $statTotal, $statReady;

  // ── BLUEPRINT ──
    const BLUEPRINT = [
    { phase:'phase1', color:'#3B82F6', name:'阶段一 · 基础原语', en:'Fundamentals', sub:'像素级原子操作', diff:1, subs:null, algo:[
      {id:'grayscale',   n:'灰度转换',       en:'Grayscale',         d:'RGB↔灰度亮度图，加权公式'},
      {id:'histogram',   n:'直方图与均衡化', en:'Histogram',         d:'像素统计+CLAHE对比度增强'},
      {id:'threshold',   n:'阈值化',         en:'Thresholding',      d:'全局/自适应/Otsu二值化'},
      {id:'noise',       n:'噪声模型',       en:'Noise Models',      d:'椒盐/高斯噪声生成与分析'},
      {id:'convolution', n:'卷积操作',       en:'Convolution',       d:'1D→2D滑窗，CV的数学心脏'},
      {id:'smoothing',   n:'平滑与去噪',     en:'Smoothing',         d:'高斯/中值/双边统一对比'},
      {id:'sobel',       n:'Sobel梯度',      en:'Sobel Gradient',    d:'一阶导数，梯度幅值与方向'},
      {id:'live',        n:'实时摄像头滤镜', en:'Live Camera',       d:'Webcam实时卷积效果演示'},
    ]},
    { phase:'phase2', color:'#10B981', name:'阶段二 · 经典特征检测', en:'Classical Features', sub:'从像素到结构', diff:2, subs:null, algo:[
      {id:'edge',        n:'Canny边缘检测',  en:'Canny Edge',        d:'五步流水线：高斯→Sobel→NMS→双阈值→滞后'},
      {id:'corner',      n:'Harris角点检测', en:'Harris Corner',     d:'结构张量M→角点响应R，特征值三分法'},
      {id:'sift',        n:'SIFT特征检测',   en:'SIFT',              d:'四阶段：DoG→极值→方向→128D描述子'},
      {id:'morphology',  n:'形态学操作',     en:'Morphology',        d:'腐蚀/膨胀/开/闭，SE结构元素'},
    ]},
    { phase:'phase3', color:'#F59E0B', name:'阶段三 · 中级视觉', en:'Intermediate Vision', sub:'几何验证与特征匹配', diff:3, subs:null, algo:[
      {id:'match',       n:'特征匹配（SIFT+RANSAC）', en:'Feature Matching', d:'L2距离→ratio test→RANSAC单应性，3D重建/SLAM核心'},
    ]},
    { phase:'phase4', color:'#EF4444', name:'阶段四 · 深度学习时代', en:'Deep Learning Era', sub:'从CNN到生成模型', diff:4, subs:[
      {key:'4A', label:'卷积全流程', color:'#F87171', algo:[
        {id:'cnn_basics', n:'CNN基础',        en:'CNN Basics',        d:'Conv→ReLU→Pool→FC四步法'},
        {id:'lenet',      n:'LeNet手写数字',   en:'LeNet',             d:'7层CNN，DL视觉入门第一课'},
        {id:'conv_training',n:'训练过程观察',  en:'Training Watch',    d:'实时观察loss/梯度/激活值变化'},
      ]},
      {key:'4B', label:'检测与分割', color:'#FB923C', algo:[
        {id:'detection',  n:'目标检测',       en:'Object Detection',  d:'Faster R-CNN两阶段+YOLO单阶段'},
        {id:'semantic',   n:'语义分割',       en:'Semantic Seg.',     d:'FCN+U-Net+DeepLabV3逐像素分类'},
        {id:'instance',   n:'实例分割',       en:'Instance Seg.',     d:'Mask R-CNN检测+掩码分支'},
      ]},
      {key:'4C', label:'骨干网络', color:'#FBBF24', algo:[
        {id:'resnet',     n:'ResNet+Grad-CAM',en:'ResNet+Grad-CAM',   d:'残差连接+梯度加权类激活图'},
      ]},
      {key:'4D', label:'生成模型', color:'#C084FC', algo:[
        {id:'gan',        n:'GAN生成对抗',    en:'GAN',               d:'生成器/判别器博弈'},
        {id:'diffusion',  n:'扩散模型基础',   en:'Diffusion Basics',  d:'前向加噪→反向去噪'},
        {id:'ddpm',       n:'DDPM',           en:'DDPM',              d:'扩散奠基作，逐步去噪证明'},
        {id:'sd',         n:'Stable Diffusion',en:'Stable Diffusion', d:'潜空间扩散，VAE+UNet+CrossAttn'},
      ]},
    ]},
    { phase:'phase5', color:'#8B5CF6', name:'阶段五 · 前沿基础模型', en:'Foundation Models', sub:'Transformer统一视觉', diff:5, subs:null, algo:[
      {id:'vit',        n:'Vision Transformer',en:'ViT',            d:'Patch→Transformer，视觉新范式'},
      {id:'detr',       n:'DETR',          en:'DETR',              d:'Transformer端到端，二分匹配替代NMS'},
      {id:'clip',       n:'CLIP',          en:'CLIP',              d:'4亿图文对对比学习，多模态基石'},
      {id:'sam',        n:'SAM',           en:'SAM',               d:'ViT+提示编码+Mask解码，分割基础模型'},
      {id:'nerf',       n:'NeRF',          en:'NeRF',              d:'MLP隐式3D，体渲染开山之作'},
    ]},
  ];;

  function _allAlgos(){ const a=[]; BLUEPRINT.forEach(p=>{if(p.subs)p.subs.forEach(g=>g.algo.forEach(x=>a.push(x)));else p.algo.forEach(x=>a.push(x));}); return a; }
  function _phaseAlgos(ph){ if(ph.subs){ const a=[]; ph.subs.forEach(g=>a.push(...g.algo)); return a; } return ph.algo; }

  let _apiMap = {};
  let _apiModules = [];
  let _current = null;
  // Only count modules with actual algorithm.py (not just skeleton __init__.py)
  // Maps blueprint IDs → actual registered module IDs
  const IDMAP={
    live:'live',conv_training:'conv_training',lenet:'lenet',
    convolution:'convolution',canny:'edge',harris:'corner',edge:'edge',corner:'corner',sift:'sift',
    morphology:'morphology',
    colorspace:'colorspace',grayscale:'grayscale',histogram:'histogram',threshold:'threshold',
    noise:'noise',smoothing:'smoothing',gaussian:'gaussian',sobel:'sobel',median:'median',
    bilateral:'bilateral',
    match:'match',sift_ransac:'match',
    detection:'detection',semantic:'semantic',instance:'instance',
    faster_rcnn:'detection',fcn:'semantic',mask_rcnn:'instance',
    gan:'gan',diffusion:'diffusion',resnet:'resnet',
    vit:'vit',detr:'detr',sam:'sam',clip:'clip',nerf:'nerf',
    sd:'stable_diffusion',ddpm:'ddpm',cnn_basics:'cnn_basics',
  };
  // Auto-detect: IDMAP first, then direct lookup, then fuzzy match
  function _resolveId(id){
    if(IDMAP[id]&&_apiMap[IDMAP[id]]) return IDMAP[id];
    if(_apiMap[id]) return id;
    // Fuzzy: try matching by partial ID or registered name
    for(const rid of Object.keys(_apiMap)){
      if(rid.includes(id)||id.includes(rid)) return rid;
    }
    return id;
  }
  // Modules with verified working backends (tested 28/28)
  const VERIFIED=new Set([
    'colorspace','grayscale','histogram','threshold','noise','convolution','smoothing','gaussian','sobel','median','bilateral','live',
    'edge','corner','sift','morphology',
    'match',
    'cnn_basics','lenet','conv_training','resnet','detection','semantic','instance',
    'gan','diffusion','ddpm','sd',
    'vit','detr','clip','sam','nerf',
  ]);
  const SPECIAL_PAGE_IDS=new Set(['colorspace','edge','corner','sift','match','cnn_basics',
    'morphology','smoothing','median','sobel','histogram','threshold','gaussian','bilateral']);
  function _implInfo(id){
    const m=_apiMap[id];
    if(m && m.implementation) return m.implementation;
    if(window.AlgorithmContent && window.AlgorithmContent[id] && window.AlgorithmContent[id].implementation){
      return window.AlgorithmContent[id].implementation;
    }
    return null;
  }
  function _implRealModel(impl){
    return !!(impl && (impl.realModel===true || impl.real_model===true));
  }
  function _implRunnable(impl){
    if(!impl) return false;
    const local = impl.local_inference !== undefined ? impl.local_inference : impl.localInference;
    return !!(local !== false && impl.category !== 'not_implemented' && impl.category !== 'model_not_available' && impl.category !== 'requires_external_weights');
  }
  function _moduleState(id){
    const rid=_resolveId(id);
    const api=_apiMap[rid]||_apiMap[id];
    const impl=(api&&api.implementation)||_implInfo(rid)||_implInfo(id);
    const hasTeaching=!!(window.AlgorithmContent&&window.AlgorithmContent[id]);
    const hasPage=!!((api&&api.page)||hasTeaching);
    const runnable=_implRunnable(impl);
    const apiReady=!!(api&&api.page&&VERIFIED.has(rid)&&(!impl||runnable));
    const teachingReady=!!(hasTeaching&&runnable);
    const ready=!!(hasPage&&(runnable||apiReady||teachingReady));
    return {id, rid, api, impl, ready, runnable, hasPage, hasTeaching};
  }
  function _isReady(id){
    return _moduleState(id).ready;
  }
  function _openIfReady(id){
    const rid=_resolveId(id);
    const state=_moduleState(id);
    const impl=state.impl;
    const m=_apiMap[rid];
    const a=_allAlgos().find(x=>x.id===id)||_allAlgos().find(x=>x.id===rid)||{id,n:id,en:''};
    if(m&&m.page&&(!impl||_implRunnable(impl))){
      return Object.assign({}, m, {id:id, name:a.n||m.name||id, name_en:a.en||m.name_en||''});
    }
    if(impl && !_implRunnable(impl) && window.AlgorithmContent && window.AlgorithmContent[id]){
      return {id, name:a.n||id, name_en:a.en||'', page:'teaching.html?id='+encodeURIComponent(id)};
    }
    if(impl && !_implRunnable(impl)) return null;
    if(!_isReady(id) && !(window.AlgorithmContent && window.AlgorithmContent[id] && !impl)) return null;
    if(window.AlgorithmContent && window.AlgorithmContent[id] && !SPECIAL_PAGE_IDS.has(id)&&!SPECIAL_PAGE_IDS.has(rid)){
      return {id, name:a.n||id, name_en:a.en||'', page:'teaching.html?id='+encodeURIComponent(id)};
    }
    if(m&&m.page) return m;
    return null;
  }

  // ============================================================
  function init(){
    $metro=document.getElementById('phase-metro');
    $overlay=document.getElementById('detail-overlay');
    $ifr=document.getElementById('detail-iframe');
    $dname=document.getElementById('detail-name');
    $den=document.getElementById('detail-en');
    $searchInput=document.getElementById('search-input');
    $searchClear=document.getElementById('search-clear');
    $statTotal=document.getElementById('stat-total');
    $statReady=document.getElementById('stat-ready');

    document.getElementById('btn-back').addEventListener('click',closeDetail);

    // Theme toggle (iOS-style switch)
    const savedTheme=localStorage.getItem('theme')||'dark';
    document.documentElement.setAttribute('data-theme',savedTheme);
    document.getElementById('theme-toggle').addEventListener('click',()=>{
      const cur=document.documentElement.getAttribute('data-theme');
      const next=cur==='light'?'dark':'light';
      document.documentElement.setAttribute('data-theme',next);
      localStorage.setItem('theme',next);
      // Broadcast to iframe
      try{$ifr.contentWindow.postMessage({type:'theme',theme:next},'*');}catch(e){}
    });

    document.getElementById('btn-network').addEventListener('click',openNetwork);
    document.querySelectorAll('[data-course-open]').forEach(btn=>{
      btn.addEventListener('click',()=>Router.go('/module/'+btn.dataset.courseOpen));
    });
    document.addEventListener('keydown',e=>{if(e.key==='Escape'&&$overlay.classList.contains('active'))closeDetail();});

    // Search box
    $searchInput.addEventListener('input', Utils.debounce(_onSearch, 180));
    $searchClear.addEventListener('click', ()=>{ $searchInput.value=''; $searchClear.classList.add('hidden'); _clearSearch(); });

    window.addEventListener('message',_onIframeMsg);
    _loadAndRender();
    Router.on('/',()=>closeDetail());
    Router.on('/module/:id',params=>openModule(params.id));
  }

  async function _loadAndRender(){
    try{
      const res=await fetch('/api/modules'); const data=await res.json();
      _apiModules=[]; _apiMap={};
      (data.phases||[]).forEach(ph=>{(ph.modules||[]).forEach(m=>{_apiModules.push(m);_apiMap[m.id]=m;});});
    }catch(e){console.warn('[App] API unreachable');}
    _renderMetro();
    _syncCounts();
  }
  function _syncCounts(){
    const all=_allAlgos();
    if($statTotal) $statTotal.textContent=all.length;
    if($statReady) $statReady.textContent=all.filter(a=>_isReady(a.id)).length;
  }

  // ── Topic Cards ──
    const TOPIC_CARDS = [
    // ── Phase 1 ──
    { id:'color-and-contrast', phase:0,
      kicker:'像素启蒙', title:'色彩与对比：让计算机"看见"光线',
      narrative:'计算机看到的是数字矩阵。色彩空间转换赋予颜色以意义，直方图揭示像素分布的统计秘密，阈值化做出第一个"是/否"的二元判断。噪声是 CV 的第一道坎——了解它，才能对抗它。',
      colorFrom:'#3B82F6', colorTo:'#60a5fa',
      cover:'<div class="cv-cover cv-colors"><div class="cv-color-bar"><span style="height:60%;background:#ef4444"></span><span style="height:85%;background:#22c55e"></span><span style="height:45%;background:#3b82f6"></span><span style="height:70%;background:#f59e0b"></span><span style="height:55%;background:#8b5cf6"></span></div><div class="cv-thresh-split"><span></span><span></span></div></div>',
      algos:[{id:'colorspace',role:'空间转换',label:'色彩空间'},{id:'histogram',role:'像素统计',label:'直方图均衡化'},{id:'threshold',role:'二元决策',label:'阈值化'},{id:'noise',role:'了解敌人',label:'噪声模型'}]},
    { id:'filter-and-convolution', phase:0,
      kicker:'数学核心', title:'滤波与卷积：像素的数学魔法',
      narrative:'卷积是 CV 的数学心脏——一个权重窗口在像素矩阵上滑动。高斯用它温柔降噪，Sobel 用它发现梯度方向，中值滤波铁腕去除椒盐噪声，双边滤波聪明地在降噪的同时保留边缘——空间近且颜色近才加权。',
      colorFrom:'#3B82F6', colorTo:'#93c5fd',
      cover:'<div class="cv-cover cv-conv"><div class="cv-kernel-3x3"><i></i><i></i><i></i><i></i><i class="active"></i><i></i><i></i><i></i><i></i></div><div class="cv-scan-lines"><span></span><span></span><span></span></div></div>',
      algos:[{id:'convolution',role:'数学基础',label:'卷积操作'},{id:'smoothing',role:'统一降噪专题',label:'平滑与去噪'},{id:'sobel',role:'发现梯度',label:'Sobel梯度'}]},

    // ── Phase 2 ──
    { id:'edges-to-corners', phase:1,
      kicker:'结构发现', title:'从边缘到角点：几何结构的诞生',
      narrative:'Canny 用五步流水线把梯度精炼成连续边缘，其中非极大值抑制已经作为 Canny 的内部步骤讲解。Harris 在边缘的拐弯处计算结构张量，找到那些"无论往哪移变化都很大"的角点。Shi-Tomasi 改进了判据让跟踪更稳定。',
      colorFrom:'#10B981', colorTo:'#34d399',
      cover:'<div class="cv-cover cv-edge-corner"><div class="cv-edge-line-glow"></div><div class="cv-corner-spark"></div><div class="cv-edge-dots"><i></i><i></i><i></i><i></i></div></div>',
      algos:[{id:'canny',role:'精确边缘',label:'Canny边缘'},{id:'harris',role:'发现角点',label:'Harris角点'}]},
    { id:'descriptors-and-shapes', phase:1,
      kicker:'物体记忆', title:'描述与形状：让计算机"记住"物体',
      narrative:'SIFT 用 DoG 找关键点，128 维描述子不惧旋转缩放。形态学操作像一把精细手术刀——腐蚀去噪、膨胀填洞，让二值结构变得干净可分析。',
      colorFrom:'#10B981', colorTo:'#6ee7b7',
      cover:'<div class="cv-cover cv-descriptors"><div class="cv-desc-patch"></div><div class="cv-hough-circle"></div><div class="cv-contour-blob"></div></div>',
      algos:[{id:'sift',role:'特征描述',label:'SIFT'},{id:'morphology',role:'形状精修',label:'形态学操作'}]},

    // ── Phase 3 ──
    { id:'feature-matching', phase:2,
      kicker:'几何验证', title:'特征匹配：SIFT + RANSAC',
      narrative:'SIFT 找到图像中的关键点并给每个点赋予 128 维描述子。两张图的描述子用 L2 距离做最近邻匹配——但直接匹配会有大量错误。RANSAC 的核心思想很朴素：随机抽最小样本拟合模型（如单应矩阵），然后数有多少其他点也符合这个模型。反复多次，选"内点"最多的那一次。这就是在噪声中找共识。',
      colorFrom:'#F59E0B', colorTo:'#fbbf24',
      cover:'<div class="cv-cover cv-match-cover"><div class="cv-match-pair"><span class="mp-dot mp-a1"></span><span class="mp-dot mp-a2"></span><span class="mp-line mp-l1"></span><span class="mp-line mp-l2"></span><span class="mp-dot mp-b1"></span><span class="mp-dot mp-b2"></span></div></div>',
      algos:[{id:'match',role:'特征匹配与几何验证',label:'SIFT + RANSAC'}]},

    // ── Bridge: AI 之眼 ──
    { id:'ai-eye-bridge', phase:2, isBridge:true,
      kicker:'承上启下 · The AI Eye', title:'AI 之眼：同一张图的三种视角',
      narrative:'Phase 3 的 Watershed、GrabCut、SLIC 教会我们"分割"的直觉。进入 Phase 4，DeepLabV3、Faster R-CNN、Mask R-CNN 将这种直觉推向语义与实例级别。用同一张街景图，看三种算法视角如何层层递进。',
      colorFrom:'#06b6d4', colorTo:'#22d3ee',
      cover:'<div class="cv-cover cv-ai-eye-cover"><div class="cv-eye-stage"><div class="cv-eye-thumb box-mode"><span class="box-a"></span><span class="box-b"></span></div><span class="cv-eye-arr">→</span><div class="cv-eye-thumb wash-mode"><span class="wash-a"></span><span class="wash-b"></span></div><span class="cv-eye-arr">→</span><div class="cv-eye-thumb inst-mode"><span class="inst-a"></span><span class="inst-b"></span></div></div></div>',
      algos:[{id:'watershed',role:'经典分割',label:'分水岭'},{id:'grabcut',role:'图割抠图',label:'GrabCut'},{id:'detection',role:'找→画框',label:'目标检测'},{id:'semantic',role:'像素→类',label:'语义分割'},{id:'instance',role:'同→分开',label:'实例分割'}],
      bridgeBtns:[{label:'目标检测',icon:'🎯',cls:'detection-btn',mid:'detection'},{label:'语义分割',icon:'🎨',cls:'semantic-btn',mid:'semantic'},{label:'实例分割',icon:'✂️',cls:'instance-btn',mid:'instance'}]},

    // ── Phase 4 ──
    { id:'ai-eye-detection-seg', phase:3,
      kicker:'检测与分割', title:'深度学习时代的检测与分割',
      narrative:'Faster R-CNN 用 RPN 生成候选区域——两阶段精准定位。YOLO 把图划分成网格一次性回归——单阶段快如闪电。FCN 开创全卷积像素分类，U-Net 用对称编解码+跳跃连接精细还原边界。Mask R-CNN 加掩码分支把同类物体一个一个分开。',
      colorFrom:'#EF4444', colorTo:'#f87171',
      cover:'<div class="cv-cover cv-dl-detect"><div class="cv-grid-overlay"><span></span><span></span><span></span></div><div class="cv-mask-shape"></div></div>',
      algos:[{id:'faster_rcnn',role:'两阶段',label:'Faster R-CNN'},{id:'yolo',role:'单阶段',label:'YOLO'},{id:'fcn',role:'全卷积',label:'FCN'},{id:'unet',role:'编解码',label:'U-Net'},{id:'mask_rcnn',role:'实例',label:'Mask R-CNN'}]},
    { id:'cnn-foundations', phase:3,
      kicker:'深度基石', title:'CNN 基石：深度学习的视觉根基',
      narrative:'Conv→ReLU→Pool→FC——CNN 的四步法，从 LeNet 时代就确立了。ResNet 用残差连接（跳跃相加）让 152 层网络成为可能，解决了"越深越差"的退化问题。Grad-CAM 用梯度加权类激活图可视化"网络在看哪里"。',
      colorFrom:'#EF4444', colorTo:'#fb7185',
      cover:'<div class="cv-cover cv-cnn-stack"><div class="cv-layer-stack"><span class="l1">Conv</span><span class="l2">ReLU</span><span class="l3">Pool</span><span class="l4">FC</span></div><div class="cv-skip-conn"></div></div>',
      algos:[{id:'cnn_basics',role:'CNN四步',label:'CNN基础'},{id:'resnet',role:'残差连接',label:'ResNet'}]},
    { id:'generative-models', phase:3,
      kicker:'生成模型', title:'AI 的想象力：从博弈到扩散',
      narrative:'GAN 让两个网络互相博弈——生成器造假，判别器打假，在对抗中学会"画"出逼真图像。扩散模型反其道而行：先学会把图像一步步加噪成一团混沌，再学会一步步去噪还原——结果比 GAN 更稳定、更多样。',
      colorFrom:'#EF4444', colorTo:'#c084fc',
      cover:'<div class="cv-cover cv-gen-duel"><div class="cv-gan-side"><span class="g">G</span><span class="d">D</span></div><div class="cv-diff-steps"><i></i><i></i><i></i><i></i><i></i></div></div>',
      algos:[{id:'gan',role:'对抗博弈',label:'GAN'},{id:'diffusion',role:'去噪生成',label:'扩散模型'}]},

    // ── Phase 5 ──
    { id:'transformer-vision', phase:4,
      kicker:'新范式', title:'Transformer 统一视觉：从分类到检测',
      narrative:'ViT 做了一个大胆的实验——把图像切成 Patch，像单词一样丢进 Transformer，不需要卷积也能赢。DETR 更进一步，把目标检测重新定义为集合预测问题——100 个 Object Query 经过 Transformer 直接输出 (类别, 框)，匈牙利算法做二分匹配替代 NMS。分类和检测都被 Transformer 统一了。',
      colorFrom:'#8B5CF6', colorTo:'#a78bfa',
      cover:'<div class="cv-cover cv-vit-patches"><div class="cv-patch-grid"><i></i><i></i><i></i><i class="active"></i><i></i><i></i><i></i><i></i><i class="active"></i></div><div class="cv-tf-block"><span></span><span></span></div></div>',
      algos:[{id:'vit',role:'Patch→TF',label:'ViT'},{id:'detr',role:'检测TF',label:'DETR'}]},
    { id:'foundation-models', phase:4,
      kicker:'基础模型', title:'分割与多模态：基础模型时代',
      narrative:'CLIP 用 4 亿图文对做对比学习，把图像和文本映射到同一向量空间——学会了零样本"看图说话"。SAM 把分割变成了"指点江山"——点一下、画个框，哪里都能分割，同一个模型可分割任意物体。NeRF 用一个 MLP 隐式记住整个 3D 场景——给一个视角，还你一张逼真渲染图。',
      colorFrom:'#8B5CF6', colorTo:'#f9a8d4',
      cover:'<div class="cv-cover cv-sam-clip"><div class="cv-sam-point"></div><div class="cv-sam-mask"></div><div class="cv-nerf-sphere" style="width:24px;height:24px;margin:0 0 0 20px"><span class="ray"></span><span class="ray"></span></div></div>',
      algos:[{id:'clip',role:'图文对齐',label:'CLIP'},{id:'sam',role:'提示分割',label:'SAM'},{id:'nerf',role:'隐式3D',label:'NeRF'}]},
  ];


  function _renderTopicCard(container, meta){
    const card=document.createElement('div');
    card.className='topic-card'+(meta.isBridge?' bridge-highlight':'');
    card.id='topic-'+meta.id;

    var visualHTML = meta.cover || ('<div class="vis-gradient-bar" style="background:linear-gradient(90deg,'+meta.colorFrom+','+meta.colorTo+')"></div>');

    let actionsHTML='';
    if(meta.bridgeBtns){
      actionsHTML='<div class="bridge-actions">'+meta.bridgeBtns.map(function(b){return '<button class="bridge-open '+b.cls+'" data-course-open="'+b.mid+'"><span>'+b.icon+'</span> '+b.label+'</button>';}).join('')+'</div>';
    }else{
      actionsHTML='<div class="bridge-algo-tags">'+meta.algos.map(function(a){return '<button class="bridge-algo-tag" data-mid="'+a.id+'"><span class="bat-role">'+a.role+'</span><span class="bat-name">'+a.label+'</span></button>';}).join('')+'</div>';
    }

    card.innerHTML='<div class="bridge-body topic-'+meta.id+'"><div class="topic-visual-wrap">'+visualHTML+'</div><div class="bridge-copy"><span class="topic-card-kicker" style="color:'+meta.colorFrom+'">'+meta.kicker+'</span><h3>'+meta.title+'</h3><p>'+meta.narrative+'</p>'+actionsHTML+'</div></div>';
    card.querySelectorAll('.bridge-algo-tag').forEach(function(btn){btn.addEventListener('click',function(){Router.go('/module/'+btn.dataset.mid);});});
    card.querySelectorAll('.bridge-open').forEach(function(btn){btn.addEventListener('click',function(){Router.go('/module/'+btn.dataset.courseOpen);});});
    container.appendChild(card);
  }

  // ── Metro ──
  function _renderMetro(){
    $metro.innerHTML='';
    BLUEPRINT.forEach(function(phase, idx){
      var line=document.createElement('div');
      line.className='phase-line topic-rail';
      var algos=_phaseAlgos(phase);
      var ready=algos.filter(function(a){return _isReady(a.id);}).length;
      line.innerHTML='<div class="phase-line-header"><span class="station-dot" style="background:'+phase.color+';box-shadow:0 0 10px '+phase.color+'"></span><span class="phase-label">'+phase.name+' / '+phase.en+'</span><span class="phase-sub">'+phase.sub+'</span><span class="phase-count">'+ready+'/'+algos.length+' ready</span></div><div class="topic-row"></div>';
      $metro.appendChild(line);
      var row=line.querySelector('.topic-row');
      TOPIC_CARDS.filter(function(c){return c.phase===idx;}).forEach(function(c){_renderTopicCard(row,c);});
    });
  }

  // ── Detail ──
  function openModule(id){
    const m=_openIfReady(id); if(!m){Utils.toast('模块 "'+id+'" 尚未实现');return;}
    _current=m; $dname.textContent=m.name; $den.textContent=m.name_en||'';
    if(m.page){
      const sep=m.page.indexOf('?')>=0?'&':'?';
      $ifr.src='/static/pages/'+m.page+sep+'v='+Date.now();
    }else{
      $ifr.src='';
    }
    $overlay.classList.add('active');
  }
  function closeDetail(){$overlay.classList.remove('active');$ifr.src='';_current=null;}
  function openNetwork(){
    $dname.textContent='算法关系网络';$den.textContent='Algorithm Relationship Graph';
    $ifr.src='/static/algorithm_network.html';$overlay.classList.add('active');
    const t=document.documentElement.getAttribute('data-theme')||'dark';
    $ifr.onload=()=>{try{$ifr.contentWindow.postMessage({type:'theme',theme:t},'*');}catch(e){}};
  }
  // ── Search (inline) ──
  function _onSearch(e){
    const q=(e.target.value||'').toLowerCase().trim();
    if(q){$searchClear.classList.remove('hidden');}else{$searchClear.classList.add('hidden');_clearSearch();return;}
    const hits=_allAlgos().filter(a=>a.n.toLowerCase().includes(q)||a.en.toLowerCase().includes(q)||a.d.toLowerCase().includes(q));
    document.querySelectorAll('.topic-card').forEach(card=>{
      const ids=Array.from(card.querySelectorAll('[data-mid],[data-course-open]')).map(btn=>btn.dataset.mid||btn.dataset.courseOpen);
      card.classList.toggle('hidden',!ids.some(id=>hits.some(h=>h.id===id||_resolveId(h.id)===_resolveId(id))));
    });
    document.querySelectorAll('.topic-rail').forEach(rail=>{
      rail.classList.toggle('hidden',!rail.querySelector('.topic-card:not(.hidden)'));
    });
    if(hits.length===1) Router.go('/module/'+hits[0].id);
  }
  function _clearSearch(){
    document.querySelectorAll('.topic-card.hidden,.topic-rail.hidden').forEach(c=>c.classList.remove('hidden'));
  }

  // ── Iframe messages ──
  function _onIframeMsg(e){const{type,payload}=e.data||{};if(!type)return;
    switch(type){case'toast':Utils.toast(payload?.message||'',payload?.duration);break;
      case'navigate':Router.go('/module/'+(payload?.moduleId||''));break;
      case'closeDetail':closeDetail();break;}
  }

  window.addEventListener('DOMContentLoaded',init);
  return{openModule,closeDetail,getModules:()=>_apiModules};
})();
