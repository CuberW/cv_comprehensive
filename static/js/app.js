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
      {id:'colorspace',  n:'色彩空间转换',   en:'Color Space Conv.', d:'RGB↔HSV↔Lab↔灰度'},
      {id:'histogram',   n:'直方图与均衡化', en:'Histogram',         d:'像素统计+CLAHE对比度增强'},
      {id:'threshold',   n:'阈值化',         en:'Thresholding',      d:'全局/自适应/Otsu二值化'},
      {id:'noise',       n:'噪声模型',       en:'Noise Models',      d:'椒盐/高斯噪声生成'},
      {id:'convolution', n:'卷积操作',       en:'Convolution',       d:'1D→2D滑窗,整个CV的数学核心'},
      {id:'gaussian',    n:'高斯模糊',       en:'Gaussian Blur',     d:'高斯核+σ与窗口,尺度空间基石'},
      {id:'sobel',       n:'Sobel梯度',      en:'Sobel Gradient',    d:'一阶导数,梯度幅值与方向'},
      {id:'median',      n:'中值滤波',       en:'Median Filter',     d:'非线性去噪,椒盐杀手'},
      {id:'bilateral',   n:'双边滤波',       en:'Bilateral Filter',  d:'保边平滑,空间+色彩双核'},
    ]},
    { phase:'phase2', color:'#10B981', name:'阶段二 · 经典特征检测', en:'Classical Features', sub:'从像素到结构', diff:2, subs:null, algo:[
      {id:'canny',       n:'Canny边缘检测',  en:'Canny Edge',        d:'五步流水线:高斯→Sobel→NMS→双阈值→滞后'},
      {id:'harris',      n:'Harris角点检测', en:'Harris Corner',     d:'结构张量M→角点响应R,特征值三分法'},
      {id:'shitomasi',   n:'Shi-Tomasi角点', en:'Shi-Tomasi',        d:'R=min(λ₁,λ₂),光流追踪首选', planned:true},
      {id:'sift',        n:'SIFT特征',       en:'SIFT',              d:'四阶段:DoG→极值→方向→128D描述子'},
      {id:'tpl_match',   n:'模板匹配',       en:'Template Match',    d:'CCORR/NCC滑窗,多目标Quickselect'},
      {id:'hough',       n:'Hough变换',      en:'Hough Transform',   d:'(ρ,θ)/(x,y,r)参数空间投票'},
      {id:'morphology',  n:'形态学操作',     en:'Morphology',        d:'腐蚀/膨胀/开/闭,SE结构元素'},
      {id:'contour',     n:'轮廓查找',       en:'Contour Finding',   d:'层级树,面积/周长/凸包分析'},
      {id:'nms',         n:'非极大值抑制',   en:'NMS',               d:'边缘→角点→检测框的通用后处理'},
    ]},
    { phase:'phase3', color:'#F59E0B', name:'阶段三 · 中级视觉', en:'Intermediate Vision', sub:'', diff:3, subs:[
      {key:'3A', label:'图像分割', color:'#F59E0B', algo:[
        {id:'kmeans',    n:'K-Means分割',   en:'K-Means Seg',      d:'RGB像素聚类,无监督分割入口'},
        {id:'ncuts',     n:'Normalized Cuts',en:'Normalized Cuts',  d:'谱聚类:拉普拉斯→Fiedler向量', planned:true},
        {id:'watershed', n:'分水岭分割',     en:'Watershed',        d:'梯度图→地形浸没+标记控制'},
        {id:'grabcut',   n:'GrabCut',        en:'GrabCut',          d:'GraphCut+迭代GMM交互式抠图'},
        {id:'slic',      n:'SLIC超像素',     en:'SLIC Superpixel',  d:'Lab+xy五维K-means聚类'},
      ]},
      {key:'3B', label:'传统识别管线', color:'#FBBF24', algo:[
        {id:'hog_svm',   n:'HOG+SVM',       en:'HOG+SVM',          d:'梯度直方图+LinearSVM行人检测'},
        {id:'bovw_spm',  n:'BoVW+SPM',      en:'BoVW+SPM',         d:'SIFT→K-Means词汇→SPM→Chi2SVM', planned:true},
        {id:'sift_ransac',n:'SIFT+RANSAC',  en:'SIFT+RANSAC',      d:'L2距离→ratio test→RANSAC单应性'},
      ]},
      {key:'3C', label:'运动估计', color:'#FB923C', algo:[
        {id:'optical_flow',n:'光流',        en:'Optical Flow',      d:'LK稀疏+Farneback稠密+金字塔'},
      ]},
      {key:'3D', label:'几何视觉', color:'#EF4444', algo:[
        {id:'calibration',n:'相机标定',     en:'Camera Calibration',d:'针孔→K[R|t]→畸变→DLT→Cholesky', planned:true},
        {id:'epipolar',  n:'对极几何',      en:'Epipolar Geometry', d:'F/E矩阵→8点法→SVD恢复R,t', planned:true},
        {id:'stereo',    n:'立体匹配',       en:'Stereo Matching',   d:'SAD/SSD→视差图→深度Z=fB/d'},
        {id:'sfm',       n:'三角测量与SfM', en:'SfM',               d:'P₀,P₁→SVD→稀疏点云', planned:true},
        {id:'stitching', n:'图像拼接',       en:'Image Stitching',   d:'单应性+warpPerspective+融合'},
      ]},
    ]},
    { phase:'phase4', color:'#EF4444', name:'阶段四 · 深度学习时代', en:'Deep Learning Era', sub:'', diff:4, subs:[
      {key:'4A', label:'基础架构', color:'#F87171', algo:[
        {id:'cnn_basics',n:'CNN基础',       en:'CNN Basics',        d:'Conv→ReLU→Pool→FC,滤波器可视化', planned:true},
        {id:'resnet',    n:'ResNet+Grad-CAM',en:'ResNet+Grad-CAM',  d:'残差连接+Grad-CAM决策可视化', planned:true},
      ]},
      {key:'4B', label:'语义分割', color:'#FB923C', algo:[
        {id:'fcn',       n:'FCN',           en:'FCN',               d:'全卷积+转置卷积+跳跃融合,开山之作', planned:true},
        {id:'unet',      n:'U-Net',         en:'U-Net',             d:'对称编解码+长跳跃连接', planned:true},
      ]},
      {key:'4C', label:'目标检测', color:'#FBBF24', algo:[
        {id:'faster_rcnn',n:'Faster R-CNN+FPN',en:'Faster R-CNN',  d:'两阶段:RPN+RoIPool+FPN', planned:true},
        {id:'yolo',      n:'YOLO',          en:'YOLO',              d:'单阶段:网格→回归(x,y,w,h,class)', planned:true},
      ]},
      {key:'4D', label:'实例分割', color:'#A3E635', algo:[
        {id:'mask_rcnn', n:'Mask R-CNN',    en:'Mask R-CNN',        d:'FasterRCNN+掩码分支+RoIAlign', planned:true},
      ]},
      {key:'4E', label:'生成模型', color:'#C084FC', algo:[
        {id:'gan',       n:'GAN(基础)',     en:'GAN Basics',        d:'生成器/判别器博弈,模式坍塌'},
        {id:'diffusion', n:'扩散模型基础',  en:'Diffusion Basics',  d:'前向加噪→反向去噪,与GAN对比'},
      ]},
    ]},
    { phase:'phase5', color:'#8B5CF6', name:'阶段五 · 基础模型与前沿感知', en:'Foundation Models & Perception', sub:'2020-2025', diff:5, subs:[
      {key:'5.1', label:'视觉骨干网络 (4)', color:'#A78BFA', algo:[
        {id:'vit',       n:'ViT',           en:'Vision Transformer',d:'Patch→Transformer,CNN替代范式'},
        {id:'swin',      n:'Swin Transformer',en:'Swin Transformer',d:'窗口注意力+移位窗口,多尺度ViT', planned:true},
        {id:'dino',      n:'DINO/DINOv2',   en:'DINO/DINOv2',       d:'自监督ViT,注意力自动涌现分割', planned:true},
        {id:'mae',       n:'MAE',           en:'MAE',               d:'遮盖75%→重建,SAM的预训练方法', planned:true},
      ]},
      {key:'5.2', label:'现代检测范式 (3)', color:'#C084FC', algo:[
        {id:'detr',      n:'DETR',          en:'DETR',              d:'Transformer端到端,Object Query+二分匹配'},
        {id:'dino_det',  n:'DINO(检测)',    en:'DINO Detection',    d:'DETR+对比去噪+混合Query', planned:true},
        {id:'grdino',    n:'Grounding DINO',en:'Grounding DINO',    d:'开放词汇检测,文本条件Query', planned:true},
      ]},
      {key:'5.3', label:'通用分割 (3)', color:'#E879F9', algo:[
        {id:'mask2former',n:'Mask2Former',  en:'Mask2Former',       d:'掩码注意力统一语义/实例/全景', planned:true},
        {id:'sam',       n:'SAM',           en:'SAM',               d:'ViT+提示编码+Mask解码,分割基础模型'},
        {id:'sam2',      n:'SAM 2',         en:'SAM 2',             d:'扩展到视频,记忆注意力+时间传播', planned:true},
      ]},
      {key:'5.4', label:'多模态视觉-语言 (2)', color:'#F0ABFC', algo:[
        {id:'clip',      n:'CLIP',          en:'CLIP',              d:'4亿图文对对比学习,双塔对齐'},
        {id:'blip2',     n:'BLIP-2',        en:'BLIP-2',            d:'Q-Former桥接冻结ViT+LLM', planned:true},
      ]},
      {key:'5.5', label:'生成式模型 (6)', color:'#F9A8D4', algo:[
        {id:'ddpm',      n:'DDPM',          en:'DDPM',              d:'扩散奠基作,加噪→去噪逐步生成', planned:true},
        {id:'sd',        n:'Stable Diffusion',en:'Stable Diffusion',d:'潜空间扩散,VAE+UNet+Cross-Attn'},
        {id:'controlnet',n:'ControlNet',    en:'ControlNet',        d:'零卷积+冻结SD,空间控制信号', planned:true},
        {id:'dit',       n:'DiT',           en:'DiT',               d:'Transformer替代UNet做扩散', planned:true},
        {id:'flux',      n:'Flux',          en:'Flux',              d:'DiT+Flow Matching,直线路径', planned:true},
        {id:'stylegan',  n:'StyleGAN',      en:'StyleGAN',          d:'风格映射+AdaIN→解调→抗混叠', planned:true},
      ]},
      {key:'5.6', label:'自监督学习 (4)', color:'#FDA4AF', algo:[
        {id:'simclr',    n:'SimCLR',        en:'SimCLR',            d:'大batch+强增强+MLP投影,NT-Xent', planned:true},
        {id:'moco',      n:'MoCo',          en:'MoCo',              d:'动量编码器+动态队列,解耦batch', planned:true},
        {id:'byol',      n:'BYOL',          en:'BYOL',              d:'无负样本,Online→Target(EMA)', planned:true},
        {id:'ijepa',     n:'I-JEPA',        en:'I-JEPA',            d:'表征空间预测,非像素重建', planned:true},
      ]},
      {key:'5.7', label:'神经3D与空间感知 (7)', color:'#FDBA74', algo:[
        {id:'nerf',      n:'NeRF',          en:'NeRF',              d:'MLP隐式3D,采样→编码→体渲染'},
        {id:'3dgs',      n:'3D Gaussian Splat.',en:'3D Gaussian Splat.',d:'显式高斯椭球→可微光栅化,实时', planned:true},
        {id:'dust3r',    n:'DUSt3R',        en:'DUSt3R',            d:'无需相机参数,Transformer→3D点云', planned:true},
        {id:'pointnet',  n:'PointNet',      en:'PointNet',          d:'MaxPool对称函数+T-Net,点云DL入口', planned:true},
        {id:'orbslam3',  n:'ORB-SLAM3',     en:'ORB-SLAM3',         d:'三线程:跟踪+建图+回环,空间计算基石', planned:true},
        {id:'bev',       n:'BEV Perception',en:'BEV Perception',    d:'Lift-Splat,环视→鸟瞰图特征', planned:true},
        {id:'occupy',    n:'Occupancy Net.', en:'Occupancy Networks',d:'体素占据预测,比3D框更精细', planned:true},
      ]},
      {key:'5.8', label:'时序理解 (3)', color:'#FCD34D', algo:[
        {id:'c3d',       n:'C3D',           en:'C3D',               d:'3D卷积(时空核),动作识别入门', planned:true},
        {id:'bytetrack', n:'ByteTrack',     en:'ByteTrack',         d:'低分检测框+两阶段关联MOT', planned:true},
        {id:'botsort',   n:'BoT-SORT',      en:'BoT-SORT',          d:'ByteTrack+CMC+ReID', planned:true},
      ]},
      {key:'5.9', label:'人体姿态 (4)', color:'#FDE68A', algo:[
        {id:'deeppose',  n:'DeepPose',      en:'DeepPose',          d:'CNN直接回归(x,y),DL姿态起点', planned:true},
        {id:'openpose',  n:'OpenPose',      en:'OpenPose',          d:'PAF+热力图→二分图匹配,自底向上', planned:true},
        {id:'mediapipe', n:'MediaPipe Pose',en:'MediaPipe Pose',    d:'MobileNetV3,33关键点,30+FPS', planned:true},
        {id:'vitpose',   n:'ViTPose',       en:'ViTPose',           d:'纯ViT→直接输出关键点热力图', planned:true},
      ]},
    ]},
  ];

  function _allAlgos(){ const a=[]; BLUEPRINT.forEach(p=>{if(p.subs)p.subs.forEach(g=>g.algo.forEach(x=>a.push(x)));else p.algo.forEach(x=>a.push(x));}); return a; }
  function _phaseAlgos(ph){ if(ph.subs){ const a=[]; ph.subs.forEach(g=>a.push(...g.algo)); return a; } return ph.algo; }

  let _apiMap = {};
  let _apiModules = [];
  let _current = null;
  // Only count modules with actual algorithm.py (not just skeleton __init__.py)
  // Maps blueprint IDs → actual registered module IDs
  const IDMAP={
    colorspace:'grayscale',canny:'edge',harris:'corner',sift_ransac:'match',
    stitching:'match',convolution:'convolution',sift:'sift',hough:'hough',
    morphology:'morphology',contour:'contour',grabcut:'grabcut',slic:'slic',
    watershed:'watershed',hog_svm:'hog_svm',optical_flow:'optical_flow',
    stereo:'stereo',histogram:'histogram',threshold:'threshold',
    gan:'gan',diffusion:'diffusion',lenet:'lenet',
    fcn:'semantic',faster_rcnn:'detection',mask_rcnn:'instance',
    detection:'detection',semantic:'semantic',instance:'instance',
    frequency:'frequency',
    noise:'noise',gaussian:'gaussian',sobel:'sobel',median:'median',
    bilateral:'bilateral',nms:'nms',tpl_match:'template_match',
    kmeans:'kmeans',live:'live',conv_training:'conv_training',
    vit:'vit',detr:'detr',sam:'sam',clip:'clip',nerf:'nerf',
    sd:'stable_diffusion',
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
    'noise','gaussian','sobel','median','bilateral','grayscale','histogram','threshold',
    'convolution','edge','corner','sift','hough','morphology','contour','nms','template_match',
    'kmeans','watershed','grabcut','slic','hog_svm','optical_flow','stereo','match','frequency',
    'lenet','gan','detection','semantic','instance','diffusion',
    'live','conv_training','vit','detr','sam','clip','nerf','stable_diffusion',
    'shitomasi','ncuts','bovw_spm','calibration','epipolar','sfm','cnn_basics','resnet',
    'fcn','unet','faster_rcnn','yolo','mask_rcnn','ddpm','simclr','moco','byol','ijepa',
    '3dgs','pointnet','bev','occupy','c3d','bytetrack','botsort','deeppose','openpose',
  ]);
  const SPECIAL_PAGE_IDS=new Set(['convolution','edge','corner','sift','match']);
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
    if(window.AlgorithmContent && window.AlgorithmContent[id] && !SPECIAL_PAGE_IDS.has(id)){
      const a=_allAlgos().find(x=>x.id===id)||{id,n:id,en:''};
      return {id, name:a.n||id, name_en:a.en||'', page:'teaching.html?id='+encodeURIComponent(id)};
    }
    if(impl && !_implRunnable(impl) && window.AlgorithmContent && window.AlgorithmContent[id]){
      const a=_allAlgos().find(x=>x.id===id)||{id,n:id,en:''};
      return {id, name:a.n||id, name_en:a.en||'', page:'teaching.html?id='+encodeURIComponent(id)};
    }
    if(impl && !_implRunnable(impl)) return null;
    if(!_isReady(id) && !(window.AlgorithmContent && window.AlgorithmContent[id] && !impl)) return null;
    if(window.AlgorithmContent && window.AlgorithmContent[id] && !SPECIAL_PAGE_IDS.has(id)&&!SPECIAL_PAGE_IDS.has(rid)){
      const a=_allAlgos().find(x=>x.id===id)||{id,n:id,en:''};
      return {id, name:a.n||id, name_en:a.en||'', page:'teaching.html?id='+encodeURIComponent(id)};
    }
    const m=_apiMap[rid];
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
    Array.from(document.querySelectorAll('.phase-line')).forEach((line,idx)=>{
      const phase=BLUEPRINT[idx];
      if(!phase) return;
      const algos=_phaseAlgos(phase);
      const phaseNode=line.querySelector('.phase-count');
      if(phaseNode) phaseNode.textContent=algos.filter(a=>_isReady(a.id)).length+'/'+algos.length+' 已实现';
      if(phase.subs){
        Array.from(line.querySelectorAll('.sub-group')).forEach((group,gidx)=>{
          const grp=phase.subs[gidx];
          const node=group.querySelector('.sub-group-count');
          if(node&&grp) node.textContent=grp.algo.filter(a=>_isReady(a.id)).length+'/'+grp.algo.length;
        });
      }
    });
  }

  // ── Metro ──
  function _renderMetro(){
    $metro.innerHTML='';
    BLUEPRINT.forEach(phase=>{
      const line=document.createElement('div'); line.className='phase-line';
      const algos=_phaseAlgos(phase);
      const ready=algos.filter(a=>_isReady(a.id)).length;
      let h=`<div class="phase-line-header">
        <span class="station-dot" style="background:${phase.color};box-shadow:0 0 10px ${phase.color}"></span>
        <span class="phase-label">${phase.name} · ${phase.en}</span>
        <span class="phase-sub">${phase.sub}</span>
        <span class="phase-count">${ready}/${algos.length} 已实现</span></div>`;
      if(phase.subs){
        phase.subs.forEach(grp=>{
          const gr=grp.algo.filter(a=>_isReady(a.id)).length;
          h+=`<div class="sub-group"><div class="sub-group-header">
            <span class="sub-group-badge" style="background:${grp.color}">${grp.key}</span>
            <span class="sub-group-label">${grp.label}</span>
            <span class="sub-group-count">${gr}/${grp.algo.length}</span></div>
            <div class="phase-cards">${grp.algo.map(a=>_card(a)).join('')}</div></div>`;
        });
      }else{
        h+=`<div class="phase-cards">${phase.algo.map(a=>_card(a)).join('')}</div>`;
      }
      line.innerHTML=h;
      line.querySelectorAll('.algo-card').forEach(c=>c.addEventListener('click',()=>Router.go('/module/'+c.dataset.mid)));
      $metro.appendChild(line);
    });
  }
  function _card(a){
    const state=_moduleState(a.id);
    const impl=state.impl||{};
    const ready=state.ready;
    const isNew=a.planned;
    const cls=['algo-card',ready?'realized':(isNew?'planned':'pending')].filter(Boolean).join(' ');
    const tag=_implRealModel(impl) ? (impl.category==='requires_external_weights' ? '需要权重' : '真实模型') : (impl.category==='not_implemented' ? '未接入' : (impl.category==='requires_external_weights' ? '需要权重' : (ready?'可体验':(isNew?'规划中':'待实现'))));
    // Difficulty stars: find which phase this algo belongs to
    let diff=1; BLUEPRINT.forEach(p=>{if(_phaseAlgos(p).some(x=>x.id===a.id))diff=p.diff;});
    return `<div class="${cls}" data-mid="${a.id}" title="${a.d}"><div class="card-name">${a.n}</div><div class="card-en">${a.en}</div><div class="card-desc">${a.d}</div><div class="card-meta"><span class="card-tag">${tag}</span></div></div>`;
  }

  // ── Detail ──
  function openModule(id){
    const m=_openIfReady(id); if(!m){Utils.toast('模块 "'+id+'" 尚未实现');return;}
    _current=m; $dname.textContent=m.name; $den.textContent=m.name_en||'';
    $ifr.src=m.page?'/static/pages/'+m.page:''; $overlay.classList.add('active');
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
    const all=_allAlgos();
    const hits=all.filter(a=>a.n.toLowerCase().includes(q)||a.en.toLowerCase().includes(q)||a.d.toLowerCase().includes(q));
    document.querySelectorAll('.algo-card').forEach(c=>{
      c.classList.toggle('hidden',!hits.some(h=>h.id===c.dataset.mid));
    });
    document.querySelectorAll('.sub-group').forEach(g=>{
      g.classList.toggle('hidden',!g.querySelector('.algo-card:not(.hidden)'));
    });
    if(hits.length===1) Router.go('/module/'+hits[0].id);
  }
  function _clearSearch(){
    document.querySelectorAll('.algo-card.hidden,.sub-group.hidden').forEach(c=>c.classList.remove('hidden'));
  }

  // ── Iframe messages ──
  function _onIframeMsg(e){const{type,payload}=e.data||{};if(!type)return;
    switch(type){case'toast':Utils.toast(payload?.message||'',payload?.duration);break;
      case'navigate':Router.go('/module/'+(payload?.moduleId||''));break;}
  }

  window.addEventListener('DOMContentLoaded',init);
  return{openModule,closeDetail,getModules:()=>_apiModules};
})();
