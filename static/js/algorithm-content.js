(function() {
  const common = {
    uploadHint: '上传一张图片后，页面会调用当前项目中的 NumPy 实现，并把中间结果逐步展开。',
  };

  const implementationMeta = {
    gan: { status: '未接入真实实现', category: 'not_implemented', localInference: false, realModel: false, model: '' },
    diffusion: { status: '真实预训练模型', category: 'pretrained_model', localInference: true, realModel: true, model: 'runwayml/stable-diffusion-v1-5' },
    detection: { status: '真实预训练模型', category: 'pretrained_model', localInference: true, realModel: true, model: 'fasterrcnn_resnet50_fpn' },
    semantic: { status: '真实预训练模型', category: 'pretrained_model', localInference: true, realModel: true, model: 'fcn_resnet50' },
    instance: { status: '真实预训练模型', category: 'pretrained_model', localInference: true, realModel: true, model: 'maskrcnn_resnet50_fpn' },
    vit: { status: '真实预训练模型', category: 'pretrained_model', localInference: true, realModel: true, model: 'google/vit-base-patch16-224' },
    detr: { status: '真实预训练模型', category: 'pretrained_model', localInference: true, realModel: true, model: 'facebook/detr-resnet-50' },
    sam: { status: '真实预训练模型', category: 'pretrained_model', localInference: true, realModel: true, model: 'SAM ViT-B checkpoint' },
    clip: { status: '真实预训练模型', category: 'pretrained_model', localInference: true, realModel: true, model: 'openai/clip-vit-base-patch32' },
    stable_diffusion: { status: '真实预训练模型', category: 'pretrained_model', localInference: true, realModel: true, model: 'runwayml/stable-diffusion-v1-5' },
    sd: { status: '真实预训练模型', category: 'pretrained_model', localInference: true, realModel: true, model: 'runwayml/stable-diffusion-v1-5' },
    nerf: { status: '未接入真实预训练场景', category: 'not_implemented', localInference: false, realModel: false, model: 'TinyNeRF random-weight MLP' }
  };

  const externalWeightIds = new Set([
    'vit','detr','sam','clip','stable_diffusion','sd',
    'swin','dino','mae','dino_det','grdino','mask2former','sam2','blip2',
    'controlnet','dit','flux','stylegan','dust3r','orbslam3','mediapipe','vitpose'
  ]);

  const offlineTeachingIds = new Set([
    'shitomasi','ncuts','bovw_spm','calibration','epipolar','sfm','cnn_basics','resnet',
    'fcn','unet','faster_rcnn','yolo','mask_rcnn','gan','conv_training','nerf','ddpm',
    'simclr','moco','byol','ijepa','3dgs','pointnet','bev','occupy','c3d','bytetrack',
    'botsort','deeppose','openpose'
  ]);

  function attachImplementation(id, cfg) {
    let meta = implementationMeta[id] || cfg.implementation;
    if (externalWeightIds.has(id)) {
      meta = {
        status: '需要外部权重',
        category: 'requires_external_weights',
        localInference: false,
        realModel: true,
        model: (meta && meta.model) || '',
        note: '该算法需要本地预训练权重或远程模型，本轮离线模式暂不运行。'
      };
    } else if (offlineTeachingIds.has(id)) {
      meta = {
        status: '离线教学演示',
        category: 'teaching_simulation',
        localInference: true,
        realModel: false,
        model: 'NumPy/PIL offline teaching demo',
        note: '本地教学可视化，不代表真实预训练模型推理。'
      };
      cfg.endpoint = cfg.endpoint || ('/api/demo/' + id);
    }
    if (meta) {
      cfg.implementation = meta;
      cfg.status = meta.status || cfg.status;
      cfg.metrics = Object.assign({
        '实现状态': meta.status || '',
        '模型/后端': meta.model || meta.backend || '',
        '真实预训练模型': meta.realModel ? '是' : '否'
      }, cfg.metrics || {});
    }
    return cfg;
  }

  window.AlgorithmContent = {
    colorspace: {
      phase: '阶段一 · 基础原语',
      title: '色彩空间转换',
      english: 'RGB / HSV / Lab / Grayscale',
      tagline: '同一张图像可以从不同语义坐标系观察：RGB 看显示器发光，HSV 看人类调色，Lab 看感知均匀性，灰度只保留亮度。',
      status: '语义科普讲解',
      difficulty: '入门',
      formula: 'RGB \\rightarrow HSV,\\quad RGB \\rightarrow Lab,\\quad Y=0.299R+0.587G+0.114B',
      principles: [
        '色彩空间不是改变图像内容，而是改变描述颜色的坐标系。不同算法关心的语义不同，所以会选择不同空间。',
        'RGB 适合显示和通道计算；HSV 适合按色相、饱和度、明度做交互调色；Lab 更接近人眼感知距离；灰度则为阈值、梯度和形状分析压缩掉颜色。'
      ],
      core: [
        ['输入', 'RGB 彩色图像'],
        ['核心问题', '同一颜色应该用“发光通道、调色属性、感知距离还是亮度”来描述'],
        ['输出', 'RGB、HSV、Lab、灰度四种语义视角']
      ],
      pipeline: [
        ['RGB', '把颜色拆成红、绿、蓝三个显示通道，最接近屏幕如何发光。'],
        ['HSV', '把颜色拆成色相 H、饱和度 S、明度 V，最接近“选颜色”的直觉。'],
        ['Lab', '把颜色拆成亮度 L 与两条对立色轴 a/b，颜色距离更接近人眼感受。'],
        ['灰度', '把 RGB 压缩成一个亮度通道，放弃颜色，只保留结构和明暗。']
      ],
      conceptSteps: [
        { id: 'rgb_mode', name: 'RGB：屏幕发光模型', explanation: '红、绿、蓝三个通道相加生成颜色，适合显示、滤波和通道级处理。', formula: '[R,G,B]' },
        { id: 'hsv_mode', name: 'HSV：调色盘模型', explanation: 'H 表示颜色种类，S 表示颜色纯不纯，V 表示亮不亮，适合颜色筛选和交互调参。', formula: '[H,S,V]' },
        { id: 'lab_mode', name: 'Lab：感知距离模型', explanation: 'L 是亮度，a 是绿到红，b 是蓝到黄；两个颜色在 Lab 中距离近，通常人眼也觉得接近。', formula: '[L,a,b]' },
        { id: 'gray_mode', name: '灰度：结构亮度模型', explanation: '把颜色压成一个亮度值，方便阈值、边缘、角点等基础视觉算法使用。', formula: 'Y=0.299R+0.587G+0.114B' }
      ],
      visualStory: {
        intro: '同一批像素，换一个坐标系，就会突出完全不同的语义。',
        cards: [
          {
            type: 'mix',
            title: 'RGB：红绿蓝三束光',
            text: 'RGB 直接描述屏幕发光强度。算法看到的是三个颜色通道，各自可以滤波、卷积或统计。',
            labels: ['R', 'G', 'B']
          },
          {
            type: 'gradientSet',
            title: 'HSV：更像调色盘',
            text: '色相决定“是什么颜色”，饱和度决定“颜色纯不纯”，明度决定“亮不亮”。',
            rows: [
              { label: 'H', caption: '色相环', gradient: 'linear-gradient(90deg,#ef4444,#f97316,#facc15,#22c55e,#06b6d4,#3b82f6,#8b5cf6,#ef4444)' },
              { label: 'S', caption: '纯度', gradient: 'linear-gradient(90deg,#dbeafe,#60a5fa,#2563eb)' },
              { label: 'V', caption: '明度', gradient: 'linear-gradient(90deg,#020617,#2563eb,#f8fafc)' }
            ]
          },
          {
            type: 'gradientSet',
            title: 'Lab：更接近人眼距离',
            text: 'L 表示明暗，a/b 表示两组对立颜色。做颜色差异、聚类和校正时，Lab 往往比 RGB 更符合感知。',
            rows: [
              { label: 'L', caption: '黑↔白', gradient: 'linear-gradient(90deg,#020617,#64748b,#f8fafc)' },
              { label: 'a', caption: '绿↔红', gradient: 'linear-gradient(90deg,#22c55e,#f8fafc,#ef4444)' },
              { label: 'b', caption: '蓝↔黄', gradient: 'linear-gradient(90deg,#2563eb,#f8fafc,#facc15)' }
            ]
          },
          {
            type: 'gradientSet',
            title: '灰度：只留下结构',
            text: '灰度把颜色变成亮度，牺牲色彩，换来更简单稳定的形状、边缘和纹理分析入口。',
            rows: [
              { label: 'Y', caption: '亮度轴', gradient: 'linear-gradient(90deg,#020617,#1e293b,#64748b,#cbd5e1,#ffffff)' },
              { label: 'R', caption: '0.299', gradient: 'linear-gradient(90deg,#020617,#ef4444)' },
              { label: 'G', caption: '0.587', gradient: 'linear-gradient(90deg,#020617,#22c55e)' },
              { label: 'B', caption: '0.114', gradient: 'linear-gradient(90deg,#020617,#3b82f6)' }
            ]
          }
        ]
      },
      metrics: {
        '展示类型': '四种色彩空间语义对比',
        '本地推理': '讲解模式',
        '覆盖模式': 'RGB / HSV / Lab / 灰度'
      }
    },
    grayscale: {
      phase: '阶段一 · 基础原语',
      title: '灰度转换',
      english: 'Grayscale Conversion',
      tagline: '把 RGB 三个通道压缩成一个亮度通道，是后续阈值、梯度、角点和特征检测的共同入口。',
      status: '真实 NumPy 算法',
      difficulty: '入门',
      endpoint: '/api/demo/grayscale',
      formula: 'Y = 0.299R + 0.587G + 0.114B',
      principles: [
        '灰度图不是简单地丢掉颜色，而是把颜色转成“亮度”。人眼对绿色最敏感，对蓝色较不敏感，所以加权灰度通常比三通道平均更接近视觉感受。',
        '灰度转换会减少数据维度，让很多只依赖亮度变化的算法更稳定、更快。'
      ],
      core: [
        ['输入', 'RGB 或 RGBA 图像'],
        ['核心问题', '如何用一个数表示一个像素的亮度'],
        ['输出', '单通道 0-255 灰度图']
      ],
      pipeline: [
        ['读取图像', '去掉 alpha 通道并转成 uint8。'],
        ['分离通道', '取出 R/G/B 三个亮度来源。'],
        ['亮度加权', '用感知权重合成灰度。'],
        ['结果解释', '黑色代表低亮度，白色代表高亮度。']
      ],
      stepMeta: {
        original: ['原图', '保留输入图像的颜色信息。', 'I(x,y)=[R,G,B]'],
        channels: ['通道分离', '观察 R/G/B 对亮度贡献的差异。', 'R,G,B'],
        formula: ['公式应用', '按选定方案把三通道合成为一个通道。', 'Y=w_rR+w_gG+w_bB'],
        result: ['灰度结果', '后续多数基础算法从这张图开始。', 'Y∈[0,255]']
      }
    },
    histogram: {
      phase: '阶段一 · 基础原语',
      title: '直方图与均衡化',
      english: 'Histogram & Equalization',
      tagline: '统计每个亮度值出现多少次，用分布而不是单个像素理解图像的曝光、对比度和动态范围。',
      status: '真实 NumPy 算法',
      difficulty: '入门',
      endpoint: '/api/demo/histogram',
      formula: 'h(k)=\\sum_{x,y} [I(x,y)=k],\\quad CDF(k)=\\sum_{i=0}^{k} h(i)',
      principles: [
        '直方图把空间位置暂时放下，只看亮度分布。峰值集中在左侧通常偏暗，集中在右侧通常偏亮，分布很窄说明对比度较低。',
        '均衡化用累计分布函数重新映射亮度，让可用灰度范围更充分。'
      ],
      core: [
        ['输入', '灰度亮度值'],
        ['核心问题', '亮度是否集中、偏暗或偏亮'],
        ['输出', '直方图、CDF 和增强后的图像']
      ],
      pipeline: [
        ['灰度化', '把颜色统一成亮度统计。'],
        ['计数', '统计 0-255 每个亮度出现次数。'],
        ['累计分布', '计算 CDF，观察亮度覆盖范围。'],
        ['均衡化', '用 CDF 重新分配亮度。']
      ],
      stepMeta: {
        original: ['原图', '统计开始前的输入。', 'I'],
        gray: ['灰度图', '只保留亮度。', 'Y'],
        histogram: ['亮度直方图', '展示每个亮度桶的像素数。', 'h(k)'],
        cdf: ['累计分布', '显示小于等于 k 的像素比例。', 'CDF(k)'],
        equalized: ['均衡化结果', '增强局部和整体对比度。', 'T(k)=round(255·CDF(k))']
      }
    },
    threshold: {
      phase: '阶段一 · 基础原语',
      title: '阈值化',
      english: 'Thresholding',
      tagline: '用一个或一组阈值把灰度图切成前景和背景，是最朴素也最常用的分割思想。',
      status: '真实 NumPy 算法',
      difficulty: '入门',
      endpoint: '/api/demo/threshold',
      formula: 'B(x,y)=255\\;if\\;I(x,y)\\ge T,\\;else\\;0',
      principles: [
        '全局阈值假设整张图的前景和背景可以用同一条亮度线分开。Otsu 自动寻找让类间方差最大的阈值。',
        '自适应阈值改用局部窗口均值，适合光照不均的文档、显微图和阴影场景。'
      ],
      controls: [
        { name: 'method', label: '方法', type: 'select', value: 'otsu', options: [['otsu','Otsu 自动阈值'], ['global','全局阈值'], ['adaptive','自适应均值']] },
        { name: 'threshold', label: '全局阈值', type: 'range', min: 0, max: 255, step: 1, value: 128 },
        { name: 'block_size', label: '局部窗口', type: 'range', min: 3, max: 31, step: 2, value: 11 },
        { name: 'C', label: '偏移 C', type: 'range', min: -10, max: 10, step: 1, value: 2 }
      ],
      core: [
        ['输入', '灰度图和阈值策略'],
        ['核心问题', '前景和背景能否按亮度分开'],
        ['输出', '黑白二值图']
      ],
      pipeline: [
        ['灰度化', '把颜色空间压缩为亮度。'],
        ['分析分布', '查看直方图中前景/背景是否分离。'],
        ['确定阈值', '手动、Otsu 或局部均值。'],
        ['二值化', '输出可供形态学和轮廓使用的黑白图。']
      ],
      stepMeta: {
        original: ['原图', '待分割输入。', 'I'],
        gray: ['灰度图', '阈值化基于亮度。', 'Y'],
        histogram: ['阈值位置', '红线显示当前阈值或默认参考。', 'h(k),T'],
        thr64: ['阈值 64', '低阈值会保留更多前景。', 'T=64'],
        thr128: ['阈值 128', '中间阈值的基础效果。', 'T=128'],
        thr192: ['阈值 192', '高阈值只保留很亮区域。', 'T=192'],
        otsu: ['Otsu 结果', '自动寻找类间方差最大的阈值。', '\\max \\sigma_b^2'],
        adaptive: ['自适应结果', '每个像素使用局部阈值。', 'T=mean(\\Omega)-C'],
        result: ['二值化结果', '白色为前景，黑色为背景。', 'B']
      }
    },
    noise: {
      phase: '阶段一 · 基础原语',
      title: '噪声模型',
      english: 'Noise Models',
      tagline: '先理解噪声如何破坏图像，才知道为什么需要滤波、平滑和鲁棒特征。',
      status: '真实 NumPy 算法',
      difficulty: '入门',
      endpoint: '/api/demo/noise',
      formula: 'I_{noisy}=I+n,\\quad n\\sim\\mathcal{N}(0,\\sigma^2)',
      principles: [
        '高斯噪声像连续的亮度抖动，通常来自传感器和电子信号。',
        '椒盐噪声会把少量像素直接变成黑点或白点，中值滤波通常比线性滤波更适合处理它。'
      ],
      core: [['输入', '原始图像'], ['核心问题', '噪声分布如何影响像素'], ['输出', '带噪图像对比']],
      pipeline: [['读取图像', '得到原图像素。'], ['随机采样', '根据噪声分布生成扰动。'], ['叠加噪声', '改变部分或全部像素。'], ['观察退化', '为后续滤波建立直觉。']],
      stepMeta: {
        original: ['原图', '干净输入。', 'I'],
        salt_pepper: ['椒盐噪声', '随机像素变成 0 或 255。', 'p(I=0/255)=amount'],
        gaussian_noise: ['高斯噪声', '每个像素叠加正态扰动。', 'I+n']
      }
    },
    gaussian: {
      phase: '阶段一 · 基础原语',
      title: '高斯模糊',
      english: 'Gaussian Blur',
      tagline: '用距离中心越远权重越小的核做平滑，是尺度空间、Canny 和 SIFT 的前置基础。',
      status: '真实 NumPy 算法',
      difficulty: '入门',
      endpoint: '/api/demo/gaussian',
      formula: 'G(x,y)=\\frac{1}{2\\pi\\sigma^2}e^{-(x^2+y^2)/(2\\sigma^2)}',
      principles: [
        '高斯核不会平均对待窗口内的所有像素，中心像素权重大，远处像素权重小。',
        'σ 越大，模糊越强，细节和噪声越容易被压掉。'
      ],
      core: [['输入', '灰度或彩色图'], ['核心问题', '如何平滑又尽量少引入硬边界'], ['输出', '降噪后的平滑图']],
      pipeline: [['灰度化', '简化亮度计算。'], ['构造高斯核', '按距离分配权重。'], ['二维卷积', '滑窗加权求和。'], ['输出平滑图', '为梯度或分割降低噪声。']],
      stepMeta: {
        original: ['原图', '待平滑输入。', 'I'],
        gray: ['灰度图', '平滑前亮度。', 'Y'],
        gaussian: ['高斯模糊', '按高斯权重进行局部平均。', 'G*I']
      }
    },
    sobel: {
      phase: '阶段一 · 基础原语',
      title: 'Sobel 梯度',
      english: 'Sobel Gradient',
      tagline: '计算图像在 x/y 方向的亮度变化，边缘、角点、HOG 和光流都建立在梯度之上。',
      status: '真实 NumPy 算法',
      difficulty: '入门',
      endpoint: '/api/demo/sobel',
      formula: '|\\nabla I|=\\sqrt{G_x^2+G_y^2},\\quad \\theta=atan2(G_y,G_x)',
      principles: [
        'Sobel 核一边做差分，一边做轻微平滑，因此比简单相邻差分更抗噪。',
        '梯度幅值表示变化强度，梯度方向表示亮度上升最快的方向。'
      ],
      core: [['输入', '灰度图'], ['核心问题', '哪里变化最大、朝哪个方向变'], ['输出', 'Gx、Gy、幅值和方向']],
      pipeline: [['灰度化', '得到亮度图。'], ['水平核卷积', '检测垂直边缘。'], ['垂直核卷积', '检测水平边缘。'], ['合成幅值方向', '形成边缘强度和角度。']],
      stepMeta: {
        original: ['原图', '输入图像。', 'I'],
        gx: ['水平梯度 Gx', '亮处表示水平方向变化大。', 'G_x=I*K_x'],
        gy: ['垂直梯度 Gy', '亮处表示垂直方向变化大。', 'G_y=I*K_y'],
        magnitude: ['梯度幅值', '综合两个方向的变化强度。', '\\sqrt{G_x^2+G_y^2}'],
        angle: ['梯度方向', '用灰度编码方向角。', 'atan2(G_y,G_x)']
      }
    },
    median: {
      phase: '阶段一 · 基础原语',
      title: '中值滤波',
      english: 'Median Filter',
      tagline: '用窗口内的中位数替代中心像素，擅长消除椒盐噪声，同时比均值滤波更保边。',
      status: '真实 NumPy 算法',
      difficulty: '入门',
      endpoint: '/api/demo/median',
      formula: 'I^{\\prime}(x,y)=median\\{I(u,v)|(u,v)\\in\\Omega\\}',
      principles: ['中值滤波是非线性的，它不做加权平均，而是选择排序后的中间值。', '孤立黑白点往往是极端值，会在中位数选择中被自然排除。'],
      core: [['输入', '含噪图像'], ['核心问题', '如何去掉孤立异常值'], ['输出', '去椒盐后的图像']],
      pipeline: [['读取图像', '保留原图。'], ['加入或面对噪声', '观察异常像素。'], ['滑动窗口排序', '取局部中位数。'], ['输出结果', '噪点减少，边界相对保留。']],
      stepMeta: {
        original: ['原图', '滤波前输入。', 'I'],
        noisy: ['椒盐噪声', '孤立黑白异常点。', 'I+n'],
        median: ['中值滤波结果', '局部排序取中位数。', 'median(\\Omega)']
      }
    },
    bilateral: {
      phase: '阶段一 · 基础原语',
      title: '双边滤波',
      english: 'Bilateral Filter',
      tagline: '同时考虑空间距离和颜色差异，平滑同质区域，同时尽量保留边缘。',
      status: '真实 NumPy 算法',
      difficulty: '基础',
      endpoint: '/api/demo/bilateral',
      formula: 'I_p^{\\prime}=\\frac{1}{W_p}\\sum_q G_s(\\|p-q\\|)G_r(\\|I_p-I_q\\|)I_q',
      principles: ['普通高斯只看空间距离，跨过边缘也会平均，容易把边界抹糊。', '双边滤波要求“离得近”且“颜色像”才给高权重，所以边缘两侧不容易互相污染。'],
      core: [['输入', '彩色或灰度图'], ['核心问题', '怎样平滑而不跨边缘平均'], ['输出', '保边平滑图']],
      pipeline: [['空间权重', '距离越近权重越大。'], ['颜色权重', '亮度或颜色越像权重越大。'], ['组合权重', '两个条件同时满足才贡献大。'], ['归一化输出', '得到保边平滑结果。']],
      stepMeta: {
        original: ['原图', '待平滑输入。', 'I'],
        gaussian: ['普通高斯平滑', '只考虑空间距离。', 'G_s*I'],
        bilateral: ['双边滤波', '空间核和颜色核共同作用。', 'G_sG_rI']
      }
    },
    hough: {
      phase: '阶段二 · 经典特征检测',
      title: 'Hough 变换',
      english: 'Hough Transform',
      tagline: '让边缘像素在参数空间投票，用“很多点共同支持一个模型”的方式找直线或圆。',
      status: '真实 NumPy 算法',
      difficulty: '进阶',
      endpoint: '/api/demo/hough',
      formula: '\\rho=x\\cos\\theta+y\\sin\\theta',
      principles: ['图像空间里一条直线上的许多点，在参数空间会汇聚到同一个 (ρ,θ)。', 'Hough 的优势是即使边缘有断裂，投票峰值仍然能把整体几何形状找出来。'],
      core: [['输入', '边缘图'], ['核心问题', '哪些边缘点支持同一条直线'], ['输出', '参数空间峰值和检测线']],
      pipeline: [['边缘检测', '提取可能属于形状的像素。'], ['参数投票', '每个边缘点对多个 θ 投票。'], ['峰值选择', '寻找累加器高峰。'], ['线段回绘', '把参数空间结果映射回图像。']],
      stepMeta: {
        original: ['原图', '输入图像。', 'I'],
        edges: ['边缘图', '只让边缘点参与投票。', 'E'],
        accumulator: ['累加器', '参数空间中的投票强度。', 'A(ρ,θ)'],
        lines: ['检测线', '把峰值线绘制回原图。', 'ρ=xcosθ+ysinθ']
      }
    },
    morphology: {
      phase: '阶段二 · 经典特征检测',
      title: '形态学操作',
      english: 'Morphology',
      tagline: '用结构元素在二值图上做集合操作，控制白色区域的收缩、扩张、去噪和补洞。',
      status: '真实 NumPy 算法',
      difficulty: '基础',
      endpoint: '/api/demo/morphology',
      formula: 'A\\oplus B,\\quad A\\ominus B',
      principles: ['腐蚀要求结构元素覆盖区域全部为白，能让前景缩水并去掉小噪点。', '膨胀只要结构元素覆盖到白色就扩张，能补小断裂。开运算是先腐蚀后膨胀，闭运算相反。'],
      core: [['输入', '二值图'], ['核心问题', '如何用形状规则修改前景'], ['输出', '腐蚀、膨胀、开闭运算']],
      pipeline: [['阈值化', '得到二值前景。'], ['选择结构元素', '定义局部邻域形状。'], ['集合运算', '腐蚀/膨胀/开/闭。'], ['形状清理', '减少噪点或填补缝隙。']],
      stepMeta: {
        original: ['原图', '输入图像。', 'I'],
        binary: ['二值图', '形态学操作对象。', 'A'],
        erode: ['腐蚀', '白色区域收缩。', 'A⊖B'],
        dilate: ['膨胀', '白色区域扩张。', 'A⊕B'],
        opening: ['开运算', '去除小噪点。', '(A⊖B)⊕B'],
        closing: ['闭运算', '填补小孔洞。', '(A⊕B)⊖B'],
        gradient: ['形态学梯度', '膨胀和腐蚀之差近似边界。', '(A⊕B)-(A⊖B)']
      }
    },
    contour: {
      phase: '阶段二 · 经典特征检测',
      title: '轮廓查找',
      english: 'Contour Finding',
      tagline: '在二值图中沿着前景边界行走，把像素区域变成可测量的形状对象。',
      status: '真实 NumPy 算法',
      difficulty: '基础',
      endpoint: '/api/demo/contour',
      formula: 'C=\\{(x,y)|B(x,y)=1,\\exists neighbor=0\\}',
      principles: ['轮廓是前景与背景的交界，它把像素块压缩成边界点序列。', '有了轮廓之后，就能计算面积、周长、外接框、近似多边形和目标数量。'],
      core: [['输入', '二值图'], ['核心问题', '哪些像素位于前景边界'], ['输出', '轮廓叠加与形状指标']],
      pipeline: [['阈值化', '把目标从背景中分离。'], ['边界追踪', '找到前景外沿。'], ['轮廓近似', '减少冗余点。'], ['形状分析', '统计面积和数量。']],
      stepMeta: {
        original: ['原图', '待分析输入。', 'I'],
        binary: ['二值图', '轮廓来自前景边界。', 'B'],
        contours: ['轮廓叠加', '把边界点画回图像。', 'C'],
        approximation: ['轮廓近似', '用更少点描述形状。', 'approx(C)']
      }
    },
    nms: {
      phase: '阶段二 · 经典特征检测',
      title: '非极大值抑制',
      english: 'Non-Maximum Suppression',
      tagline: '只保留局部最强响应，让厚边缘变细，也让重复检测框变成一个结果。',
      status: '真实 NumPy 算法',
      difficulty: '进阶',
      endpoint: '/api/demo/nms',
      formula: 'keep(p) \\iff R(p)=\\max_{q\\in direction(p)}R(q)',
      principles: ['在 Canny 中，NMS 沿梯度方向比较相邻响应，只保留中心最大的像素。', '同一个思想也用于目标检测框：分数高的框留下，和它高度重叠的框删掉。'],
      core: [['输入', '响应图或候选框'], ['核心问题', '怎样删除重复局部响应'], ['输出', '稀疏且更准确的结果']],
      pipeline: [['计算响应', '得到边缘强度或候选分数。'], ['确定比较方向', '沿梯度方向或 IoU 重叠比较。'], ['局部最大判断', '保留最强者。'], ['输出稀疏结果', '边缘变细或框去重。']],
      stepMeta: {
        original: ['原图', '输入。', 'I'],
        gradient: ['梯度幅值', '未抑制前边缘较厚。', '|∇I|'],
        nms: ['NMS 后', '只保留局部最大响应。', 'local max']
      }
    },
    template_match: {
      phase: '阶段二 · 经典特征检测',
      title: '模板匹配',
      english: 'Template Matching',
      tagline: '让小模板在大图中滑动，用相似度峰值定位目标，是最直接的目标查找方法。',
      status: '真实 NumPy 算法',
      difficulty: '基础',
      endpoint: '/api/demo/template_match',
      formula: 'NCC(x,y)=\\frac{\\sum (I-\\bar I)(T-\\bar T)}{\\|I-\\bar I\\|\\|T-\\bar T\\|}',
      principles: ['互相关衡量模板和当前位置窗口是否相像，归一化互相关能减少整体亮度变化影响。', '模板匹配适合目标外观稳定、尺度和旋转变化不大的场景。'],
      core: [['输入', '搜索图和模板'], ['核心问题', '哪个位置最像模板'], ['输出', '相似度热图和最佳位置']],
      pipeline: [['选取模板', '从图中或预设区域得到目标样子。'], ['滑窗比较', '逐位置计算相似度。'], ['峰值寻找', '选出最高响应位置。'], ['结果框定', '把匹配位置画回原图。']],
      stepMeta: {
        original: ['原图', '搜索区域。', 'I'],
        template: ['模板', '被查找的小图块。', 'T'],
        response: ['响应图', '亮处表示相似度高。', 'NCC'],
        match: ['匹配结果', '最佳相似位置。', 'argmax NCC']
      }
    },
    kmeans: {
      phase: '阶段三 · 中级视觉',
      title: 'K-Means 分割',
      english: 'K-Means Segmentation',
      tagline: '把像素颜色看作点，用聚类中心把图像粗分成若干颜色区域。',
      status: '真实 NumPy 算法',
      difficulty: '进阶',
      endpoint: '/api/demo/kmeans',
      formula: '\\min\\sum_i\\|x_i-\\mu_{c_i}\\|^2',
      principles: ['每个像素是 RGB 空间里的一个点，K-Means 反复执行“分配最近中心”和“更新中心”。', '它没有语义理解，只按颜色相似度分组，因此适合做无监督分割直觉入门。'],
      controls: [{ name: 'k', label: '聚类数 K', type: 'range', min: 2, max: 8, step: 1, value: 4 }],
      core: [['输入', 'RGB 像素点'], ['核心问题', '哪些像素颜色相近'], ['输出', '聚类标签和重着色图']],
      pipeline: [['采样像素', '把图像拉平成 RGB 点集。'], ['初始化中心', '选择 K 个颜色中心。'], ['迭代分配更新', '每轮靠近最近中心。'], ['重建图像', '用中心颜色替代像素。']],
      stepMeta: {
        original: ['原图', '待聚类颜色。', 'x_i=[R,G,B]'],
        clustered: ['聚类结果', '同类像素使用同一中心颜色。', 'μ_c'],
        labels: ['标签图', '显示每个像素所属簇。', 'c_i']
      }
    },
    watershed: {
      phase: '阶段三 · 中级视觉',
      title: '分水岭分割',
      english: 'Watershed Segmentation',
      tagline: '把梯度图想成地形，从标记点向外注水，水盆相遇的位置就是分割边界。',
      status: '真实 NumPy 算法',
      difficulty: '进阶',
      endpoint: '/api/demo/watershed',
      formula: 'label(p)=label(argmin\\ gradient\\ path)',
      principles: ['亮度变化大的地方像山脊，不同区域从低处扩张，遇到山脊就形成边界。', '标记控制可以减少过分割，让算法从可信前景和背景种子出发。'],
      core: [['输入', '梯度图和种子'], ['核心问题', '不同区域在哪里相遇'], ['输出', '分割标签和边界']],
      pipeline: [['计算梯度', '把边缘看作地形高度。'], ['生成标记', '确定初始水源。'], ['按梯度扩张', '低梯度区域优先合并。'], ['边界绘制', '不同标签相遇处形成边界。']],
      stepMeta: {
        original: ['原图', '待分割图像。', 'I'],
        gradient: ['梯度地形', '亮处是边界山脊。', '|∇I|'],
        markers: ['标记点', '初始区域种子。', 'M'],
        labels: ['标签图', '区域扩张后的分区。', 'L'],
        boundary: ['分割边界', '不同标签相遇的位置。', '∂L']
      }
    },
    grabcut: {
      phase: '阶段三 · 中级视觉',
      title: 'GrabCut 前景提取',
      english: 'GrabCut',
      tagline: '用一个矩形给出粗略前景范围，再通过颜色模型和图割迭代分离前景背景。',
      status: '真实 NumPy 教学实现',
      difficulty: '进阶',
      endpoint: '/api/demo/grabcut',
      formula: 'E(L)=\\sum_p D_p(L_p)+\\lambda\\sum_{p,q}V_{p,q}[L_p\\ne L_q]',
      principles: ['GrabCut 把“颜色像不像前景/背景”和“相邻像素是否应该同类”合在一个能量函数中。', '真实工业实现通常用 GMM 和 max-flow；当前页面展示教学版的核心迭代直觉。'],
      controls: [
        { name: 'x', label: '框 x', type: 'range', min: 0, max: 200, step: 5, value: 30 },
        { name: 'y', label: '框 y', type: 'range', min: 0, max: 200, step: 5, value: 30 },
        { name: 'w', label: '框宽', type: 'range', min: 40, max: 260, step: 5, value: 160 },
        { name: 'h', label: '框高', type: 'range', min: 40, max: 260, step: 5, value: 160 }
      ],
      core: [['输入', '图像和前景矩形'], ['核心问题', '哪些颜色更像前景'], ['输出', '前景掩码和抠图结果']],
      pipeline: [['矩形初始化', '框内可能前景，框外背景。'], ['颜色建模', '估计前景/背景颜色分布。'], ['迭代更新', '根据颜色和边界平滑更新标签。'], ['生成掩码', '输出前景区域。']],
      stepMeta: {
        original: ['原图', '输入图像。', 'I'],
        init: ['矩形初始化', '粗略指定目标范围。', 'R'],
        mask: ['前景掩码', '估计的前景区域。', 'L'],
        result: ['抠图结果', '只保留前景。', 'I·L']
      }
    },
    slic: {
      phase: '阶段三 · 中级视觉',
      title: 'SLIC 超像素',
      english: 'SLIC Superpixels',
      tagline: '在颜色和空间的五维特征中做局部 K-Means，把像素组织成紧凑小区域。',
      status: '真实 NumPy 算法',
      difficulty: '进阶',
      endpoint: '/api/demo/slic',
      formula: 'D=\\sqrt{d_c^2+(m/S)^2d_s^2}',
      principles: ['SLIC 不追求最终语义分割，而是把图像变成更少、更规则的视觉单元。', 'compactness 越大，空间规则性越强；越小，超像素越贴合颜色边界。'],
      controls: [
        { name: 'num_superpixels', label: '超像素数', type: 'range', min: 50, max: 400, step: 10, value: 200 },
        { name: 'compactness', label: '紧凑度', type: 'range', min: 1, max: 30, step: 1, value: 10 }
      ],
      core: [['输入', '颜色和坐标'], ['核心问题', '如何形成颜色相近且空间紧凑的小块'], ['输出', '超像素标签和边界']],
      pipeline: [['初始化网格中心', '均匀放置聚类中心。'], ['局部搜索', '只在中心附近比较。'], ['更新中心', '按颜色和坐标重新平均。'], ['绘制边界', '显示超像素分块。']],
      stepMeta: {
        original: ['原图', '输入图像。', 'I'],
        labels: ['标签图', '每个超像素的编号。', 'L'],
        boundaries: ['超像素边界', '紧凑区域分界线。', '∂L'],
        result: ['平均色结果', '每个超像素用平均颜色表示。', 'mean(I|L)']
      }
    },
    hog_svm: {
      phase: '阶段三 · 中级视觉',
      title: 'HOG + SVM',
      english: 'Histogram of Oriented Gradients',
      tagline: '把局部梯度方向统计成特征向量，是深度学习前行人检测的经典表示。',
      status: '真实 NumPy 特征提取',
      difficulty: '进阶',
      endpoint: '/api/demo/hog_svm',
      formula: 'H_{cell}(b)=\\sum |\\nabla I|[\\theta\\in bin_b]',
      principles: ['HOG 关注边缘方向分布，而不是原始像素值，因此对光照和小形变更稳。', 'SVM 可以在 HOG 特征空间中学习“像不像目标”的线性分界。'],
      core: [['输入', '灰度图'], ['核心问题', '局部轮廓方向如何编码'], ['输出', 'HOG 可视化和特征维度']],
      pipeline: [['计算梯度', '得到幅值和方向。'], ['划分 cell', '每个小格统计方向直方图。'], ['block 归一化', '降低光照影响。'], ['拼接特征', '形成分类器输入向量。']],
      stepMeta: {
        original: ['原图', '输入图像。', 'I'],
        gray: ['灰度图', '梯度计算基础。', 'Y'],
        magnitude: ['梯度幅值', '边缘强度。', '|∇I|'],
        hog: ['HOG 可视化', '线段方向和长度表示局部方向统计。', 'H_cell']
      }
    },
    optical_flow: {
      phase: '阶段三 · 中级视觉',
      title: '光流',
      english: 'Optical Flow',
      tagline: '估计相邻帧之间像素的运动方向和速度，用矢量场描述视频中的运动。',
      status: '真实 NumPy 教学实现',
      difficulty: '进阶',
      endpoint: '/api/demo/optical_flow',
      formula: 'I_xu+I_yv+I_t=0',
      principles: ['亮度恒定假设认为同一个点在短时间内亮度不变，于是空间梯度和时间梯度共同约束运动。', 'Lucas-Kanade 在局部窗口里解最小二乘，适合小位移和局部平滑运动。'],
      core: [['输入', '两帧图像'], ['核心问题', '像素从哪里移动到哪里'], ['输出', '运动矢量或颜色编码流场']],
      pipeline: [['生成相邻帧', '模拟或读取两帧。'], ['计算 Ix/Iy/It', '空间和时间梯度。'], ['局部求解', '估计 u/v 运动。'], ['绘制流场', '用箭头或颜色显示运动。']],
      stepMeta: {
        frame1: ['第一帧', '运动前图像。', 'I_t'],
        frame2: ['第二帧', '运动后图像。', 'I_{t+1}'],
        flow: ['光流场', '箭头显示局部运动。', '(u,v)'],
        magnitude: ['运动强度', '亮处表示速度更大。', '\\sqrt{u^2+v^2}']
      }
    },
    stereo: {
      phase: '阶段三 · 中级视觉',
      title: '立体匹配',
      english: 'Stereo Matching',
      tagline: '从左右视角寻找同名点，视差越大通常表示物体越近。',
      status: '真实 NumPy 教学实现',
      difficulty: '进阶',
      endpoint: '/api/demo/stereo',
      formula: 'Z=\\frac{fB}{d}',
      principles: ['校正后的双目图像中，同一个点只会沿水平方向移动，搜索从二维降到一维。', '视差 d 是左右图横向位置差，深度 Z 与视差成反比。'],
      core: [['输入', '左右视图'], ['核心问题', '每个像素在另一张图中对应哪里'], ['输出', '视差图和深度直觉']],
      pipeline: [['左右图准备', '模拟双目视角。'], ['窗口匹配', '用 SAD/SSD 比较候选视差。'], ['选择最小代价', '得到每个像素的视差。'], ['深度解释', '视差大近，视差小远。']],
      stepMeta: {
        left: ['左图', '参考视角。', 'I_L'],
        right: ['右图', '匹配视角。', 'I_R'],
        cost: ['匹配代价', '不同视差下的误差。', 'SAD(d)'],
        disparity: ['视差图', '亮处通常更近。', 'd=x_L-x_R'],
        depth: ['深度图', '由视差换算深度直觉。', 'Z=fB/d']
      }
    },
    frequency: {
      phase: '阶段三 · 中级视觉',
      title: '频域分析',
      english: 'Frequency Domain Analysis',
      tagline: '把图像拆成不同频率的波，低频对应大面积平滑，高频对应边缘、纹理和噪声。',
      status: '真实 NumPy 算法',
      difficulty: '进阶',
      endpoint: '/api/demo/frequency',
      formula: 'F(u,v)=\\sum_x\\sum_y f(x,y)e^{-j2\\pi(ux/M+vy/N)}',
      principles: ['空间域看像素位置，频域看变化速度。越靠中心通常越低频，越远离中心通常越高频。', '滤掉高频会模糊，保留高频会强调边缘和噪声。'],
      core: [['输入', '灰度图'], ['核心问题', '图像由哪些频率组成'], ['输出', '频谱和滤波效果']],
      pipeline: [['灰度化', '准备二维信号。'], ['FFT', '变换到频域。'], ['频谱中心化', '低频移动到中心便于观察。'], ['滤波重建', '观察低通/高通效果。']],
      stepMeta: {
        original: ['原图', '空间域输入。', 'f(x,y)'],
        spectrum: ['频谱', '亮处表示对应频率能量强。', '|F(u,v)|'],
        lowpass: ['低通结果', '保留平滑结构。', 'H_lowF'],
        highpass: ['高通结果', '强调边缘纹理。', 'H_highF']
      }
    }
  };

  const phaseLabel = {
    phase1: '阶段一 · 基础原语',
    phase2: '阶段二 · 经典特征检测',
    phase3: '阶段三 · 中级视觉',
    phase4: '阶段四 · 深度学习时代',
    phase5: '阶段五 · 基础模型与前沿感知'
  };

  function makeTeachingConfig(spec) {
    return {
      phase: phaseLabel[spec.phase] || spec.phase || '算法讲解',
      title: spec.title,
      english: spec.english || spec.title,
      tagline: spec.tagline,
      status: spec.status || '论文/参考实现讲解',
      difficulty: spec.difficulty || '进阶',
      formula: spec.formula || '',
      principles: spec.principles || [
        spec.tagline,
        '该页面先讲清输入、核心表征、优化目标和输出，再给出参考实现仓库，方便继续深入代码。'
      ],
      core: spec.core || [
        ['输入', spec.input || '图像、视频、文本或空间观测'],
        ['核心问题', spec.question || '如何把视觉信号转换成可用于任务的结构化表示'],
        ['输出', spec.output || '特征、预测、掩码、轨迹、3D结构或生成结果']
      ],
      pipeline: spec.pipeline || [
        ['输入编码', '把原始视觉信号整理成模型或算法可处理的张量。'],
        ['特征提取', '通过手写特征、卷积、Transformer 或几何约束得到中间表示。'],
        ['任务头/优化', '根据检测、分割、匹配、生成或3D重建目标进行预测或求解。'],
        ['结果解释', '把中间张量还原成边界框、掩码、深度、注意力图、点云或生成图像。']
      ],
      conceptSteps: spec.conceptSteps || spec.pipeline,
      references: spec.references || [],
      metrics: spec.metrics || {
        '展示类型': spec.status || '论文/参考实现讲解',
        '本地推理': '未接入',
        '参考实现': spec.references && spec.references.length ? '已列出' : '待补充'
      }
    };
  }

  [
    {
      id: 'shitomasi', phase: 'phase2', title: 'Shi-Tomasi 角点', english: 'Shi-Tomasi Corner',
      tagline: '用结构张量最小特征值选择角点，是光流跟踪中 goodFeaturesToTrack 的经典准则。',
      formula: 'R=\\min(\\lambda_1,\\lambda_2)',
      pipeline: [['梯度计算', '计算局部 Ix/Iy。'], ['结构张量', '在窗口内累计 Ix²、Iy²、IxIy。'], ['最小特征值', '用较小特征值判断两个方向都是否有变化。'], ['局部筛选', '阈值和 NMS 保留稳定角点。']]
    },
    {
      id: 'ncuts', phase: 'phase3', title: 'Normalized Cuts', english: 'Normalized Cuts',
      tagline: '把图像看成加权图，用谱聚类寻找既切断弱连接、又保持区域内部强连接的分割。',
      formula: 'Ncut(A,B)=cut(A,B)/assoc(A,V)+cut(A,B)/assoc(B,V)',
      pipeline: [['构图', '像素或超像素作为节点，颜色/空间相似度作为边权。'], ['拉普拉斯矩阵', '构建 D-W 和归一化形式。'], ['特征向量', '求第二小特征向量作为软分割方向。'], ['二值切分', '阈值化 Fiedler 向量得到区域。']]
    },
    {
      id: 'bovw_spm', phase: 'phase3', title: 'BoVW + SPM', english: 'Bag of Visual Words + SPM',
      tagline: '把局部 SIFT 描述子量化成视觉词，再用空间金字塔保留粗略布局，是手工特征分类时代的完整管线。',
      formula: 'K(x,y)=\\sum_i \\min(x_i,y_i)',
      pipeline: [['提取 SIFT', '从训练图像收集局部描述子。'], ['视觉词典', 'K-Means 聚成 codebook。'], ['硬分配直方图', '每张图统计视觉词频。'], ['空间金字塔', '1x1、2x2、4x4 多级网格加权拼接。'], ['SVM 分类', '用 Chi-square kernel 或线性近似分类。']]
    },
    {
      id: 'calibration', phase: 'phase3', title: '相机标定', english: 'Camera Calibration',
      tagline: '从棋盘格角点估计内参 K、外参 R/t 和畸变参数，把像素坐标连接到真实相机几何。',
      formula: 's\\begin{bmatrix}u\\\\v\\\\1\\end{bmatrix}=K[R|t]\\begin{bmatrix}X\\\\Y\\\\Z\\\\1\\end{bmatrix}',
      pipeline: [['检测棋盘格角点', '建立 3D 世界点和 2D 像素点对应。'], ['估计单应性', '每张平面标定图提供一个 H。'], ['求内参', '用约束解 K。'], ['求外参和畸变', '估计 R/t 并优化重投影误差。']]
    },
    {
      id: 'epipolar', phase: 'phase3', title: '对极几何', english: 'Epipolar Geometry',
      tagline: '用基础矩阵 F 或本质矩阵 E 描述双视图约束，把点匹配限制到对极线。',
      formula: 'p_2^T F p_1=0',
      pipeline: [['匹配点归一化', '把像素坐标中心化和缩放。'], ['八点法', '线性求解 F。'], ['秩约束', 'SVD 强制 rank(F)=2。'], ['恢复位姿', '由 E 分解得到 R/t 候选。']]
    },
    {
      id: 'sfm', phase: 'phase3', title: '三角测量与 SfM', english: 'Triangulation & SfM',
      tagline: '从多视图匹配、相机位姿和三角测量逐步恢复稀疏三维点云。',
      formula: 'x=P X,\\quad AX=0',
      pipeline: [['特征匹配', '用 SIFT/RANSAC 得到跨图对应点。'], ['估计相机运动', '基础矩阵/本质矩阵恢复 R/t。'], ['三角测量', '用两个投影矩阵线性求 3D 点。'], ['Bundle Adjustment', '联合优化相机和点云重投影误差。']]
    },
    {
      id: 'cnn_basics', phase: 'phase4', title: 'CNN 基础', english: 'CNN Basics',
      tagline: '从卷积、ReLU、池化到全连接，理解深度学习如何把手写卷积变成可学习特征。',
      formula: 'Y_{c}=\\sum_k W_{c,k}*X_k+b_c',
      pipeline: [['卷积层', '多个可学习 kernel 提取边缘和纹理。'], ['非线性', 'ReLU 保留正响应。'], ['池化/步幅', '降低分辨率并扩大感受野。'], ['分类头', '把空间特征聚合成类别概率。']],
      references: [{title:'LeNet-5 paper', url:'http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf'}]
    },
    {
      id: 'resnet', phase: 'phase4', title: 'ResNet + Grad-CAM', english: 'ResNet + Grad-CAM',
      tagline: '残差连接让超深网络可训练，Grad-CAM 用梯度加权激活图解释模型关注区域。',
      formula: 'y=F(x)+x,\\quad L^c=ReLU(\\sum_k\\alpha_k^c A^k)',
      pipeline: [['残差块', '学习相对输入的增量 F(x)。'], ['全局分类', '深层特征经过 GAP/FC 输出类别。'], ['梯度回传', '对目标类别求最后卷积层梯度。'], ['热力图', '加权激活并 ReLU 得到关注区域。']],
      references: [{title:'ResNet official repository', url:'https://github.com/KaimingHe/deep-residual-networks'}]
    },
    {
      id: 'fcn', phase: 'phase4', title: 'FCN', english: 'Fully Convolutional Network',
      tagline: '把分类网络改成全卷积结构，通过上采样和跳跃连接输出像素级语义。',
      formula: 'score=upsample(f_{deep})+f_{shallow}',
      pipeline: [['骨干网络', 'VGG/ResNet 提取多尺度特征。'], ['1x1 卷积', '把通道映射为类别 logits。'], ['转置卷积', '上采样回原图大小。'], ['跳跃融合', '结合浅层细节和深层语义。']],
      references: [{title:'FCN paper project', url:'https://github.com/shelhamer/fcn.berkeleyvision.org'}]
    },
    {
      id: 'unet', phase: 'phase4', title: 'U-Net', english: 'U-Net',
      tagline: '对称编码器-解码器和长跳跃连接让小数据医学分割也能保留边界细节。',
      formula: 'decoder_l=up(decoder_{l+1})\\oplus encoder_l',
      pipeline: [['编码器', '逐层下采样提取语义。'], ['瓶颈层', '获得全局上下文。'], ['解码器', '逐层上采样恢复分辨率。'], ['跳跃连接', '拼接同尺度浅层细节。']],
      references: [{title:'U-Net original page', url:'https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/'}]
    },
    {
      id: 'faster_rcnn', phase: 'phase4', title: 'Faster R-CNN + FPN', english: 'Faster R-CNN + FPN',
      tagline: '两阶段目标检测：先用 RPN 产生候选框，再对 RoI 分类和回归。',
      formula: 'L=L_{cls}+L_{box}+L_{rpn}',
      pipeline: [['Backbone/FPN', '提取多尺度特征金字塔。'], ['RPN', '用锚框预测 objectness 和候选框。'], ['RoIAlign/Pooling', '把候选区域裁成固定大小特征。'], ['检测头', '分类并精修边界框。']],
      references: [{title:'Detectron2 reference implementation', url:'https://github.com/facebookresearch/detectron2'}]
    },
    {
      id: 'yolo', phase: 'phase4', title: 'YOLO', english: 'You Only Look Once',
      tagline: '单阶段检测直接从网格或特征点回归边界框、类别和置信度，强调实时速度。',
      formula: 'box=(x,y,w,h,obj,class)',
      pipeline: [['特征提取', 'Backbone 提取多尺度图像特征。'], ['检测头', '每个位置预测框和类别。'], ['置信度筛选', '按 objectness 和 class score 过滤。'], ['NMS', '去除高度重叠框。']],
      references: [{title:'Ultralytics YOLO reference', url:'https://github.com/ultralytics/ultralytics'}]
    },
    {
      id: 'mask_rcnn', phase: 'phase4', title: 'Mask R-CNN', english: 'Mask R-CNN',
      tagline: '在 Faster R-CNN 上增加并行掩码分支，输出每个实例的像素级轮廓。',
      formula: 'L=L_{cls}+L_{box}+L_{mask}',
      pipeline: [['候选框检测', '沿用 Faster R-CNN 的 RPN 和 RoI。'], ['RoIAlign', '避免量化误差，精确采样特征。'], ['分类/框回归', '确定实例类别和框。'], ['掩码分支', '对每个 RoI 输出二值 mask。']],
      references: [{title:'Detectron2 Mask R-CNN', url:'https://github.com/facebookresearch/detectron2'}]
    },
    {
      id: 'gan', phase: 'phase4', title: 'GAN 基础', english: 'Generative Adversarial Network',
      tagline: '生成器和判别器相互博弈：一个试图生成逼真样本，另一个试图区分真假。',
      status: '教学模拟 + 真实概念讲解',
      formula: '\\min_G\\max_D E_x[\\log D(x)]+E_z[\\log(1-D(G(z)))]',
      pipeline: [['采样噪声', '从随机向量 z 开始。'], ['生成器', '把 z 映射成图像。'], ['判别器', '判断真实图和生成图。'], ['对抗更新', 'D 学会识别，G 学会欺骗 D。']],
      references: [{title:'DCGAN reference implementation', url:'https://github.com/pytorch/examples/tree/main/dcgan'}]
    },
    {
      id: 'diffusion', phase: 'phase4', title: '扩散模型基础', english: 'Diffusion Basics',
      tagline: '前向过程逐步加噪，反向网络学习一步步去噪生成样本。',
      status: '教学模拟 + 真实概念讲解',
      formula: 'x_t=\\sqrt{\\bar{\\alpha}_t}x_0+\\sqrt{1-\\bar{\\alpha}_t}\\epsilon',
      pipeline: [['前向加噪', '把图像逐步破坏为高斯噪声。'], ['时间编码', '告诉网络当前噪声强度。'], ['预测噪声', 'UNet/Transformer 预测 ε。'], ['反向采样', '从纯噪声迭代还原图像。']],
      references: [{title:'DDPM official repository', url:'https://github.com/hojonathanho/diffusion'}]
    },
    {
      id: 'vit', phase: 'phase5', title: 'ViT', english: 'Vision Transformer',
      tagline: '把图像切成 patch token，用 Transformer 自注意力建模全局关系。',
      formula: 'z_0=[x_{class};x_p^1E;...;x_p^NE]+E_{pos}',
      pipeline: [['Patchify', '把图像切成固定大小 patch。'], ['Token Embedding', '线性投影为 token 并加位置编码。'], ['Transformer Encoder', '多头自注意力建模全局依赖。'], ['CLS Head', '用 class token 做分类或迁移。']],
      references: [{title:'Google Research ViT', url:'https://github.com/google-research/vision_transformer'}]
    },
    {
      id: 'swin', phase: 'phase5', title: 'Swin Transformer', english: 'Swin Transformer',
      tagline: '窗口注意力和移位窗口让 Transformer 拥有层级特征和线性复杂度。',
      formula: 'MSA(W-MSA)\\rightarrow SW-MSA',
      references: [{title:'Official Swin Transformer', url:'https://github.com/microsoft/Swin-Transformer'}]
    },
    {
      id: 'dino', phase: 'phase5', title: 'DINO / DINOv2', english: 'Self-Supervised ViT',
      tagline: '用学生-教师自监督学习训练 ViT，注意力图常自然涌现物体区域。',
      formula: 'L=H(P_t(x_2),P_s(x_1))',
      references: [{title:'DINOv2 official repository', url:'https://github.com/facebookresearch/dinov2'}]
    },
    {
      id: 'mae', phase: 'phase5', title: 'MAE', english: 'Masked Autoencoder',
      tagline: '随机遮盖大部分 patch，只编码可见 token，再重建被遮盖像素。',
      formula: 'L=\\|x_{masked}-\\hat{x}_{masked}\\|^2',
      references: [{title:'MAE official repository', url:'https://github.com/facebookresearch/mae'}]
    },
    {
      id: 'detr', phase: 'phase5', title: 'DETR', english: 'DEtection TRansformer',
      tagline: '用 Object Query 和二分匹配把目标检测变成集合预测问题。',
      formula: '\\hat{y}=Transformer(f_{cnn}(x),Q),\\quad \\min_\\sigma \\sum_i L(y_i,\\hat{y}_{\\sigma(i)})',
      references: [{title:'DETR official repository', url:'https://github.com/facebookresearch/detr'}]
    },
    {
      id: 'dino_det', phase: 'phase5', title: 'DINO 检测', english: 'DINO Detection',
      tagline: '在 DETR 系列中加入对比去噪训练和混合 query，加快收敛并提升检测精度。',
      formula: 'L=L_{match}+L_{denoise}',
      references: [{title:'IDEA-Research DINO', url:'https://github.com/IDEA-Research/DINO'}]
    },
    {
      id: 'grdino', phase: 'phase5', title: 'Grounding DINO', english: 'Grounding DINO',
      tagline: '把文本短语作为条件，让检测框和开放词汇描述对齐。',
      formula: 'score=sim(q_{object},q_{text})',
      references: [{title:'Grounding DINO official repository', url:'https://github.com/IDEA-Research/GroundingDINO'}]
    },
    {
      id: 'mask2former', phase: 'phase5', title: 'Mask2Former', english: 'Mask2Former',
      tagline: '用掩码注意力统一语义、实例和全景分割任务。',
      formula: 'Attention(Q,K,V;M)',
      references: [{title:'Mask2Former official repository', url:'https://github.com/facebookresearch/Mask2Former'}]
    },
    {
      id: 'sam', phase: 'phase5', title: 'SAM', english: 'Segment Anything',
      tagline: '提示编码器 + 图像编码器 + 掩码解码器，实现点、框、文本等提示驱动的通用分割。',
      formula: 'mask=Decoder(ImageEncoder(I),PromptEncoder(p))',
      references: [{title:'Segment Anything official repository', url:'https://github.com/facebookresearch/segment-anything'}]
    },
    {
      id: 'sam2', phase: 'phase5', title: 'SAM 2', english: 'Segment Anything 2',
      tagline: '把 SAM 扩展到视频，使用记忆机制在时间上持续传播目标掩码。',
      formula: 'M_t=Decoder(F_t,Prompt,Memory_{<t})',
      references: [{title:'SAM 2 official repository', url:'https://github.com/facebookresearch/sam2'}]
    },
    {
      id: 'clip', phase: 'phase5', title: 'CLIP', english: 'Contrastive Language-Image Pre-training',
      tagline: '用海量图文对对比学习，把图像和文本映射到同一语义空间。',
      formula: 'L=-\\log\\frac{e^{sim(I,T^+)/\\tau}}{\\sum_j e^{sim(I,T_j)/\\tau}}',
      references: [{title:'OpenAI CLIP official repository', url:'https://github.com/openai/CLIP'}]
    },
    {
      id: 'blip2', phase: 'phase5', title: 'BLIP-2', english: 'BLIP-2',
      tagline: '用 Q-Former 桥接冻结视觉编码器和大语言模型，实现图像问答与描述。',
      formula: 'Q=QFormer(ImageTokens)',
      references: [{title:'LAVIS BLIP-2 reference', url:'https://github.com/salesforce/LAVIS'}]
    },
    {
      id: 'ddpm', phase: 'phase5', title: 'DDPM', english: 'Denoising Diffusion Probabilistic Models',
      tagline: '前向逐步加噪，反向学习去噪，从纯噪声逐步生成图像。',
      formula: 'q(x_t|x_{t-1})=\\mathcal{N}(\\sqrt{1-\\beta_t}x_{t-1},\\beta_tI)',
      references: [{title:'DDPM official repository', url:'https://github.com/hojonathanho/diffusion'}]
    },
    {
      id: 'sd', phase: 'phase5', title: 'Stable Diffusion', english: 'Latent Diffusion',
      tagline: '在 VAE 潜空间中做扩散，用文本 cross-attention 控制生成内容。',
      formula: '\\epsilon_\\theta(z_t,t,c)',
      references: [{title:'CompVis Stable Diffusion', url:'https://github.com/CompVis/stable-diffusion'}]
    },
    {
      id: 'controlnet', phase: 'phase5', title: 'ControlNet', english: 'ControlNet',
      tagline: '给冻结扩散模型增加可训练控制分支，让边缘、深度、姿态等条件精确控制生成。',
      formula: 'F_{control}=ZeroConv(F_{cond})+F_{frozen}',
      references: [{title:'ControlNet official repository', url:'https://github.com/lllyasviel/ControlNet'}]
    },
    {
      id: 'dit', phase: 'phase5', title: 'DiT', english: 'Diffusion Transformer',
      tagline: '用 Transformer 替代 UNet 做扩散模型的去噪网络。',
      formula: 'x_{patch}\\rightarrow TransformerBlocks\\rightarrow \\epsilon',
      references: [{title:'DiT official repository', url:'https://github.com/facebookresearch/DiT'}]
    },
    {
      id: 'flux', phase: 'phase5', title: 'Flux', english: 'Flow Matching Transformer',
      tagline: '把扩散/流匹配与大规模 Transformer 结合，学习从噪声到数据的连续路径。',
      formula: 'v_\\theta(x_t,t,c)\\approx \\frac{dx_t}{dt}',
      references: [{title:'Black Forest Labs Flux', url:'https://github.com/black-forest-labs/flux'}]
    },
    {
      id: 'stylegan', phase: 'phase5', title: 'StyleGAN', english: 'StyleGAN',
      tagline: '通过风格映射、调制卷积和逐层噪声生成高质量可控人脸/图像。',
      formula: 'w=f(z),\\quad y=Conv(Mod(W,w),x)',
      references: [{title:'StyleGAN3 official repository', url:'https://github.com/NVlabs/stylegan3'}]
    },
    {
      id: 'simclr', phase: 'phase5', title: 'SimCLR', english: 'SimCLR',
      tagline: '对同一图像的两种增强视图做对比学习，拉近正样本、推远负样本。',
      formula: 'L_{i,j}=-\\log\\frac{e^{sim(z_i,z_j)/\\tau}}{\\sum_k e^{sim(z_i,z_k)/\\tau}}',
      references: [{title:'SimCLR official repository', url:'https://github.com/google-research/simclr'}]
    },
    {
      id: 'moco', phase: 'phase5', title: 'MoCo', english: 'Momentum Contrast',
      tagline: '用动量编码器和队列维护大量负样本，解耦 batch size 与对比学习规模。',
      formula: 'k\\leftarrow m k +(1-m)q',
      references: [{title:'MoCo official repository', url:'https://github.com/facebookresearch/moco'}]
    },
    {
      id: 'byol', phase: 'phase5', title: 'BYOL', english: 'Bootstrap Your Own Latent',
      tagline: '不用负样本，通过 online/target 双网络和 EMA 避免表征坍塌。',
      formula: '\\theta_{target}\\leftarrow \\tau\\theta_{target}+(1-\\tau)\\theta_{online}',
      references: [{title:'BYOL reference', url:'https://github.com/deepmind/deepmind-research/tree/master/byol'}]
    },
    {
      id: 'ijepa', phase: 'phase5', title: 'I-JEPA', english: 'Image JEPA',
      tagline: '在表征空间预测被遮挡区域，不做像素级重建，学习更语义化的视觉表示。',
      formula: 'L=\\|Predictor(context)-Target(block)\\|',
      references: [{title:'I-JEPA official repository', url:'https://github.com/facebookresearch/ijepa'}]
    },
    {
      id: 'nerf', phase: 'phase5', title: 'NeRF', english: 'Neural Radiance Fields',
      tagline: '用 MLP 表示连续 3D 场，沿相机射线积分颜色和密度生成新视角。',
      formula: 'C(r)=\\int T(t)\\sigma(r(t))c(r(t),d)dt',
      references: [{title:'NeRF official repository', url:'https://github.com/bmild/nerf'}]
    },
    {
      id: '3dgs', phase: 'phase5', title: '3D Gaussian Splatting', english: '3D Gaussian Splatting',
      tagline: '用可优化的三维高斯椭球显式表示场景，实现高质量实时新视角合成。',
      formula: '\\alpha_i=1-\\exp(-\\sigma_i\\delta_i)',
      references: [{title:'3DGS official repository', url:'https://github.com/graphdeco-inria/gaussian-splatting'}]
    },
    {
      id: 'dust3r', phase: 'phase5', title: 'DUSt3R', english: 'DUSt3R',
      tagline: '无需相机参数，直接从图像对预测密集 3D 点图并对齐到同一空间。',
      formula: '(P_1,P_2,conf)=Transformer(I_1,I_2)',
      references: [{title:'DUSt3R official repository', url:'https://github.com/naver/dust3r'}]
    },
    {
      id: 'pointnet', phase: 'phase5', title: 'PointNet', english: 'PointNet',
      tagline: '用共享 MLP 和 max pooling 处理无序点云，建立点云深度学习入口。',
      formula: 'f(\\{x_i\\})=\\gamma(\\max_i h(x_i))',
      references: [{title:'PointNet official repository', url:'https://github.com/charlesq34/pointnet'}]
    },
    {
      id: 'orbslam3', phase: 'phase5', title: 'ORB-SLAM3', english: 'ORB-SLAM3',
      tagline: '跟踪、局部建图和回环检测三线程协作，从视频估计相机轨迹与地图。',
      formula: '\\min \\sum \\rho(\\|x_{ij}-\\pi(T_iX_j)\\|^2)',
      references: [{title:'ORB-SLAM3 official repository', url:'https://github.com/UZ-SLAMLab/ORB_SLAM3'}]
    },
    {
      id: 'bev', phase: 'phase5', title: 'BEV Perception', english: 'Bird-Eye-View Perception',
      tagline: '把多相机图像特征提升到三维再投影到鸟瞰平面，是自动驾驶空间感知核心。',
      formula: 'BEV=Pool(Lift(ImageFeatures,Depth))',
      references: [{title:'BEVFusion reference', url:'https://github.com/mit-han-lab/bevfusion'}]
    },
    {
      id: 'occupy', phase: 'phase5', title: 'Occupancy Networks', english: 'Occupancy Networks',
      tagline: '预测三维空间每个体素是否被占据，比 3D 框更细粒度地描述可通行空间。',
      formula: 'o=f_\\theta(x,z)\\in[0,1]',
      references: [{title:'Occupancy Networks reference', url:'https://github.com/autonomousvision/occupancy_networks'}]
    },
    {
      id: 'c3d', phase: 'phase5', title: 'C3D', english: '3D ConvNet',
      tagline: '用三维卷积同时在空间和时间上提取视频动作特征。',
      formula: 'Y(t,x,y)=\\sum_{\\tau,i,j}W(\\tau,i,j)X(t+\\tau,x+i,y+j)',
      references: [{title:'C3D project page', url:'https://github.com/facebookarchive/C3D'}]
    },
    {
      id: 'bytetrack', phase: 'phase5', title: 'ByteTrack', english: 'ByteTrack',
      tagline: '把低分检测框也纳入二阶段关联，减少遮挡或低置信目标的 ID 丢失。',
      formula: 'associate(high)\\rightarrow associate(low)',
      references: [{title:'ByteTrack official repository', url:'https://github.com/ifzhang/ByteTrack'}]
    },
    {
      id: 'botsort', phase: 'phase5', title: 'BoT-SORT', english: 'BoT-SORT',
      tagline: '在 ByteTrack 基础上加入相机运动补偿和 ReID 特征，提升多目标跟踪稳定性。',
      formula: 'cost=\\lambda IoU+(1-\\lambda)ReID',
      references: [{title:'BoT-SORT official repository', url:'https://github.com/NirAharon/BoT-SORT'}]
    },
    {
      id: 'deeppose', phase: 'phase5', title: 'DeepPose', english: 'DeepPose',
      tagline: '早期深度姿态估计方法，直接用 CNN 回归人体关键点坐标。',
      formula: '\\hat{y}=CNN(I)',
      references: [{title:'DeepPose paper', url:'https://arxiv.org/abs/1312.4659'}]
    },
    {
      id: 'openpose', phase: 'phase5', title: 'OpenPose', english: 'OpenPose',
      tagline: '自底向上预测关键点热力图和肢体亲和场，再做二分匹配组成人体实例。',
      formula: 'PAF(p)=unit(limb)',
      references: [{title:'OpenPose official repository', url:'https://github.com/CMU-Perceptual-Computing-Lab/openpose'}]
    },
    {
      id: 'mediapipe', phase: 'phase5', title: 'MediaPipe Pose', english: 'MediaPipe Pose',
      tagline: '面向实时移动端的人体姿态管线，输出 33 个关键点和可选 3D 坐标。',
      formula: 'landmarks=PoseLandmarker(I)',
      references: [{title:'MediaPipe official repository', url:'https://github.com/google-ai-edge/mediapipe'}]
    },
    {
      id: 'vitpose', phase: 'phase5', title: 'ViTPose', english: 'ViTPose',
      tagline: '用纯 ViT 骨干输出关键点热力图，展示 Transformer 在人体姿态上的强迁移能力。',
      formula: 'Heatmap=Head(ViT(I))',
      references: [{title:'ViTPose official repository', url:'https://github.com/ViTAE-Transformer/ViTPose'}]
    }
  ].forEach(function(spec) {
    if (!window.AlgorithmContent[spec.id]) {
      window.AlgorithmContent[spec.id] = makeTeachingConfig(spec);
    }
  });

  if (window.AlgorithmContent.sd && !window.AlgorithmContent.stable_diffusion) {
    window.AlgorithmContent.stable_diffusion = window.AlgorithmContent.sd;
  }
  if (window.AlgorithmContent.template_match && !window.AlgorithmContent.tpl_match) {
    window.AlgorithmContent.tpl_match = Object.assign({}, window.AlgorithmContent.template_match);
    window.AlgorithmContent.tpl_match.endpoint = '/api/demo/template_match';
  }

  const phaseOneVisualStories = {
    grayscale: {
      intro: '灰度不是去掉颜色，而是把 RGB 按人眼敏感度压成一个亮度通道。',
      cards: [
        {
          type: 'bars',
          title: '三种颜色贡献不一样',
          text: '绿色对人眼亮度影响最大，蓝色最小，所以灰度常用感知加权而不是简单平均。',
          items: [
            { label: 'R', value: 30, caption: '0.299', color: '#ef4444' },
            { label: 'G', value: 59, caption: '0.587', color: '#22c55e' },
            { label: 'B', value: 11, caption: '0.114', color: '#3b82f6' }
          ]
        },
        {
          type: 'pixels',
          title: '彩色像素变成亮度格',
          text: '每个像素从 [R,G,B] 变成一个 0-255 的亮度值，后续阈值、梯度和角点都更容易处理。',
          rows: 8,
          cols: 12,
          palette: ['#1e293b','#334155','#475569','#64748b','#94a3b8','#cbd5e1','#f8fafc']
        }
      ]
    },
    histogram: {
      intro: '直方图把图像从“在哪里亮”改成“有多少像素这么亮”。',
      cards: [
        {
          type: 'histogram',
          title: '亮度分布就是图像性格',
          text: '柱子挤在左边通常偏暗，挤在右边偏亮，范围很窄说明对比度不足。',
          values: [58, 72, 66, 50, 28, 14, 10, 8, 7, 10, 15, 24, 30, 25, 16, 9]
        },
        {
          type: 'bars',
          title: '均衡化像拉开橡皮筋',
          text: 'CDF 映射会把拥挤的亮度段拉开，让暗部和亮部使用更多可见灰度。',
          items: [
            { label: '原始动态范围', value: 38, caption: '窄', color: '#94a3b8' },
            { label: '均衡后范围', value: 92, caption: '宽', color: '#66f2c2' }
          ]
        }
      ]
    },
    threshold: {
      intro: '阈值化是在亮度轴上画一条线：线的一边是背景，另一边是前景。',
      cards: [
        {
          type: 'threshold',
          title: '一条线切开前景和背景',
          text: '阈值越低，保留下来的白色前景越多；阈值越高，只留下最亮区域。',
          position: 52
        },
        {
          type: 'histogram',
          title: 'Otsu 在两团像素之间找谷底',
          text: '当前景和背景形成两个峰时，Otsu 会选择让两类分得最开的阈值。',
          values: [8, 18, 48, 76, 58, 24, 8, 5, 7, 14, 36, 70, 74, 42, 18, 8]
        }
      ]
    },
    noise: {
      intro: '噪声是图像里的“不可信像素”：有的轻微抖动，有的直接坏成黑白点。',
      cards: [
        {
          type: 'pixels',
          title: '椒盐噪声像坏点',
          text: '少量像素突然变成纯黑或纯白，中值滤波通常能把这种极端值排除掉。',
          rows: 8,
          cols: 12,
          palette: ['#020617','#111827','#1e293b','#334155','#f8fafc']
        },
        {
          type: 'bars',
          title: '高斯噪声像连续抖动',
          text: '大多数像素只偏一点点，少数像素偏得较多，所以它更像传感器读数的随机误差。',
          items: [
            { label: '小扰动', value: 82, caption: '常见', color: '#66f2c2' },
            { label: '大扰动', value: 24, caption: '少见', color: '#f97316' }
          ]
        }
      ]
    },
    gaussian: {
      intro: '高斯模糊是温和的局部平均：中心像素说话最大声，越远声音越小。',
      cards: [
        {
          type: 'kernel',
          title: '中心权重大，边缘权重小',
          text: '3x3 或更大的高斯核会按距离分配权重，因此平滑噪声但不会像均值滤波那样生硬。',
          values: [1, 2, 1, 2, 4, 2, 1, 2, 1]
        },
        {
          type: 'window',
          title: '滑窗加权平均',
          text: '窗口扫过图像，每次用周围像素加权求和，sigma 越大，影响范围越宽。',
          values: [20, 38, 52, 72, 160, 86, 96, 112, 128]
        }
      ]
    },
    sobel: {
      intro: 'Sobel 关心“变化”：哪里亮度突然变了，哪里就可能有边缘。',
      cards: [
        {
          type: 'kernel',
          title: '差分核寻找方向变化',
          text: '一组核看左右变化得到 Gx，另一组看上下变化得到 Gy。',
          values: [-1, 0, 1, -2, 0, 2, -1, 0, 1]
        },
        {
          type: 'arrows',
          title: '两个方向合成边缘强度',
          text: 'Gx 和 Gy 像两个分力，合成后的幅值表示边缘强弱，角度表示变化方向。',
          labels: ['Gx', 'Gy', '|∇I|']
        }
      ]
    },
    median: {
      intro: '中值滤波不平均，而是排序后取中间值，所以特别擅长处理极端坏点。',
      cards: [
        {
          type: 'window',
          title: '极端值被挤到队尾',
          text: '窗口里即使有 240 这样的异常亮点，排序后中位数仍来自正常邻域。',
          values: [32, 35, 36, 38, 240, 39, 40, 42, 44]
        },
        {
          type: 'pixels',
          title: '去噪时更保边',
          text: '因为结果必须来自邻域已有像素，中值滤波比均值滤波更不容易把边缘抹灰。',
          rows: 8,
          cols: 12,
          palette: ['#020617','#111827','#1e293b','#f8fafc','#e2e8f0']
        }
      ]
    },
    bilateral: {
      intro: '双边滤波有两个条件：离得近、颜色像。两个都满足，才会参与平滑。',
      cards: [
        {
          type: 'bilateral',
          title: '不跨边缘乱平均',
          text: '边缘另一侧虽然空间上很近，但颜色差异大，所以权重会被压低。',
        },
        {
          type: 'bars',
          title: '空间核 × 颜色核',
          text: '普通高斯只看距离；双边滤波再乘一个颜色相似度，因此能平滑同质区域并保留边界。',
          items: [
            { label: '空间相近', value: 90, caption: '高', color: '#38bdf8' },
            { label: '颜色相似', value: 72, caption: '决定保边', color: '#66f2c2' },
            { label: '跨边缘权重', value: 18, caption: '低', color: '#f97316' }
          ]
        }
      ]
    }
  };

  Object.keys(phaseOneVisualStories).forEach(function(id) {
    if (window.AlgorithmContent[id]) {
      window.AlgorithmContent[id].visualStory = phaseOneVisualStories[id];
    }
  });

  const phaseTwoVisualStories = {
    shitomasi: {
      intro: '角点不是“亮点”，而是窗口向任意方向移动都会明显变差的位置。',
      cards: [
        { type: 'arrows', title: '两个方向都要有变化', text: '边缘只有一个方向变化明显；角点在 x/y 两个方向都变化明显，所以最小特征值也高。', labels: ['λ1', 'λ2', 'min'] },
        { type: 'threshold', title: '响应超过阈值才算角点', text: 'Shi-Tomasi 用 min(λ1,λ2) 做响应，阈值线扫过后只留下稳定位置。', position: 62 }
      ]
    },
    hough: {
      intro: 'Hough 的直觉是投票：图像空间的很多边缘点，一起支持参数空间里的同一条线。',
      cards: [
        { type: 'vote', title: '边缘点向参数空间投票', text: '每个边缘点对应很多可能直线；真正的直线会让投票在某个参数位置形成峰值。' },
        { type: 'histogram', title: '峰值就是共同证据', text: '累加器里越高的柱子，表示越多边缘点支持同一个几何模型。', values: [6, 12, 18, 28, 70, 34, 20, 16, 12, 58, 76, 38, 22, 14, 8, 5] }
      ]
    },
    morphology: {
      intro: '形态学把二值图当作形状集合，用一个结构元素在图上滑动来改变白色区域。',
      cards: [
        { type: 'morph', mode: 'erode', title: '腐蚀：形状向内收缩', text: '结构元素必须完全落在前景里才保留中心点，小噪点会被吃掉。' },
        { type: 'morph', mode: 'dilate', title: '膨胀：形状向外扩张', text: '结构元素只要碰到前景就扩张中心点，小断裂会被连接起来。' }
      ]
    },
    contour: {
      intro: '轮廓把像素块压缩成一条可测量的边界线。',
      cards: [
        { type: 'contour', title: '沿着前景边界行走', text: '算法只关心前景和背景交界处，把区域边界记录为点序列。' },
        { type: 'bars', title: '轮廓变成形状指标', text: '有了边界点，就能计算面积、周长、外接框和近似多边形。', items: [
          { label: '面积', value: 72, caption: '区域大小', color: '#66f2c2' },
          { label: '周长', value: 58, caption: '边界长度', color: '#38bdf8' },
          { label: '近似', value: 42, caption: '点数压缩', color: '#f97316' }
        ] }
      ]
    },
    nms: {
      intro: '非极大值抑制是在一串响应里只留下最强代表，删除旁边重复响应。',
      cards: [
        { type: 'nms', title: '只保留局部最高峰', text: '边缘响应通常是一条厚带；沿梯度方向比较后，只保留中心最大值。' },
        { type: 'threshold', title: '先变细，再阈值', text: 'NMS 负责去重变细，阈值负责过滤弱响应，两者合起来得到清晰边缘。', position: 58 }
      ]
    },
    template_match: {
      intro: '模板匹配像拿一张小卡片在大图上滑动，哪里最像，响应热图哪里最亮。',
      cards: [
        { type: 'template', title: '滑窗搜索整张图', text: '模板逐位置比较相似度，窗口移动时响应热图同步变化。' },
        { type: 'histogram', title: '最大响应给出目标位置', text: '响应图最高峰就是最相似的位置；多目标时可以取多个局部峰值。', values: [4, 8, 10, 16, 24, 38, 66, 82, 55, 32, 18, 12, 40, 72, 50, 16] }
      ]
    }
  };

  Object.keys(phaseTwoVisualStories).forEach(function(id) {
    if (window.AlgorithmContent[id]) {
      window.AlgorithmContent[id].visualStory = phaseTwoVisualStories[id];
    }
  });

  const refinedPhaseTwoVisualStories = {
    shitomasi: {
      intro: '第二阶段的关键是从“像素变化”走向“几何结构”：角点、直线、形状和局部峰值都要有明确判据。',
      cards: [
        { type: 'semanticAnim', anim: 'corner', title: '逐窗口计算结构张量响应', text: '画布会按扫描顺序移动窗口，并同步显示 lambda1、lambda2 和 min(lambda1, lambda2)。只有两个方向都强的位置才被保留为角点。', caption: 'structure tensor scan' },
        { type: 'threshold', title: '用最小特征值做稳定性门槛', text: 'Shi-Tomasi 不奖励单方向强边缘，而是看 min(lambda1, lambda2)。只有两个方向都稳定强响应的位置才留下。', position: 64 }
      ]
    },
    hough: {
      intro: 'Hough 不是凭空找线，而是让边缘点在参数空间投票：同一条真实直线会把票集中到同一个峰值。',
      cards: [
        { type: 'semanticAnim', anim: 'hough', title: '边缘点逐个投票到累加器', text: '左侧是图像空间的边缘点，右侧是 rho/theta 累加器。点越多，真实直线对应的参数峰值越明显。', caption: 'image -> accumulator' },
        { type: 'histogram', title: '峰值就是共同证据', text: '累加器高峰表示很多边缘点同意同一条几何模型；阈值只保留证据足够强的直线。', values: [6, 12, 18, 28, 70, 34, 20, 16, 12, 58, 76, 38, 22, 14, 8, 5] }
      ]
    },
    morphology: {
      intro: '形态学把二值图看成形状集合，结构元素像一把小尺子，在图上滑动并决定保留、删除或扩张。',
      cards: [
        { type: 'semanticAnim', anim: 'morph_erode', title: '腐蚀：逐位置判定“全命中”', text: '左边是输入二值形状，右边按扫描顺序显示输出。只有结构元素完全落在前景里，输出中心才保留。', caption: 'all hit -> keep' },
        { type: 'semanticAnim', anim: 'morph_dilate', title: '膨胀：逐位置判定“有命中”', text: '结构元素只要碰到任何前景，输出中心就点亮。因此目标会变粗，裂缝和小洞会被填上。', caption: 'any hit -> grow' }
      ]
    },
    contour: {
      intro: '轮廓不是整块像素，而是沿前景边界走出来的一串点；有了点序列，形状才能被测量和比较。',
      cards: [
        { type: 'semanticAnim', anim: 'contour', title: '边界追踪生成有序点列', text: '动画沿着前景边界逐点记录轮廓，右侧同步生成点序列。面积、周长、外接框都来自这条边界链。', caption: 'boundary -> ordered points' },
        { type: 'bars', title: '轮廓点产生形状指标', text: '轮廓把像素块变成可计算对象：面积看内部，周长看边界长度，近似多边形看结构复杂度。', items: [
          { label: 'area', value: 72, caption: 'region', color: '#66f2c2' },
          { label: 'perim', value: 58, caption: 'boundary', color: '#38bdf8' },
          { label: 'approx', value: 42, caption: 'vertices', color: '#f97316' }
        ] }
      ]
    },
    nms: {
      intro: '非极大值抑制不是普通阈值，而是沿响应方向做局部竞争：赢家留下，邻居被压暗。',
      cards: [
        { type: 'nms', title: '只保留局部最高峰', text: '厚边缘通常是一条响应带。NMS 沿梯度方向比较三个位置，只保留中间最高点，把边缘压成细线。' },
        { type: 'threshold', title: '先去重变细，再做强弱筛选', text: 'NMS 负责去掉重复响应，阈值负责过滤弱响应；两者配合才会得到清晰、稳定的边缘。', position: 58 }
      ]
    },
    template_match: {
      intro: '模板匹配像拿一张小卡片在大图上滑动，每个位置都计算相似度，响应图最高峰就是最可能的位置。',
      cards: [
        { type: 'semanticAnim', anim: 'template', title: '模板按扫描顺序生成响应图', text: '窗口从左到右、从上到下扫描大图；右下角响应图按同样顺序被填充，目标位置形成最亮峰值。', caption: 'raster scan response' },
        { type: 'histogram', title: '响应峰值给出目标坐标', text: '多个峰值可以表示多个相似目标；如果峰值不突出，说明模板和图像内容并不够匹配。', values: [4, 8, 10, 16, 24, 38, 66, 82, 55, 32, 18, 12, 40, 72, 50, 16] }
      ]
    }
  };

  Object.keys(refinedPhaseTwoVisualStories).forEach(function(id) {
    if (window.AlgorithmContent[id]) {
      window.AlgorithmContent[id].visualStory = refinedPhaseTwoVisualStories[id];
    }
  });

  const phaseThreeEnhancements = {
    kmeans: {
      controls: [{ name: 'k', label: '聚类数 K', type: 'range', min: 2, max: 8, step: 1, value: 4 }],
      visualStory: { intro: '阶段三开始从像素走向区域和结构，K-Means 是最直接的无监督分组入口。', cards: [
        { type: 'semanticAnim', anim: 'kmeans', title: '颜色点分配到最近中心', text: '像素被看成 RGB 空间里的点。每轮先按最近中心分配，再把中心移动到本簇均值。', caption: 'assign -> update' },
        { type: 'bars', title: 'K 控制分割粗细', text: 'K 越小，颜色区域越粗；K 越大，图像能保留更多颜色层次，但也更容易碎。', items: [
          { label: 'K小', value: 35, caption: '粗分区', color: '#38bdf8' },
          { label: 'K大', value: 82, caption: '细分区', color: '#fb7185' }
        ] }
      ] }
    },
    ncuts: {
      visualStory: { intro: 'Normalized Cuts 把图像看成图：节点是像素或超像素，边权表示相似度。', cards: [
        { type: 'semanticAnim', anim: 'ncuts', title: '切弱连接，保留强关联', text: '好的切分不是只让 cut 小，还要考虑各区域内部关联 assoc，避免切出孤立小碎片。', caption: 'cut / association' },
        { type: 'gradientSet', title: '谱向量像一条软分割轴', text: 'Fiedler 向量给每个节点一个连续值，再用阈值把软分割变成区域。', rows: [
          { label: 'v2', caption: 'soft partition', gradient: 'linear-gradient(90deg,#38bdf8,#e2e8f0,#fb7185)' }
        ] }
      ] }
    },
    watershed: {
      controls: [{ name: 'marker_distance', label: '种子间距', type: 'range', min: 4, max: 40, step: 1, value: 15 }],
      visualStory: { intro: '分水岭把梯度图当作地形：低处开始涨水，水盆相遇的山脊就是边界。', cards: [
        { type: 'semanticAnim', anim: 'watershed', title: '标记点向外淹没地形', text: '种子越密，区域越多；种子越稀，分割越粗。边缘梯度高的位置像山脊，会阻止区域合并。', caption: 'markers flood basins' },
        { type: 'threshold', title: '标记控制减少过分割', text: '如果每个小坑都当种子，结果会碎；用可靠前景/背景种子能让结果更稳定。', position: 46 }
      ] }
    },
    grabcut: {
      controls: [
        { name: 'x', label: '框 x', type: 'range', min: 0, max: 200, step: 5, value: 30 },
        { name: 'y', label: '框 y', type: 'range', min: 0, max: 200, step: 5, value: 30 },
        { name: 'w', label: '框宽', type: 'range', min: 40, max: 260, step: 5, value: 160 },
        { name: 'h', label: '框高', type: 'range', min: 40, max: 260, step: 5, value: 160 }
      ],
      visualStory: { intro: 'GrabCut 是典型交互式分割：用户给粗框，算法迭代细化前景和背景。', cards: [
        { type: 'semanticAnim', anim: 'grabcut', title: '矩形框初始化前景概率', text: '框外先当背景，框内是可能前景；随后颜色模型和边界平滑项一起决定最终 mask。', caption: 'rect -> model -> mask' },
        { type: 'bars', title: '能量函数的两股力量', text: '数据项看颜色像不像前景，平滑项鼓励相邻相似像素同类。', items: [
          { label: '颜色项', value: 72, caption: 'GMM', color: '#fb7185' },
          { label: '平滑项', value: 58, caption: '邻域', color: '#38bdf8' }
        ] }
      ] }
    },
    slic: {
      controls: [
        { name: 'num_superpixels', label: '超像素数', type: 'range', min: 50, max: 400, step: 10, value: 200 },
        { name: 'compactness', label: '紧凑度', type: 'range', min: 1, max: 30, step: 1, value: 10 }
      ],
      visualStory: { intro: 'SLIC 是局部 K-Means：同时考虑颜色接近和空间接近。', cards: [
        { type: 'semanticAnim', anim: 'slic', title: '中心只在局部窗口里移动', text: '每个中心只和附近像素竞争，所以比全局 K-Means 更快，也更容易形成紧凑小块。', caption: 'Lab + xy clustering' },
        { type: 'bars', title: 'compactness 平衡形状和边界', text: '紧凑度高时小块更规则；紧凑度低时更贴近颜色边界。', items: [
          { label: '颜色', value: 70, caption: '贴边界', color: '#66f2c2' },
          { label: '空间', value: 55, caption: '更规则', color: '#facc15' }
        ] }
      ] }
    },
    hog_svm: {
      controls: [
        { name: 'cell_size', label: 'cell 大小', type: 'range', min: 4, max: 16, step: 2, value: 8 },
        { name: 'num_bins', label: '方向桶数', type: 'range', min: 6, max: 12, step: 1, value: 9 }
      ],
      visualStory: { intro: 'HOG 把局部边缘方向统计成直方图，让形状比原始像素更稳定。', cards: [
        { type: 'semanticAnim', anim: 'hog', title: '梯度方向投票到方向桶', text: '每个 cell 统计 0-180 度方向分布，强边缘贡献更多票；block normalize 减少光照影响。', caption: 'gradients -> histogram' },
        { type: 'arrows', title: 'SVM 在 HOG 向量空间分类', text: 'HOG 负责描述形状，SVM 负责学习“像不像目标”的线性分界。', labels: ['HOG', 'SVM', 'score'] }
      ] }
    },
    bovw_spm: {
      visualStory: { intro: 'BoVW 把局部特征当作视觉单词，SPM 再把粗略空间布局拼进去。', cards: [
        { type: 'semanticAnim', anim: 'bovw', title: '描述子量化成视觉词', text: 'SIFT 描述子先被 codebook 量化为词频，再按 1x1、2x2、4x4 网格池化。', caption: 'descriptor -> word histogram' },
        { type: 'histogram', title: '直方图表示图像内容', text: '同类图像往往有相近视觉词频；SPM 保留“词大概出现在图像哪里”。', values: [12, 28, 52, 34, 70, 46, 20, 58, 36, 16, 44, 62] }
      ] }
    },
    optical_flow: {
      controls: [
        { name: 'shift_x', label: '模拟横向位移', type: 'range', min: -8, max: 8, step: 1, value: 2 },
        { name: 'shift_y', label: '模拟纵向位移', type: 'range', min: -8, max: 8, step: 1, value: 1 },
        { name: 'window_size', label: 'LK 窗口', type: 'range', min: 7, max: 31, step: 2, value: 15 }
      ],
      visualStory: { intro: '光流估计相邻帧中局部纹理的移动方向和速度。', cards: [
        { type: 'semanticAnim', anim: 'optical_flow', title: '亮度恒定约束求局部运动', text: 'Lucas-Kanade 在局部窗口里用 Ix、Iy、It 解 u/v，窗口太小不稳，太大又会混合不同运动。', caption: 'Ix u + Iy v + It = 0' },
        { type: 'arrows', title: '向量场表达运动', text: '箭头方向表示移动方向，长度表示速度；真实视频里常在角点或纹理点上估计稀疏光流。', labels: ['frame t', 'frame t+1', 'flow'] }
      ] }
    },
    calibration: {
      visualStory: { intro: '相机标定把 3D 世界点和 2D 像素点对应起来，求相机内参和畸变。', cards: [
        { type: 'semanticAnim', anim: 'calibration', title: '棋盘格角点约束相机参数', text: '每张标定图提供一组 3D 平面点到 2D 像素点的映射，优化目标是最小重投影误差。', caption: '3D corners -> 2D image' },
        { type: 'bars', title: '重投影误差越小越可信', text: '标定质量通常看平均重投影误差；误差大说明角点检测、模型或图片覆盖范围有问题。', items: [
          { label: 'K', value: 74, caption: '内参', color: '#38bdf8' },
          { label: 'dist', value: 42, caption: '畸变', color: '#fb7185' }
        ] }
      ] }
    },
    epipolar: {
      visualStory: { intro: '对极几何把双目匹配从“整张图找点”压缩成“沿一条线找点”。', cards: [
        { type: 'semanticAnim', anim: 'epipolar', title: '一个点对应另一幅图上的一条线', text: '基础矩阵 F 把左图点 p1 映射成右图对极线 l2，正确匹配点必须落在这条线上。', caption: 'p2^T F p1 = 0' },
        { type: 'threshold', title: 'RANSAC 去掉错误匹配', text: '真实匹配里会有离群点，RANSAC 用多数一致的几何约束估计 F。', position: 66 }
      ] }
    },
    stereo: {
      controls: [
        { name: 'max_disparity', label: '最大视差', type: 'range', min: 8, max: 64, step: 4, value: 32 },
        { name: 'block_size', label: '匹配窗口', type: 'range', min: 5, max: 15, step: 2, value: 7 }
      ],
      visualStory: { intro: '校正后的双目图里，同名点只沿水平线搜索，视差越大通常越近。', cards: [
        { type: 'semanticAnim', anim: 'stereo', title: '沿对极行做块匹配', text: '左图 patch 在右图同一行搜索最小 SAD/SSD 的位置，横向位移就是视差。', caption: 'row search -> disparity' },
        { type: 'gradientSet', title: '视差到深度是反比关系', text: 'Z=fB/d：视差大表示近，视差小表示远；无视差区域通常不可靠。', rows: [
          { label: 'd', caption: '近', gradient: 'linear-gradient(90deg,#2563eb,#22c55e,#f97316,#fb7185)' },
          { label: 'Z', caption: '远', gradient: 'linear-gradient(90deg,#fb7185,#f97316,#22c55e,#2563eb)' }
        ] }
      ] }
    },
    sfm: {
      visualStory: { intro: 'SfM 把多视图匹配、相机位姿和三角测量串起来，恢复稀疏三维结构。', cards: [
        { type: 'semanticAnim', anim: 'sfm', title: '两条相机射线交会成 3D 点', text: '已知两台相机位姿和匹配点，就能用三角测量求空间点；更多视图再联合优化。', caption: 'rays -> 3D point' },
        { type: 'bars', title: 'Bundle Adjustment 联合优化', text: 'BA 同时调整相机和点云，让所有投影点尽量贴近观测点。', items: [
          { label: 'pose', value: 58, caption: '相机', color: '#38bdf8' },
          { label: 'points', value: 72, caption: '点云', color: '#66f2c2' },
          { label: 'error', value: 28, caption: '误差', color: '#fb7185' }
        ] }
      ] }
    }
  };

  Object.keys(phaseThreeEnhancements).forEach(function(id) {
    if (!window.AlgorithmContent[id]) return;
    var patch = phaseThreeEnhancements[id];
    if (patch.controls) window.AlgorithmContent[id].controls = patch.controls;
    if (patch.visualStory) window.AlgorithmContent[id].visualStory = patch.visualStory;
  });

  const cannyStepVisualCards = {
    grayscale: [
      {
        type: 'cardAnim',
        anim: 'gray',
        title: '动态：彩色像素被压成亮度',
        text: '直接复用 Canny 流程里的灰度化动画：每个彩色格子不再保留 RGB 三个通道，而是按感知权重汇成一个亮度值。',
        caption: 'RGB -> luminance'
      }
    ],
    threshold: [
      {
        type: 'cardAnim',
        anim: 'binary',
        title: '动态：阈值线扫过亮度图',
        text: '亮度高于阈值的格子变成白色前景，低于阈值的格子变成黑色背景；阈值移动时，分割结果会立刻改变。',
        caption: 'gray -> binary mask'
      }
    ],
    gaussian: [
      {
        type: 'cardAnim',
        anim: 'gaussian',
        title: '动态：中心权重大，四周温柔参与',
        text: 'Canny 的第一步通常先高斯平滑。动画展示卷积窗口滑过图像时，中心像素影响最大，远处像素只轻轻参与平均。',
        caption: 'Gaussian weighted window'
      }
    ],
    sobel: [
      {
        type: 'cardAnim',
        anim: 'sobel_x',
        title: '动态：Gx 找左右方向的突变',
        text: '横向差分核一边减、一边加，像在问：左侧和右侧亮度是不是差很多？差很多的位置就是垂直边缘候选。',
        caption: 'Sobel X'
      },
      {
        type: 'cardAnim',
        anim: 'sobel_y',
        title: '动态：Gy 找上下方向的突变',
        text: '纵向差分核比较上方和下方的亮度变化，用来捕捉水平边缘候选。两个方向合在一起才能描述完整边缘。',
        caption: 'Sobel Y'
      },
      {
        type: 'cardAnim',
        anim: 'magnitude',
        title: '动态：Gx 与 Gy 合成边缘强度',
        text: '水平变化和垂直变化像两个分量，合成后的梯度幅值越亮，表示这里越可能是真正的边缘。',
        caption: '|gradient|'
      }
    ],
    nms: [
      {
        type: 'cardAnim',
        anim: 'nms',
        title: '动态：沿梯度方向只留下最高峰',
        text: 'Canny 中的 NMS 会把一条厚边缘压细：如果某个响应不是局部最高，就被压暗，只留下最像中心线的像素。',
        caption: 'thin edge ridge'
      }
    ]
  };

  Object.keys(cannyStepVisualCards).forEach(function(id) {
    var cfg = window.AlgorithmContent[id];
    if (!cfg) return;
    var story = cfg.visualStory || { cards: [] };
    var oldCards = story.cards || [];
    story.cards = cannyStepVisualCards[id].concat(oldCards);
    cfg.visualStory = story;
  });

  Object.keys(window.AlgorithmContent).forEach(function(id) {
    if (id !== 'common') attachImplementation(id, window.AlgorithmContent[id]);
  });

  window.AlgorithmContent.common = common;
})();
