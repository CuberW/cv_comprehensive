(function() {
  const common = {
    uploadHint: '上传一张图片后，页面会调用当前项目中的 NumPy 实现，并把中间结果逐步展开。',
  };

  window.AlgorithmContent = {
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

  window.AlgorithmContent.common = common;
})();
