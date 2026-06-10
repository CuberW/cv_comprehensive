"""
路由中枢。
- 根路径返回主外壳页面
- /api/modules 返回模块注册信息（供前端构建导航）
- 遍历所有已注册模块，挂载各自的 API 端点
"""
from flask import Blueprint, render_template, jsonify, current_app
from app.modules import MODULE_REGISTRY

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def index():
    """主入口 —— 返回 SPA 外壳。"""
    return render_template('index.html')


@main_bp.route('/api/modules')
def api_modules():
    """返回所有已注册模块的元信息，前端据此构建侧边栏导航。"""
    cats = {}
    for mid, cls in MODULE_REGISTRY.items():
        cat = cls.category or 'other'
        cats.setdefault(cat, []).append({
            'id': cls.module_id,
            'name': cls.name,
            'name_en': cls.name_en,
            'difficulty': cls.difficulty,
            'description': cls.description,
            'dependencies': cls.dependencies or [],
            'page': cls.get_page() if hasattr(cls, 'get_page') else None,
        })
    return jsonify({
        'categories': cats,
        'total': len(MODULE_REGISTRY),
    })


# ---- 遍历注册所有模块的 API 端点 ----
for _mid, _cls in MODULE_REGISTRY.items():
    if hasattr(_cls, 'get_api_endpoints'):
        for ep in _cls.get_api_endpoints():
            main_bp.add_url_rule(
                ep['rule'],
                endpoint=ep.get('endpoint'),
                view_func=ep['handler'],
                methods=ep.get('methods', ['GET']),
            )
