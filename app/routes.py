"""
Route hub.
- Root path returns the main shell page
- /api/modules returns module registry info (for frontend navigation)
- Iterates all registered modules and mounts their API endpoints
"""
from flask import Blueprint, render_template, jsonify
from app.modules import MODULE_REGISTRY, get_modules_by_phase

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def index():
    """Main entry point -- returns the SPA shell."""
    return render_template('index.html')


@main_bp.route('/api/modules')
def api_modules():
    """
    Return all registered module metadata organized by phase.
    Frontend uses this to build the sidebar navigation.
    """
    phases = get_modules_by_phase()
    # Add page info to each module
    for phase in phases:
        for mod in phase['modules']:
            cls = MODULE_REGISTRY.get(mod['id'])
            if cls and hasattr(cls, 'get_page'):
                mod['page'] = cls.get_page()

    total = len(MODULE_REGISTRY)
    return jsonify({
        'phases': phases,
        'total': total,
    })


# ---- Register all module API endpoints ----
for _mid, _cls in list(MODULE_REGISTRY.items()):
    if hasattr(_cls, 'get_api_endpoints'):
        for ep in _cls.get_api_endpoints():
            main_bp.add_url_rule(
                ep['rule'],
                endpoint=ep.get('endpoint'),
                view_func=ep['handler'],
                methods=ep.get('methods', ['GET']),
            )
