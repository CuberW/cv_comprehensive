"""
模块注册表。
所有算法模块通过 AlgorithmModule.__init_subclass__ 自动注册到这里。
"""

MODULE_REGISTRY = {}  # module_id → AlgorithmModule 子类


def register_module(cls):
    """将一个 AlgorithmModule 子类加入全局注册表。"""
    from app.modules.base import AlgorithmModule
    if not issubclass(cls, AlgorithmModule):
        raise TypeError(f'{cls.__name__} 必须继承 AlgorithmModule')
    if not cls.module_id:
        raise ValueError(f'{cls.__name__} 必须定义 module_id')
    MODULE_REGISTRY[cls.module_id] = cls


def get_modules_by_category():
    """按分类组织模块，供前端导航使用。"""
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
        })
    return cats
