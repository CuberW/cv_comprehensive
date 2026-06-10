"""
算法模块基类。
所有 CV 算法模块必须继承 AlgorithmModule 并实现其抽象方法。
"""

class AlgorithmModule:
    """
    一个模块代表一个完整的 CV 算法教学单元。
    包含：路由注册、API 端点、页面信息。
    """
    # 子类必须覆盖 ——
    module_id: str = ''          # 唯一标识，如 'grayscale'
    name: str = ''               # 中文名，如 '灰度转换'
    name_en: str = ''            # 英文名
    category: str = ''           # 分类: seeing / understanding / transforming / generating
    difficulty: int = 1          # 难度 1-4
    description: str = ''        # 一句话描述
    dependencies: list = None    # 前置模块 id 列表

    def __init_subclass__(cls, **kwargs):
        """子类定义时自动注册到模块注册表。"""
        super().__init_subclass__(**kwargs)
        if cls.module_id:
            from app.modules import register_module
            register_module(cls)

    @staticmethod
    def get_page():
        """返回模块对应的 HTML 页面文件名（相对于 static/pages/）。"""
        raise NotImplementedError

    @staticmethod
    def get_api_endpoints():
        """
        返回该模块的 API 端点列表，每个端点为 dict:
          { 'rule': '/api/xxx', 'methods': ['POST'], 'handler': function }
        由 routes.py 统一注册。
        """
        return []
