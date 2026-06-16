from app.modules.base import AlgorithmModule
class 中值滤波Module(AlgorithmModule):
    module_id='median';name='中值滤波';name_en='Median Filter'
    phase='phase1_fundamentals';difficulty=1
    description='非线性去噪,对椒盐噪声特效。与高斯模糊形成线性vs非线性对比。'
    dependencies=['noise']
    @staticmethod
    def get_page(): return 'median.html'
