from app.modules.base import AlgorithmModule
class 高斯模糊Module(AlgorithmModule):
    module_id='gaussian';name='高斯模糊';name_en='Gaussian Blur'
    phase='phase1_fundamentals';difficulty=1
    description='高斯核的数学构造,sigma与窗口尺寸的关系。尺度空间基石。'
    dependencies=['convolution']
    @staticmethod
    def get_page(): return 'gaussian.html'
