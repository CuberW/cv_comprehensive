from app.modules.base import AlgorithmModule
class 双边滤波Module(AlgorithmModule):
    module_id='bilateral';name='双边滤波';name_en='Bilateral Filter'
    phase='phase1_fundamentals';difficulty=1
    description='保边平滑,空间+色彩双高斯核。SLIC、GrabCut等分割算法的前置。'
    dependencies=['gaussian']
    @staticmethod
    def get_page(): return 'bilateral.html'
