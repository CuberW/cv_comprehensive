
from app.modules.phase2_classical.edge.processor import build_canny_pipeline
def build_pipeline(upload_path):
    return build_canny_pipeline(upload_path)
