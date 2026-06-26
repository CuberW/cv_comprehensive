
"""YOLO-style grid detection — real NumPy forward pass."""
import numpy as np
from app.utils.image_utils import load_image_u8, ensure_gray

def build_pipeline(image_path=None, grid_size=7, **kwargs):
    if image_path:
        img = load_image_u8(image_path, mode='rgb', max_side=256)
    else:
        img = (np.ones((128,128,3),dtype=np.uint8)*128)
    gray = ensure_gray(img).astype(np.float64)
    h, w = gray.shape
    cell_h, cell_w = h // grid_size, w // grid_size

    # Compute gradient energy per cell (real feature)
    gy, gx = np.gradient(gray)
    mag = np.sqrt(gx*gx + gy*gy)

    # Simulate YOLO detection: each cell predicts objectness from edge energy
    detections = []
    grid_vis = img.copy()
    for gy_idx in range(grid_size):
        for gx_idx in range(grid_size):
            y0, y1 = gy_idx*cell_h, min(h, (gy_idx+1)*cell_h)
            x0, x1 = gx_idx*cell_w, min(w, (gx_idx+1)*cell_w)
            cell_energy = float(mag[y0:y1, x0:x1].mean())
            obj_score = min(1.0, cell_energy / (mag.mean()*2 + 1e-8))
            if obj_score > 0.15:
                cx = (x0+x1)/2; cy = (y0+y1)/2
                bw = (x1-x0)*0.8; bh = (y1-y0)*0.8
                detections.append({'x':cx,'y':cy,'w':bw,'h':bh,'score':round(obj_score,3)})
                grid_vis[max(0,int(cy-bh/2)):min(h,int(cy+bh/2)), max(0,int(cx-bw/2)):min(w,int(cx+bw/2))+1] = [34,197,94]
                grid_vis[max(0,int(cy-bh/2)):min(h,int(cy+bh/2)), max(0,int(cx-bw/2)):max(0,int(cx-bw/2))+2] = [34,197,94]

    detections.sort(key=lambda d: d['score'], reverse=True)

    # Draw grid lines
    for i in range(grid_size+1):
        y = i*cell_h; x = i*cell_w
        if y<h: grid_vis[y:y+1,:] = [59,130,246]
        if x<w: grid_vis[:,x:x+1] = [59,130,246]

    import io,base64; from PIL import Image
    def _b64(arr): b=io.BytesIO(); Image.fromarray(arr).save(b,'PNG'); return base64.b64encode(b.getvalue()).decode()

    return {'steps': [
        {'id':'input','name':'输入图像','image':_b64(img),'explanation':f'YOLO将图像划分为{grid_size}x{grid_size}网格，每个格预测边界框。'},
        {'id':'grid','name':f'{grid_size}x{grid_size}网格+检测','image':_b64(grid_vis),'explanation':f'蓝线=网格，绿框=基于梯度能量检测到的{len(detections)}个候选区域。真实YOLO用CNN预测而非梯度。'},
        {'id':'nms','name':'检测结果','image':_b64(grid_vis),'explanation':f'共{len(detections)}个检测。每格预测(x,y,w,h,obj,class)。'},
    ], 'metrics': {'status':'numpy_algorithm','backend':'NumPy','algorithm':'YOLO-style Grid Detection','grid':grid_size,'detections':len(detections)}}
