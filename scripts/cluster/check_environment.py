"""Run on the allocated GPU, before conversion; no local GPU work required."""
import argparse,json,os,sys,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
import numpy as np
import torch,cv2,open3d,trimesh,pytorch3d,ipctk,triangle,osqp,clarabel,sam2
assert torch.cuda.is_available(), 'GPU unavailable'
assert torch.cuda.get_device_capability()[0]>=8, 'Ampere or newer required'
from pytorch3d.ops import knn_points
x=torch.rand(1,32,3,device='cuda');knn_points(x,x,K=1);torch.cuda.synchronize()
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from realpc.cloth_metric import project
y=np.array([[0.,0.,0.],[1.,0.,0.]],np.float64)
assert project(y,np.array([[0,1]]),np.array([1.]),np.ones(2),.02,1) is not None, 'Native cloth compiler unavailable'
subprocess.run(['ffmpeg','-version'],check=True,stdout=subprocess.DEVNULL)
out=dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),
         capability=torch.cuda.get_device_capability(),numpy=np.__version__,opencv=cv2.__version__,
         open3d=open3d.__version__,host=os.uname().nodename)
Path(a.out).write_text(json.dumps(out,indent=2));print(out)
