"""Local GPU behavioral checks for current-frame evidence and full occlusion."""
import sys,os
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from realpc.meshtrack import MeshTracker,CamView
from realpc.v4 import load_library
from realpc.calib_rig import RigCalib
from realpc.io import Episode
from realpc.config import load_config
from realpc.diffcalib import TorchArm
os.environ['REALPC_SIL_METHOD']='contour';os.environ['REALPC_BROOM_MATERIAL']='1';torch.set_num_threads(2)
cal=RigCalib.load(ROOT/'configs/calib_yam_rig_v3.json');ep=Episode('/home/howardji/data_ext/agentwm_probe/hf/sweep/episode_0000');cfg=load_config(ROOT/'configs/yam_bimanual.json');r=cfg['robot'];arm=TorchArm(str(Path(cfg['_dir'])/r['urdf']),r['arm_joints'],r['finger_joints']);o=next(x for x in load_library(str(ROOT/'assets/objects_v3/sweep'))[0] if x.name=='broom');d=np.load('/home/howardji/data_ext/realpc_out/v1/sweep/episode_0000.npz');T=d['obj_T'][list(d['obj_names']).index('broom'),250].astype(float);trk=MeshTracker([o],{o.name:T},arm,{'sil_refine':True});bases={'left':np.eye(4),'right':cal.T_world_rb()};q={s:d['q_'+s][250].copy() for s in ['left','right']};q['left'][6]=.1
rob=trk.robot(q,bases);st=trk.state[o.name];st.held='left';st.grip_at_grasp=.1;st.T_grip_obj=np.linalg.inv(rob['left']['G'])@T
views=[]
for c in ['top','side']:
 H,W=(720,1280) if c=='top' else (480,640);views.append(CamView(c,np.zeros((H,W),np.float32),ep.intrinsics(c),cal.T_world_cam(c),labels=np.zeros((H,W),np.uint8)))
q['left'][0]+=.07;expected=trk.robot(q,bases)['left']['G']@st.T_grip_obj
actual=trk.step(251,views,q,bases)[o.name];assert np.allclose(actual,expected,atol=1e-7),np.max(abs(actual-expected));assert st.mode=='fk';print('PASS: no masks/depth in both views -> exact calibrated gripper FK')
# Neither blank view may create a visual correction.
Tn,b,a=trk.sil_refine(o,T,views,[np.full((v.R.H,v.R.W),np.inf) for v in views],1);assert Tn is None and b==a==0;print('PASS: absent evidence -> no pose correction')
# The refiner is a stateless current-frame operation: repeating it is deterministic.
z=np.load('/home/howardji/data_ext/realpc_v4/seg/sweep/episode_0000_side.npz');views[1].labels=z['labels'][250];rz=[np.full((v.R.H,v.R.W),np.inf) for v in views]
A=trk.sil_refine(o,T,views,rz,1);B=trk.sil_refine(o,T,views,rz,1);assert np.allclose(A[0],B[0],atol=1e-9);assert A[2]>=A[1];print('PASS: side camera alone supports a deterministic non-regressing correction',A[1:])
