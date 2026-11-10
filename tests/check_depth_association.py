"""Behavioral checks: table/robot rejection and no correction without depth."""
import os,sys
from pathlib import Path
os.environ['REALPC_DEPTH_MESH']='1';os.environ['REALPC_BROOM_MATERIAL']='0'
import numpy as np,torch,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from realpc.meshtrack import MeshTracker,CamView
from realpc.v4 import load_library
from realpc.calib_rig import RigCalib
from realpc.io import Episode
from realpc.config import load_config
from realpc.diffcalib import TorchArm
torch.set_num_threads(2)
ep=Episode('/home/howardji/data_ext/agentwm_probe/hf/sweep/episode_0000');cal=RigCalib.load(ROOT/'configs/calib_yam_rig_v3.json');cfg=load_config(ROOT/'configs/yam_bimanual.json');r=cfg['robot'];arm=TorchArm(str(Path(cfg['_dir'])/r['urdf']),r['arm_joints'],r['finger_joints']);o=next(o for o in load_library(ROOT/'assets/objects_v3/sweep') [0] if o.name=='broom');d=np.load('/home/howardji/data_ext/astra_rigid/planar_final/converted/sweep/episode_0000.npz');i=list(d['obj_names']).index('broom');T=d['obj_T'][i,250].astype(float);q={s:d['q_'+s][250].copy() for s in ('left','right')};q['left'][6]=.1;bases={'left':np.eye(4),'right':cal.T_world_rb()};trk=MeshTracker([o],{'broom':T},arm);trk.plane=ep.table_plane();rob=trk.robot(q,bases);st=trk.state['broom'];st.held='left';st.grip_at_grasp=.1;st.T_grip_obj=np.linalg.inv(rob['left']['G'])@T
views=[]
for c in ('top','side'):
 H,W=(720,1280) if c=='top' else (480,640);views.append(CamView(c,np.zeros((H,W),np.float32),ep.intrinsics(c),cal.T_world_cam(c)))
q['left'][0]+=.05;expected=trk.robot(q,bases)['left']['G']@st.T_grip_obj;actual=trk.step(251,views,q,bases)['broom'];assert np.allclose(actual,expected,atol=1e-7);assert trk.depth_diagnostics['broom']['visible']==0;assert all(not (v.labels==1).any() for v in views);print('PASS: absent depth preserves exact held FK and does not signal visible failure')
from realpc.depth_masks import associate_depth
n,offset=ep.table_plane();v=views[0];H,W=v.D.shape;yy,xx=np.mgrid[:H,:W];K=v.K;rays=np.stack([(xx-K[0,2])/K[0,0],(yy-K[1,2])/K[1,1],np.ones_like(xx)],-1)@v.Twc[:3,:3].T
z=-(np.asarray(n)@v.Twc[:3,3]+offset)/(rays@np.asarray(n));v.D=np.where(z>.1,z,0).astype(np.float32);v.labels=None;P=o.verts_t@torch.tensor(st.T[:3,:3].T,device='cuda',dtype=torch.float32)+torch.tensor(st.T[:3,3],device='cuda',dtype=torch.float32);oz={'broom':[v.zbuf(P,o.faces_t).cpu().numpy()]};associate_depth(trk,[v],[np.full((v.R.H,v.R.W),1e9)],oz);assert not (v.labels==1).any();print('PASS: rendered object footprint cannot turn table-plane depth into object points')
v.D=cv2.resize(oz['broom'][0],(W,H),interpolation=0);v.labels=None;associate_depth(trk,[v],[np.full((v.R.H,v.R.W),1e9)],oz);assert (v.labels==1).sum()>40
v.labels=None;v.rgb=np.zeros((H,W,3),np.uint8)
trk.lib_points={'broom':(None,None)};trk.depth_color={'broom':lambda C:np.zeros(len(C))}
associate_depth(trk,[v],[np.full((v.R.H,v.R.W),1e9)],oz)
assert (v.labels==1).sum()>40;assert len(v.masked_points(1))>40
print('PASS: zero colour likelihood cannot erase valid mesh-associated depth')
v.labels=None;rz=np.where(oz['broom'][0]>0,oz['broom'][0]-.005,1e9);associate_depth(trk,[v],[rz],oz);assert not (v.labels==1).any();print('PASS: foreground robot-render depth excludes otherwise valid object returns')
