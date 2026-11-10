"""Replay and explicit live inputs must give identical cold first-tick poses."""
import os,sys
from pathlib import Path
import cv2,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from realpc.io import Episode,read_depth_frames
from realpc.calib_rig import RigCalib
from realpc.config import load_config,build_arms
from realpc.diffcalib import TorchArm
from realpc.v4 import StreamConverter
os.environ.update(REALPC_COLD_COMPONENTS='1',REALPC_SIL='1',REALPC_CAM_EQUAL='1',REALPC_SIL_METHOD='contour')
torch.set_num_threads(2)
ep=Episode('/home/howardji/data_ext/agentwm_probe/hf/sweep/episode_0015');t=200
cal=RigCalib.load(ROOT/'configs/calib_yam_rig_v3.json');cfg=load_config(ROOT/'configs/yam_bimanual.json');r=cfg['robot'];arm=TorchArm(str(Path(cfg['_dir'])/r['urdf']),r['arm_joints'],r['finger_joints'])
seg=sys.argv[1];frames={};labels={}
for c in ('top','side'):
 D=read_depth_frames(str(Path(ep.path)/(c+'_depth.mkv')),[t])[t].astype(np.float32)*float(ep.meta['depth_scale'][c]);cap=cv2.VideoCapture(ep.video(c));cap.set(cv2.CAP_PROP_POS_FRAMES,t);ok,im=cap.read();assert ok;cap.release();frames[c]=(D,im)
 z=np.load(Path(seg)/ep.task/(ep.name+'_'+c+'.npz'));labels[c]=z['labels'][t-int(z['start'])]
 from realpc.v4 import load_library
 names=[o.name for o in load_library(str(ROOT/'assets/objects_v3/sweep'),10)[0]];lookup=np.arange(256,dtype=np.uint8)
 for i,n in enumerate(z['names'],1):lookup[i]=names.index(n)+1 if n in names else 0
 labels[c]=lookup[labels[c]]
q={a:cal.q_at(ep,a,cal.robot_tick(ep,'top',t)) for a in ('left','right')}
results=[]
for live in (False,True):
 np.random.seed(123);torch.manual_seed(123)
 conv=StreamConverter(ep,cal,arm,build_arms(cfg,ep.calib),str(ROOT/'assets/objects_v3/sweep'),start=t,cold=True,seg_dir=None if live else seg)
 results.append(conv.step(t,frames,labels=labels if live else None,q=q if live else None))
for name,T in results[0]['obj_T'].items():
 assert np.allclose(T,results[1]['obj_T'][name],atol=1e-6),(name,np.max(abs(T-results[1]['obj_T'][name])))
assert results[0]['obj_held']==results[1]['obj_held']
print('PASS: replay and live current RGB-D/labels/joints yield identical cold-start poses and attachments')
