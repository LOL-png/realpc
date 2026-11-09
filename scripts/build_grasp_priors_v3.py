#!/usr/bin/env python
"""Fixed grasp hypotheses from OTHER recordings; excludes evaluation episodes 0/15/30/45.
Used only as cold-start hypotheses, verified against current RGB-D; never as pose truth.
"""
import sys,json
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from realpc.config import load_config
from realpc.diffcalib import TorchArm
from realpc.calib_rig import RigCalib
cfg=load_config(ROOT/'configs/yam_bimanual.json');r=cfg['robot'];arm=TorchArm(str(Path(cfg['_dir'])/r['urdf']),r['arm_joints'],r['finger_joints']);cal=RigCalib.load(ROOT/'configs/calib_yam_rig_v3.json');pool={};episodes=[1,2,3,4,6,7,8,9,10,11,12,13,14,16,17,18,19,20]
for e in episodes:
 z=np.load(f'/home/howardji/data_ext/realpc_out/v1/sweep/episode_{e:04d}.npz')
 for name in ['broom','dustpan']:
  i=list(z['obj_names']).index(name)
  for side,code in [('left',1),('right',2)]:
   good=np.flatnonzero((z['obj_held'][i]==code)&(z['obj_res'][i,:,0]<.005))
   if len(good)<25:continue
   picks=good[np.linspace(15,len(good)-1,4).astype(int)]
   q=torch.tensor(z['q_'+side][picks],device='cuda');G=arm.link_tf(q)['gripper'].cpu().numpy();base=np.eye(4) if side=='left' else cal.T_world_rb()
   for k,g in zip(picks,G):
    U=np.linalg.inv(base@g)@z['obj_T'][i,k]
    if np.linalg.norm(U[:3,3])<.30:pool.setdefault(name,{}).setdefault(side,[]).append({'T_grip_obj':U.tolist(),'episode':e,'tick':int(k)})
for name in pool:
 for side,items in pool[name].items():
  X=np.array([np.r_[np.array(x['T_grip_obj'])[:3,3],np.array(x['T_grip_obj'])[:3,:3].ravel()*.06] for x in items]);chosen=[int(np.argmin(np.linalg.norm(X-np.median(X,axis=0),axis=1)))]
  while len(chosen)<min(10,len(items)):
   dist=np.min(np.linalg.norm(X[:,None]-X[chosen][None],axis=2),axis=1);dist[chosen]=-1;chosen.append(int(dist.argmax()))
  pool[name][side]=[items[k] for k in chosen]
p=ROOT/'assets/objects_v3/sweep/grasp_priors.json';json.dump({'source':'v1 causal tracked poses, hypotheses only','source_episodes':episodes,'excluded_evaluation_episodes':[0,15,30,45],'grasps':pool},open(p,'w'),indent=2);print({n:{a:len(v) for a,v in arms.items()} for n,arms in pool.items()})
