#!/usr/bin/env python
"""Quick CURRENT-frame joint-pose probe; baseline causal poses are initial guesses.
This is a correction ablation, not a closed-loop tracker evaluation.
"""
import sys,argparse,json,time
from pathlib import Path
import cv2,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from realpc.v4 import load_library
from realpc.meshtrack import MeshTracker,CamView
from realpc.calib_rig import RigCalib
from realpc.io import Episode
from realpc.diffcalib import TorchArm
from realpc.config import load_config
from object_overlay import overlay
p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--ticks',nargs='+',type=int,default=[0,150,300,450]);p.add_argument('--lib',default='assets/objects_v3');p.add_argument('--steps',type=int,default=20);p.add_argument('--out',required=True);a=p.parse_args();torch.set_num_threads(2)
out=Path(a.out);out.mkdir(parents=True,exist_ok=True);ep=Episode('/home/howardji/data_ext/agentwm_probe/hf/'+a.task+'/episode_0000');cal=RigCalib.load(ROOT/'configs/calib_yam_rig_v3.json');d=np.load('/home/howardji/data_ext/realpc_out/v1/'+a.task+'/episode_0000.npz');obs,_=load_library(str(ROOT/a.lib/a.task));cfg=load_config(ROOT/'configs/yam_bimanual.json');r=cfg['robot'];arm=TorchArm(str(Path(cfg['_dir'])/r['urdf']),r['arm_joints'],r['finger_joints']);labels={c:np.load('/home/howardji/data_ext/realpc_v4/seg/'+a.task+'/episode_0000_'+c+'.npz')['labels'] for c in ['top','side']};records=[];maps={o.name:(o.verts_t,o.faces_t) for o in obs}
for t in a.ticks:
 poses={str(n):d['obj_T'][i,t].astype(float) for i,n in enumerate(d['obj_names'])};trk=MeshTracker(obs,poses,arm,cfg={'sil_steps':a.steps});views=[]
 for c in ['top','side']:
  H,W=(720,1280) if c=='top' else (480,640);views.append(CamView(c,np.zeros((H,W),np.float32),ep.intrinsics(c),cal.T_world_cam(c),labels=labels[c][t]))
 q={s:d['q_'+s][t] for s in ['left','right']};rob=trk.robot(q,{'left':np.eye(4),'right':cal.T_world_rb()});rz=[]
 for v in views:
  zs=[v.zbuf(r['V'],arm.faces).cpu().numpy() for r in rob.values()];rz.append(np.minimum.reduce([np.where(z>0,z,np.inf) for z in zs]))
 corrected={};row={'t':t,'objects':{}};tic=time.time()
 for i,o in enumerate(obs):
  T,ib,ia=trk.sil_refine(o,poses[o.name],views,rz,i+1);corrected[o.name]=poses[o.name] if T is None else T;row['objects'][o.name]={'before':ib,'after':ia,'T':corrected[o.name].tolist()}
 row['seconds']=time.time()-tic;records.append(row);print(t,{k:(round(v['before'],3),round(v['after'],3)) for k,v in row['objects'].items()},'seconds',round(row['seconds'],1),flush=True)
 before=overlay(ep,cal,maps,poses,t);after=overlay(ep,cal,maps,corrected,t);cv2.putText(before,'BEFORE',(10,55),0,1,(255,255,255),2);cv2.putText(after,'JOINT CORRECTION',(10,55),0,1,(255,255,255),2);cv2.imwrite(str(out/f'{a.task}_{t:04d}.jpg'),np.concatenate([before,after],0));json.dump(records,open(out/'metrics.json','w'),indent=2)
