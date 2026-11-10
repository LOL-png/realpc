"""Current RGB-D + stored pose overlays in all four calibrated cameras."""
import sys,argparse
from pathlib import Path
import cv2,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from realpc.io import Episode,read_depth_frames
from realpc.calib_rig import RigCalib
from realpc.config import load_config,build_arms
from realpc.diffcalib import Renderer
from realpc.v4 import load_library
p=argparse.ArgumentParser();p.add_argument('--episode',type=int,required=True);p.add_argument('--ticks',nargs='+',type=int,required=True);p.add_argument('--converted');p.add_argument('--box-fit');p.add_argument('--out',required=True);a=p.parse_args()
ep=Episode(f'/home/howardji/data_ext/agentwm_probe/hf/sweep/episode_{a.episode:04d}');cal=RigCalib.load(ROOT/'configs/calib_yam_rig_v3.json');cfg=load_config(ROOT/'configs/yam_bimanual.json');arms=build_arms(cfg,ep.calib);arms.T_base['right']=cal.T_world_rb();obs,_=load_library(ROOT/'assets/objects_v3/sweep',100)
d=np.load(a.converted) if a.converted else None;bf=np.load(a.box_fit) if a.box_fit else None
out=Path(a.out);out.mkdir(parents=True,exist_ok=True);rows={t:[] for t in a.ticks}
for c in ['top','side','left_wrist','right_wrist']:
 cap=cv2.VideoCapture(ep.video(c));dep=read_depth_frames(str(Path(ep.path)/(c+'_depth.mkv')),a.ticks);K=ep.intrinsics(c).copy();K[:2]*=640/(1280 if c=='top' else 640);H=360 if c=='top' else 480;ren=Renderer(K,H,640)
 for t in a.ticks:
  cap.set(cv2.CAP_PROP_POS_FRAMES,t);ok,im=cap.read();assert ok;im=cv2.resize(im,(640,H));ov=im.copy()
  q={s:cal.q_at(ep,s,cal.robot_tick(ep,'top',t)) for s in ['left','right']};LP=arms.link_poses(q)
  if 'wrist' in c:
   s=c.split('_')[0];G=LP[arms.bodies.index((s,'gripper'))];A=np.eye(4);A[:3,:3]=[[0,1,0],[1,0,0],[0,0,-1]];Twc=G@A@np.asarray(ep.calib['X_'+s])
  else:Twc=cal.T_world_cam(c)
  Tc=np.linalg.inv(Twc);Tct=torch.tensor(Tc,device='cuda',dtype=torch.float32);D=cv2.resize(dep[t].astype(float)*float(ep.meta['depth_scale'][c]),(640,H),interpolation=0)
  for o in obs:
   if o.name=='box' and bf is not None:T=bf['T'];color=(0,220,255)
   elif d is not None and o.name=='broom':T=d['obj_T'][list(d['obj_names']).index(o.name),int(np.argmin(abs(d['ticks']-t)))];color=(255,50,255)
   else:continue
   T=torch.tensor(T,device='cuda',dtype=torch.float32);V=o.verts_t@T[:3,:3].T+T[:3,3];Vc=V@Tct[:3,:3].T+Tct[:3,3]
   if (Vc[:,2]>.05).sum()<3:continue
   try:_,z=ren.raster(Vc[None],o.faces_t)
   except IndexError:continue
   z=z[0].cpu().numpy();m=(z>0)&((D<.1)|(z<D+.015));cs,_=cv2.findContours(m.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(ov,cs,-1,color,1)
  row=np.concatenate([im,ov],axis=1);cv2.putText(row,f'ep{a.episode:02d} t{t} {c} | raw / mesh',(8,24),0,.65,(255,255,255),2);row=cv2.copyMakeBorder(row,0,480-H,0,0,0);rows[t].append(row)
 cap.release()
for t,r in rows.items():cv2.imwrite(str(out/f'ep{a.episode:02d}_{t:04d}.jpg'),np.concatenate(r))
