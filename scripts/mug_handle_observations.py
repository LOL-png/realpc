"""Offline multi-view mug shape observations; no effect on causal pose runtime."""
import sys,json
from pathlib import Path
import numpy as np,cv2,trimesh
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from realpc.io import Episode,read_depth_frames
from realpc.calib_rig import RigCalib
from realpc.config import load_config,build_arms
OUT=Path('/home/howardji/data_ext/astra_mug_handle/observations');REPORT=Path('/home/howardji/point_tracker/reports/ASTRA_Q29_1109/mug_handle_review')
ep=Episode('/home/howardji/data_ext/agentwm_probe/hf/hang/episode_0000');cal=RigCalib.load(ROOT/'configs/calib_yam_rig_v3.json');arms=build_arms(load_config(ROOT/'configs/yam_bimanual.json'),ep.calib);arms.T_base['right']=cal.T_world_rb()
track=np.load('/home/howardji/data_ext/astra_rigid/hang_v2stand/converted/hang/episode_0000.npz');oi=list(track['obj_names']).index('mug');mesh=trimesh.load(ROOT/'assets/objects_v3/hang/mug.ply',process=False)
ticks=sorted(set(list(range(0,499,20))+[150,175,210,250,350,420,498]));index=[]
A=np.eye(4);A[:3,:3]=[[0,1,0],[1,0,0],[0,0,-1]]
for cam in ['top','side','left_wrist','right_wrist']:
 D=read_depth_frames(ep.path+'/'+cam+'_depth.mkv',ticks);cap=cv2.VideoCapture(ep.video(cam));tiles=[]
 for t in ticks:
  cap.set(cv2.CAP_PROP_POS_FRAMES,t);ok,im=cap.read();assert ok
  dep=D[t].astype(np.float32)*float(ep.meta['depth_scale'][cam]);K=ep.intrinsics(cam).copy();T=track['obj_T'][oi,t].astype(float)
  if 'wrist' in cam:
   arm=cam.split('_')[0];q={s:cal.q_at(ep,s,t-cal.const_lag) for s in ['left','right']};LP=arms.link_poses(q);Twc=LP[arms.bodies.index((arm,'gripper'))]@A@np.asarray(ep.calib['X_'+arm])
  else:Twc=cal.T_world_cam(cam)
  Toc=np.linalg.inv(Twc)@T;P=np.asarray(mesh.vertices)@Toc[:3,:3].T+Toc[:3,3];P=P[P[:,2]>.06]
  if len(P)<30:continue
  uv=P[:,:2]/P[:,2:]*[K[0,0],K[1,1]]+[K[0,2],K[1,2]];h,w=dep.shape
  lo=np.floor(uv.min(0)-35).astype(int);hi=np.ceil(uv.max(0)+35).astype(int);lo=np.maximum(lo,0);hi=np.minimum(hi,[w,h])
  if (hi-lo).min()<15:continue
  x,y=lo;xx,yy=hi;rgb=im[y:yy,x:xx];dd=dep[y:yy,x:xx];kc=K.copy();kc[:2,2]-=lo
  hsv=cv2.cvtColor(rgb,cv2.COLOR_BGR2HSV);mask=((hsv[:,:,0]<28)|(hsv[:,:,0]>175))&(hsv[:,:,1]>90)&(hsv[:,:,2]>40)
  # Reject unrelated orange clutter using measured depth near the mug pose.
  vv,uu=np.mgrid[:len(dd),:dd.shape[1]];pc=np.stack([(uu-kc[0,2])*dd/kc[0,0],(vv-kc[1,2])*dd/kc[1,1],dd],-1);po=(pc-Toc[:3,3])@Toc[:3,:3]
  gate=np.linalg.norm(po,axis=-1)<.14;mask &= gate|(dd<.08)
  fn=f'{cam}_{t:04d}.npz';np.savez_compressed(OUT/fn,rgb=rgb,D=dd,K=kc,Tco=Toc,Twc=Twc,T=T,mask=mask,crop=lo,t=t,cam=cam)
  scale=min(300/rgb.shape[1],270/rgb.shape[0]);tile=np.full((300,320,3),245,np.uint8);r=cv2.resize(rgb,None,fx=scale,fy=scale);tile[28:28+r.shape[0],:r.shape[1]]=r;cv2.putText(tile,f'{cam} t={t}',(8,20),0,.55,(0,0,0),1);tiles.append(tile);index.append(dict(file=fn,t=t,cam=cam,color_pixels=int(mask.sum())))
 cap.release()
 while len(tiles)%5:tiles.append(np.full_like(tiles[0],245))
 cv2.imwrite(str(REPORT/f'{cam}_raw_contact.jpg'),np.vstack([np.hstack(tiles[i:i+5]) for i in range(0,len(tiles),5)]));print(cam,len(tiles),flush=True)
(OUT/'index.json').write_text(json.dumps(index,indent=2))
