"""Collect stand-colored depth across the episode, with fixed v3 calibration.
Offline shape fitting only; not an input to online pose/cloth tracking.
"""
import sys
from pathlib import Path
import numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from realpc.io import Episode,read_depth_frames,backproject
from realpc.calib_rig import RigCalib
EP=Episode('/home/howardji/data_ext/agentwm_probe/hf/hang/episode_0000');CAL=RigCalib.load(ROOT/'configs/calib_yam_rig_v3.json');d=np.load('/home/howardji/data_ext/astra_meshes/iter16/converted/hang/episode_0000.npz');oi=list(d['obj_names']).index('stand');ts=list(range(0,499,20))+[498];pts=[];cols=[];tags=[]
for c in ['top','side']:
 depth=read_depth_frames(str(Path(EP.path)/(c+'_depth.mkv')),ts);cap=cv2.VideoCapture(EP.video(c));K=EP.intrinsics(c)
 for t in ts:
  D=depth[t].astype(np.float32)*float(EP.meta['depth_scale'][c]);cap.set(cv2.CAP_PROP_POS_FRAMES,t);ok,im=cap.read();assert ok
  yy,xx=np.mgrid[0:D.shape[0]:2,0:D.shape[1]:2];z=D[yy,xx];P=np.stack([(xx-K[0,2])*z/K[0,0],(yy-K[1,2])*z/K[1,1],z],-1).reshape(-1,3);C=im[yy,xx][:,:,::-1].reshape(-1,3)
  M=CAL.T_world_cam(c);P=P@M[:3,:3].T+M[:3,3];T=d['obj_T'][oi,t];P=(P-T[:3,3])@T[:3,:3]
  hsv=cv2.cvtColor(C[None].astype(np.uint8),cv2.COLOR_RGB2HSV)[0]
  keep=(np.abs(P[:,0])<.115)&(np.abs(P[:,1])<.12)&(P[:,2]>-.10)&(P[:,2]<.24)&(hsv[:,0]>12)&(hsv[:,0]<35)&(hsv[:,1]>30)&(hsv[:,1]<190)&(hsv[:,2]>35)
  pts.append(P[keep]);cols.append(C[keep]);tags.append(np.tile([t,0 if c=='top' else 1],(keep.sum(),1)))
 cap.release()
P=np.concatenate(pts);C=np.concatenate(cols);tags=np.concatenate(tags);out=Path('/home/howardji/data_ext/astra_cloth/stand00/stand_multiview.npz');np.savez_compressed(out,P=P,C=C,tags=tags);print(len(P),out)
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,3,figsize=(15,5))
for a,(i,j) in zip(ax,[(0,2),(1,2),(0,1)]):
 k=P[:,2]>-.03;a.scatter(P[k,i],P[k,j],c=C[k]/255,s=.2);a.set_aspect('equal');a.set_xlabel('xyz'[i]);a.set_ylabel('xyz'[j]);a.grid()
fig.savefig(out.with_suffix('.png'),dpi=150)
