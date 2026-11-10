"""Render actual saved fixed-identity points over both RGB cameras (CPU only)."""
import argparse,sys,subprocess
from pathlib import Path
import numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from realpc.io import Episode,DepthStream
from realpc.calib_rig import RigCalib
p=argparse.ArgumentParser();p.add_argument('--converted',required=True);p.add_argument('--out',required=True);p.add_argument('--stride',type=int,default=4);a=p.parse_args();d=np.load(a.converted);ep=Episode('/home/howardji/data_ext/agentwm_probe/hf/'+str(d['task'])+'/'+str(d['episode']));cal=RigCalib.load(ROOT/'configs/calib_yam_rig_v3.json');out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.tmp.mp4');writer=cv2.VideoWriter(str(tmp),cv2.VideoWriter_fourcc(*'mp4v'),24,(1280,960));ticks=d['ticks'];chosen=set(range(0,len(ticks),a.stride));chosen.add(len(ticks)-1);stills={min(chosen,key=lambda x:abs(x-v)) for v in np.linspace(0,len(ticks)-1,4)};sheets=[]
cams=['top','side'];caps={c:cv2.VideoCapture(ep.video(c)) for c in cams};ds={c:DepthStream(str(Path(ep.path)/(c+'_depth.mkv')),int(ticks[0])) for c in cams};current=int(ticks[0])-1;palette={'box':(0,200,255),'broom':(255,120,0),'debris':(0,255,120),'dustpan':(220,70,220),'mug':(0,200,255),'stand':(255,120,0)}
for c in cams:caps[c].set(cv2.CAP_PROP_POS_FRAMES,int(ticks[0]))
for k,t in enumerate(ticks):
 frames={}
 for c in cams:
  _,D=ds[c].next();ok,im=caps[c].read();assert ok;frames[c]=(D,im)
 if k not in chosen:continue
 B=d['robot_body'].astype(int);RT=d['robot_T'][k,B];P=[np.einsum('nij,nj->ni',RT[:,:3,:3],d['robot_local'])+RT[:,:3,3]];C=[np.tile([170,170,170],(len(P[0]),1))]
 for i,name in enumerate(d['obj_names']):
  if not d['obj_valid'][i,k]:continue
  L=d['obj_local'][d['obj_index']==i];T=d['obj_T'][i,k];P.append(L@T[:3,:3].T+T[:3,3]);C.append(np.tile(palette.get(str(name),(200,200,0)),(len(L),1)))
 P.append(d['static_points']);C.append(d['static_color'][:,::-1]);P=np.concatenate(P);C=np.concatenate(C).astype(np.uint8);tiles=[]
 for c in cams:
  D,im=frames[c];H=360 if c=='top' else 480;im=cv2.resize(im,(640,H));ov=im.copy();Tc=np.linalg.inv(cal.T_world_cam(c));Q=P@Tc[:3,:3].T+Tc[:3,3];K=ep.intrinsics(c).copy();K[:2]*=640/(1280 if c=='top' else 640);z=Q[:,2];uv=np.rint(Q[:,:2]/np.maximum(z[:,None],.01)*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]).astype(int);valid=np.isfinite(Q).all(1)&(z>.08)&(uv[:,0]>=0)&(uv[:,0]<640)&(uv[:,1]>=0)&(uv[:,1]<H);j=np.flatnonzero(valid);u,v=uv[j].T;dd=cv2.resize(D.astype(float)*float(ep.meta['depth_scale'][c]),(640,H),interpolation=0);zbuf=np.full((H,640),np.inf);np.minimum.at(zbuf,(v,u),z[j]);keep=(z[j]<=zbuf[v,u]+.003)&((dd[v,u]<.1)|(z[j]<dd[v,u]+.015));j=j[keep];u,v=uv[j].T;ov[v,u]=(.15*ov[v,u]+.85*C[j]).astype(np.uint8)
  tile=np.concatenate([im,ov],1);cv2.putText(tile,f'{ep.task} {ep.name}  {c}  frame {t}: camera / extracted points',(8,26),0,.65,(255,255,255),2);tiles.append(cv2.copyMakeBorder(tile,0,480-H,0,0,0))
 frame=np.concatenate(tiles);writer.write(frame)
 if k in stills:sheets.append(cv2.resize(frame,(640,480)))
writer.release()
for c in cams:caps[c].release();ds[c].close()
subprocess.run(['ffmpeg','-y','-v','error','-i',str(tmp),'-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(out)],check=True);tmp.unlink()
if len(sheets)==4:cv2.imwrite(str(out.with_suffix('.jpg')),np.vstack([np.hstack(sheets[:2]),np.hstack(sheets[2:])]))
print(out,'display frames',len(chosen))
