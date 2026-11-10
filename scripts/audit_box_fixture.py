"""Two-camera box audit at start/middle/end of every selected episode.
Current color/depth residuals are automatic proxies, not hand-labelled IoU.
"""
import sys,json,argparse
from pathlib import Path
import cv2,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from realpc.io import Episode,read_depth_frames
from realpc.calib_rig import RigCalib
from realpc.objlib import depth_points,table_crop
from realpc.box_fixture import cardboard_points,refine_box_pose,surface_residual,update_box_pose,refine_box_silhouette
from realpc.v4 import load_library
from realpc.diffcalib import Renderer,TorchArm
from realpc.config import load_config
from realpc.meshtrack import MeshTracker,CamView
ap=argparse.ArgumentParser();ap.add_argument('--ticks',nargs='+',type=int);ap.add_argument('--refine',action='store_true');ap.add_argument('--sil',action='store_true');ap.add_argument('--episodes',nargs='+',type=int,default=list(range(50)));ap.add_argument('--fits',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();torch.set_num_threads(2)
out=Path(a.out);out.mkdir(parents=True,exist_ok=True);cal=RigCalib.load(ROOT/'configs/calib_yam_rig_v3.json');obs,metas=load_library(ROOT/'assets/objects_v3/sweep',100);o=next(o for o in obs if o.name=='box');dims=np.array(metas['box']['dimensions_m']);records=[]
cfg=load_config(ROOT/'configs/yam_bimanual.json');r=cfg['robot'];arm=TorchArm(str(Path(cfg['_dir'])/r['urdf']),r['arm_joints'],r['finger_joints']);trk=MeshTracker([o],{'box':np.eye(4)},arm)
for i in a.episodes:
 ep=Episode(f'/home/howardji/data_ext/agentwm_probe/hf/sweep/episode_{i:04d}');z=np.load(Path(a.fits)/(ep.name+'.npz'));T=z['T'];ts=a.ticks or [0,(len(ep.low['t_cap'])-1)//2,len(ep.low['t_cap'])-1];region=json.load(open(Path(ep.path)/'table_region.json'));per={t:[] for t in ts};images={t:[] for t in ts}
 poses={t:T.copy() for t in ts};sildata={}
 if a.refine:
  packets={t:[] for t in ts};frames={t:{} for t in ts}
  for c in ['top','side']:
   dep0=read_depth_frames(str(Path(ep.path)/(c+'_depth.mkv')),ts);cap0=cv2.VideoCapture(ep.video(c))
   for t in ts:
    cap0.set(cv2.CAP_PROP_POS_FRAMES,t);_,im0=cap0.read();D0=dep0[t].astype(float)*float(ep.meta['depth_scale'][c]);P0,C0,_,_=depth_points(D0,im0,ep.intrinsics(c),cal.T_world_cam(c),stride=3);packets[t].append((P0,C0));frames[t][c]=(D0,im0)
   cap0.release()
  Tcur=T.copy()
  for t in ts:
   for _ in range(3):Tcur,detail=update_box_pose(Tcur,np.concatenate([x[0] for x in packets[t]]),np.concatenate([x[1] for x in packets[t]]),dims,metas['box'].get('wall_planes'))
   if a.sil:
    q={side:cal.q_at(ep,side,cal.robot_tick(ep,'top',t)) for side in ('left','right')};rob=trk.robot(q,{'left':np.eye(4),'right':cal.T_world_rb()});views=[];rz=[]
    for c,(D0,im0) in frames[t].items():
     v=CamView(c,D0,ep.intrinsics(c),cal.T_world_cam(c),rgb=im0);views.append(v);zs=[v.zbuf(rr['V'],arm.faces).cpu().numpy() for rr in rob.values()];rz.append(np.minimum.reduce([np.where(z>0,z,1e9) for z in zs]))
    Tcur,detail=refine_box_silhouette(o,Tcur,views,rz);sildata[t]=detail
   poses[t]=Tcur.copy()
 for cam in ['top','side']:
  dep=read_depth_frames(str(Path(ep.path)/(cam+'_depth.mkv')),ts);cap=cv2.VideoCapture(ep.video(cam));K=ep.intrinsics(cam);Twc=cal.T_world_cam(cam);Tc=np.linalg.inv(Twc);Kl=K.copy();Kl[:2]*=640/(1280 if cam=='top' else 640);H=360 if cam=='top' else 480;ren=Renderer(Kl,H,640)
  for t in ts:
   T=poses[t]
   V=o.verts_t@torch.tensor(T[:3,:3].T,device='cuda',dtype=torch.float32)+torch.tensor(T[:3,3],device='cuda',dtype=torch.float32);_,zz=ren.raster((V@torch.tensor(Tc[:3,:3].T,device='cuda',dtype=torch.float32)+torch.tensor(Tc[:3,3],device='cuda',dtype=torch.float32))[None],o.faces_t);zz=zz[0].cpu().numpy()
   cap.set(cv2.CAP_PROP_POS_FRAMES,t);ok,im=cap.read();assert ok;D=dep[t].astype(float)*float(ep.meta['depth_scale'][cam]);P,C,_,_=depth_points(D,im,K,Twc,stride=2);keep=table_crop(P,region,*ep.table_plane(),above=-.04);P,C=P[keep],C[keep]
   try:Q,_=cardboard_points(P,C)
   except ValueError:Q=np.zeros((0,3))
   L=(Q-T[:3,3])@T[:3,:3];near=(np.abs(L[:,0])<dims[0]/2+.035)&(np.abs(L[:,1])<dims[1]/2+.035)&(L[:,2]>-.01)&(L[:,2]<dims[2]+.025);Q,L=Q[near],L[near];per[t].append(Q)
   resid=surface_residual(L,dims);Ds=cv2.resize(D,(640,H),interpolation=0);valid=(zz>0)&(Ds>.1)&(Ds>=zz-.015);dr=np.abs(Ds[valid]-zz[valid]);rec={'episode':ep.name,'tick':t,'camera':cam,'observed_points':len(Q),'surface_med_mm':float(np.median(resid)*1000) if len(resid) else None,'surface_p90_mm':float(np.percentile(resid,90)*1000) if len(resid) else None,'within_10mm':float((resid<.010).mean()) if len(resid) else None,'visible_render_pixels':int(valid.sum()),'depth_med_mm':float(np.median(dr)*1000) if len(dr) else None};rec['pass']=None if len(Q)<=100 else bool(len(Q)>100 and rec['within_10mm']>=.9 and len(dr)>100 and rec['depth_med_mm']<5);rec['material_silhouette']=sildata.get(t,{}).get('per_camera',{}).get(cam);records.append(rec)
   if i in [0,6,10,20,23,25,30,45,49]:
    im=cv2.resize(im,(640,H));ov=im.copy();m=(zz>0)&(Ds>.1)&(Ds>=zz-.015);cont,_=cv2.findContours(m.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(ov,cont,-1,(0,210,255),1);tile=np.concatenate([im,ov],1);cv2.putText(tile,f'{ep.name} t={t} {cam} box depth med {rec["depth_med_mm"]:.1f}mm',(8,22),0,.6,(255,255,255),2);images[t].append(cv2.copyMakeBorder(tile,0,480-H,0,0,0))
  cap.release()
 moves=[]
 for t in ts:
  T=poses[t];Q=np.concatenate(per[t]);Tn,detail=refine_box_pose(T,Q,dims);shift=float(np.linalg.norm(Tn[:3,3]-T[:3,3])*1000);moves.append(shift)
  for rec in records:
   if rec['episode']==ep.name and rec['tick']==t:rec['current_refit_shift_mm']=shift;rec['pose_from_start_mm']=float(np.linalg.norm(poses[t][:3,3]-poses[ts[0]][:3,3])*1000);rec['table_normal_change_deg']=float(np.degrees(np.arccos(np.clip(poses[t][:3,2]@poses[ts[0]][:3,2],-1,1))));rec['table_height_change_mm']=float((poses[t][:3,3]-poses[ts[0]][:3,3])@poses[ts[0]][:3,2]*1000)
  if images[t]:cv2.imwrite(str(out/f'{ep.name}_{t:04d}.jpg'),np.concatenate(images[t]))
 print(ep.name,'shift_mm',np.round(moves,2).tolist(),flush=True)
 (out/'metrics.json').write_text(json.dumps({'definition':'pass: >=90% material depth points within 10mm of mesh planes; visible render median depth <5mm; >=100 points/pixels; automatic current-observation proxy, not labelled IoU','records':records},indent=2))
