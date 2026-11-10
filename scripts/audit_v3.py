#!/usr/bin/env python
"""Efficient joint-camera audit: render once per camera/tick, score both mask sources.
Independent Q28 masks are evaluation only. Does not adjust poses or meshes.
"""
import sys,json,argparse
from pathlib import Path
import cv2,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from realpc.v4 import load_library
from realpc.io import Episode,DepthStream
from realpc.calib_rig import RigCalib
from realpc.diffcalib import TorchArm,Renderer
from realpc.config import load_config
p=argparse.ArgumentParser();p.add_argument('--converted',required=True);p.add_argument('--tracker-seg-dir');p.add_argument('--reference-name',default='self');p.add_argument('--assets',default='assets/objects_v3');p.add_argument('--every',type=int,default=8);p.add_argument('--out',required=True);p.add_argument('--visual',nargs='*',type=int,default=[0,100,250,400,500,650,800]);p.add_argument('--stop',type=int);p.add_argument('--video',action='store_true');p.add_argument('--scene-occlusion',action='store_true',help='depth occlusion for DISPLAY ONLY; scoring stays unchanged');a=p.parse_args();torch.set_num_threads(2);d=np.load(a.converted);task=str(d['task']);episode=str(d['episode']);ep=Episode('/home/howardji/data_ext/agentwm_probe/hf/'+task+'/'+episode);cal=RigCalib.load(ROOT/'configs/calib_yam_rig_v3.json');obs,_=load_library(str(Path(a.assets)/task),100);cfg=load_config(ROOT/'configs/yam_bimanual.json');r=cfg['robot'];arm=TorchArm(str(Path(cfg['_dir'])/r['urdf']),r['arm_joints'],r['finger_joints']);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
names=list(d['obj_names']);idx={o.name:names.index(o.name) for o in obs};ticks=d['ticks'];ks=list(range(0,len(ticks),a.every));ks=[k for k in ks if a.stop is None or ticks[k]<a.stop];vs={int(np.argmin(abs(ticks-t))) for t in a.visual if ticks[0]<=t<=ticks[-1]};ks=sorted(set(ks)|vs);rows={};scores={a.reference_name:{},'independent':{}};per={a.reference_name:{},'independent':{}}
if a.tracker_seg_dir:scores['tracker']={};per['tracker']={}
colors=[(0,200,255),(255,120,0),(0,255,120)]
for cam in ['top','side']:
 z=np.load('/home/howardji/data_ext/realpc_v4/seg/'+task+'/'+episode+'_'+cam+'.npz');raw=z['labels'];ln=['background']+list(z['names']);labelsets={a.reference_name:(np.stack([cv2.resize(raw[int(ticks[k])],(320,180 if cam=='top' else 240),interpolation=0) for k in ks]),ln,[254,255])};del raw
 if a.tracker_seg_dir:
  z=np.load(Path(a.tracker_seg_dir)/task/(episode+'_'+cam+'.npz'));raw=z['labels'];start=int(z['start']);labelsets['tracker']=(np.stack([cv2.resize(raw[int(ticks[k])-start],(320,180 if cam=='top' else 240),interpolation=0) for k in ks]),['background']+list(z['names']),[254,255]);del raw
 qp=Path('/home/howardji/data_ext/masks')/task/episode/(cam+'_masks.npz')
 if qp.exists():
  z=np.load(qp);raw=z['masks'];labelsets['independent']=(np.stack([cv2.resize(raw[int(ticks[k])],(320,180 if cam=='top' else 240),interpolation=0) for k in ks]),list(z['label_names']),[2,3]);del raw
 K=ep.intrinsics(cam).copy();K[:2]*=320/(1280 if cam=='top' else 640);R=Renderer(K,180 if cam=='top' else 240,320);Tc=torch.tensor(cal.T_cam_base(cam,'left'),device='cuda',dtype=torch.float32);cap=cv2.VideoCapture(ep.video(cam));writer=cv2.VideoWriter(str(out/(cam+'_overlay.mp4')),cv2.VideoWriter_fourcc(*'mp4v'),30/a.every,(640,240)) if a.video else None
 ds=DepthStream(str(Path(ep.path)/(cam+'_depth.mkv')),int(ticks[ks[0]])) if a.scene_occlusion else None; depth_tick=int(ticks[ks[0]])-1
 for j,k in enumerate(ks):
  t=int(ticks[k]);zs=[]
  for o in obs:
   if not d['obj_valid'][idx[o.name],k]:zs.append(np.zeros((R.H,R.W),np.float32));continue
   T=torch.tensor(d['obj_T'][idx[o.name],k],device='cuda');V=o.verts_t@T[:3,:3].T+T[:3,3];_,zz=R.raster((V@Tc[:3,:3].T+Tc[:3,3])[None],o.faces_t);zs.append(zz[0].cpu().numpy())
  rz=np.full((R.H,R.W),np.inf)
  for side in ['left','right']:
   q=torch.tensor(d['q_'+side][k][None],device='cuda');V=arm.verts(q)[0];T=torch.tensor(cal.T_cam_base(cam,side),device='cuda',dtype=torch.float32);_,zz=R.raster((V@T[:3,:3].T+T[:3,3])[None],arm.faces);zz=zz[0].cpu().numpy();rz=np.minimum(rz,np.where(zz>0,zz,np.inf))
  nearest=np.minimum.reduce([np.where(z>0,z,np.inf) for z in zs]);vis=[(z>0)&(z<=nearest+.0001)&(z<rz) for z in zs]
  for mode,(L,ln,robots) in labelsets.items():
   lab=L[j];valid=~np.isin(lab,robots)
   for i,o in enumerate(obs):
    if o.name not in ln:continue
    gt=(lab==ln.index(o.name))&valid;pred=vis[i]&valid
    if max(gt.sum(),pred.sum())<60:continue
    iou=float((gt&pred).sum()/max(1,(gt|pred).sum()));key=o.name+'/'+cam;per[mode].setdefault(key,[]).append([t,iou])
  if k in vs or a.video:
   cap.set(0 if False else cv2.CAP_PROP_POS_FRAMES,t);ok,im=cap.read();im=cv2.resize(im,(320,R.H));ov=im.copy()
   display_vis=vis
   if ds is not None:
    while depth_tick<t:
     _,depth=ds.next();depth_tick+=1
    depth=cv2.resize(depth.astype(np.float32)*float(ep.meta['depth_scale'][cam]),(320,R.H),interpolation=cv2.INTER_NEAREST)
    display_vis=[m&((depth<.1)|(z<=depth+.015)) for m,z in zip(vis,zs)]
   for i,m in enumerate(display_vis):
    ov[m]=(.22*np.array(colors[i%3])+.78*ov[m]).astype(np.uint8);cs,_=cv2.findContours(m.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(ov,cs,-1,colors[i%3],1)
   row=np.concatenate([im,ov],1);cv2.putText(row,f'{task}/{episode} {cam} t={t}',(5,18),0,.5,(255,255,255),1);row=cv2.copyMakeBorder(row,0,240-R.H,0,0,0)
   if writer is not None:writer.write(row)
   if k in vs:rows.setdefault(t,[]).append(row)
 cap.release()
 if ds is not None:ds.close()
 if writer is not None:writer.release()
for mode in per:
 for key,records in per[mode].items():
  vals=np.array(records)[:,1];scores[mode][key]={'frames':len(vals),'median':float(np.median(vals)),'pass':float((vals>=.5).mean())}
 print(mode,{k:(round(v['pass'],3),round(v['median'],3)) for k,v in scores[mode].items()},flush=True)
for t,tiles in rows.items():cv2.imwrite(str(out/f'{task}_{episode}_{t:04d}.jpg'),np.concatenate(tiles,0))
json.dump({'converted':a.converted,'assets':a.assets,'every':a.every,'display_scene_occlusion':a.scene_occlusion,'tracker_seg_dir':a.tracker_seg_dir,'episode':task+'/'+episode,'summary':scores,'per_frame':per},open(out/'metrics.json','w'),indent=2)
