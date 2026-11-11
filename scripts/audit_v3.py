#!/usr/bin/env python
"""Efficient joint-camera audit: render once per camera/tick, score both mask sources.
Independent Q28 masks are evaluation only. Does not adjust poses or meshes.
"""
import sys,json,argparse,os
from pathlib import Path
import cv2,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from realpc.v4 import load_library
from realpc.io import Episode,DepthStream
from realpc.calib_rig import RigCalib
from realpc.diffcalib import TorchArm,Renderer
from realpc.config import load_config
p=argparse.ArgumentParser();p.add_argument('--scene-seg-dir',help='Shared ScenePipeline label cache; evaluation only');p.add_argument('--converted',required=True);p.add_argument('--tracker-seg-dir');p.add_argument('--reference-name',default='self');p.add_argument('--assets',default='assets/objects_v3');p.add_argument('--every',type=int,default=8);p.add_argument('--out',required=True);p.add_argument('--visual',nargs='*',type=int,default=[0,100,250,400,500,650,800]);p.add_argument('--stop',type=int);p.add_argument('--video',action='store_true');p.add_argument('--depth-check',action='store_true',help='Report raw depth agreement separately from silhouette masks');p.add_argument('--scene-occlusion',action='store_true',help='depth occlusion for DISPLAY ONLY; scoring stays unchanged');a=p.parse_args();torch.set_num_threads(2);d=np.load(a.converted);task=str(d['task']);episode=str(d['episode']);ep=Episode(str(Path(os.environ.get('REALPC_EPISODES_ROOT','/home/howardji/data_ext/agentwm_probe/hf'))/task/episode));cal=RigCalib.load(ROOT/'configs/calib_yam_rig_v3.json');obs,_=load_library(str(Path(a.assets)/task),100);cfg=load_config(ROOT/'configs/yam_bimanual.json');r=cfg['robot'];arm=TorchArm(str(Path(cfg['_dir'])/r['urdf']),r['arm_joints'],r['finger_joints']);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
names=list(d['obj_names']);idx={o.name:names.index(o.name) for o in obs};ticks=d['ticks'];ks=list(range(0,len(ticks),a.every));ks=[k for k in ks if a.stop is None or ticks[k]<a.stop];vs={int(np.argmin(abs(ticks-t))) for t in a.visual if ticks[0]<=t<=ticks[-1]};ks=sorted(set(ks)|vs);rows={};scores={a.reference_name:{},'independent':{}};per={a.reference_name:{},'independent':{}};depth_per={};support_per={}
if a.tracker_seg_dir:scores['tracker']={};per['tracker']={}
palette={"box":(0,200,255),"broom":(255,120,0),"debris":(0,255,120),"dustpan":(220,70,220),"mug":(0,200,255),"stand":(255,120,0)}
colors=[palette.get(o.name,(180,180,180)) for o in obs]
for cam in ['top','side']:
 source=Path(a.scene_seg_dir)/task/(episode+'_'+cam+'_masks.npz') if a.scene_seg_dir else Path('/home/howardji/data_ext/realpc_v4/seg')/task/(episode+'_'+cam+'.npz')
 z=np.load(source);raw=z['labels'];ln=['background']+list(z['names']);labelsets={a.reference_name:(np.stack([cv2.resize(raw[int(ticks[k])],(320,180 if cam=='top' else 240),interpolation=0) for k in ks]),ln,[254,255])};del raw
 if a.tracker_seg_dir:
  z=np.load(Path(a.tracker_seg_dir)/task/(episode+'_'+cam+'.npz'));raw=z['labels'];start=int(z['start']);labelsets['tracker']=(np.stack([cv2.resize(raw[int(ticks[k])-start],(320,180 if cam=='top' else 240),interpolation=0) for k in ks]),['background']+list(z['names']),[254,255]);del raw
 qp=Path(os.environ.get('REALPC_MASKS_ROOT','/home/howardji/data_ext/masks'))/task/episode/(cam+'_masks.npz')
 if qp.exists():
  z=np.load(qp);raw=z['masks'];labelsets['independent']=(np.stack([cv2.resize(raw[int(ticks[k])],(320,180 if cam=='top' else 240),interpolation=0) for k in ks]),list(z['label_names']),[2,3]);del raw
 K=ep.intrinsics(cam).copy();K[:2]*=320/(1280 if cam=='top' else 640);R=Renderer(K,180 if cam=='top' else 240,320);Tc=torch.tensor(cal.T_cam_base(cam,'left'),device='cuda',dtype=torch.float32);cap=cv2.VideoCapture(ep.video(cam));writer=cv2.VideoWriter(str(out/(cam+'_overlay.mp4')),cv2.VideoWriter_fourcc(*'mp4v'),(cap.get(cv2.CAP_PROP_FPS) or 30)/a.every,(640,240)) if a.video else None
 ds=DepthStream(str(Path(ep.path)/(cam+'_depth.mkv')),int(ticks[ks[0]])) if (a.scene_occlusion or a.depth_check) else None; depth_tick=int(ticks[ks[0]])-1
 for j,k in enumerate(ks):
  t=int(ticks[k]);zs=[]
  for o in obs:
   if not d['obj_valid'][idx[o.name],k]:zs.append(np.zeros((R.H,R.W),np.float32));continue
   T=torch.tensor(d['obj_T'][idx[o.name],k],device='cuda');V=o.verts_t@T[:3,:3].T+T[:3,3];_,zz=R.raster((V@Tc[:3,:3].T+Tc[:3,3])[None],o.faces_t);zs.append(zz[0].cpu().numpy())
  rz=np.full((R.H,R.W),np.inf)
  for side in ['left','right']:
   q=torch.tensor(d['q_'+side][k][None],device='cuda');V=arm.verts(q)[0];T=torch.tensor(cal.T_cam_base(cam,side),device='cuda',dtype=torch.float32);_,zz=R.raster((V@T[:3,:3].T+T[:3,3])[None],arm.faces);zz=zz[0].cpu().numpy();rz=np.minimum(rz,np.where(zz>0,zz,np.inf))
  nearest=np.minimum.reduce([np.where(z>0,z,np.inf) for z in zs]);vis=[(z>0)&(z<=nearest+.0001)&(z<rz) for z in zs]
  depth=None
  if ds is not None:
   while depth_tick<t:
    _,depth_raw=ds.next();depth_tick+=1
   depth=cv2.resize(depth_raw.astype(np.float32)*float(ep.meta['depth_scale'][cam]),(320,R.H),interpolation=cv2.INTER_NEAREST)
  if a.depth_check:
   for i,o in enumerate(obs):
    valid_depth=vis[i]&(depth>.08)&np.isfinite(depth);delta=depth-zs[i]
    # A closer measured surface can occlude this mesh. Behind-surface/free-space
    # violations remain evidence against the predicted pose.
    comparable=valid_depth&(delta>=-.015)
    n=int(comparable.sum());nv=int(valid_depth.sum())
    if nv<60:continue
    err=np.abs(delta[comparable])*1000
    median=float(np.median(err)) if n>=60 else None
    free=float((delta[comparable]>.015).mean()) if n>=60 else None
    passed=bool(median<=10 and free<=.2) if n>=60 else None
    depth_per.setdefault(o.name+'/'+cam,[]).append(dict(t=t,visible_depth_pixels=nv,comparable_pixels=n,occluded_fraction=float((valid_depth&(delta<-.015)).sum()/nv),median_mm=median,free_fraction=free,pass_depth=passed))
  for mode,(L,ln,robots) in labelsets.items():
   lab=L[j];valid=~np.isin(lab,robots)
   for i,o in enumerate(obs):
    if o.name not in ln:continue
    gt=(lab==ln.index(o.name))&valid;pred=vis[i]&valid
    if max(gt.sum(),pred.sum())<60:continue
    iou=float((gt&pred).sum()/max(1,(gt|pred).sum()));key=o.name+'/'+cam;per[mode].setdefault(key,[]).append([t,iou]);support_per.setdefault(mode,{}).setdefault(key,[]).append(dict(t=t,gt_pixels=int(gt.sum()),pred_pixels=int(pred.sum()),intersection=int((gt&pred).sum()),union=int((gt|pred).sum()),iou=iou))
  if k in vs or a.video:
   if int(cap.get(cv2.CAP_PROP_POS_FRAMES))!=t:cap.set(cv2.CAP_PROP_POS_FRAMES,t)
   ok,im=cap.read();im=cv2.resize(im,(320,R.H));ov=im.copy()
   display_vis=vis
   if a.scene_occlusion:
    display_vis=[m&((depth<.1)|(z<=depth+.015)) for m,z in zip(vis,zs)]
   for i,m in enumerate(display_vis):
    ov[m]=(.22*np.array(colors[i])+.78*ov[m]).astype(np.uint8);cs,_=cv2.findContours(m.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(ov,cs,-1,colors[i],1)
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
json.dump({'converted':a.converted,'assets':a.assets,'every':a.every,'display_scene_occlusion':a.scene_occlusion,'tracker_seg_dir':a.tracker_seg_dir,'scene_seg_dir':a.scene_seg_dir,'reference_note':'stream consistency is not independent accuracy; Q28 masks are evaluation-only where available','episode':task+'/'+episode,'summary':scores,'per_frame':per,'support_per_frame':support_per,'depth_per_frame':depth_per,'depth_note':'Raw depth consistency: at least60 non-occluded valid-depth pixels; pass median absolute depth<=10mm and free-space fraction<=20% (15mm gate). Not semantic identity ground truth.'},open(out/'metrics.json','w'),indent=2)
