"""Frozen-pose, handle-only audit of color/depth evidence; excludes fixed cup pixels."""
import sys,json,argparse
from pathlib import Path
import numpy as np,torch,cv2,trimesh
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from realpc.mug_handle import split_mug,build_handle
from realpc.diffcalib import Renderer
p=argparse.ArgumentParser();p.add_argument('--baseline-lib',default=str(ROOT/'assets/objects_v3/hang'));p.add_argument('--params',action='append',required=True,help='label=params.json');p.add_argument('--out',required=True);a=p.parse_args();torch.set_num_threads(2)
old=trimesh.load(Path(a.baseline_lib)/'mug.ply',process=False);meta=json.load(open(Path(a.baseline_lib)/'mug.json'));body,handle=split_mug(old);models={'old':handle}
for spec in a.params:
 name,path=spec.split('=',1);models[name]=build_handle(meta,**json.load(open(path)))
tensors={n:(torch.tensor(m.vertices,dtype=torch.float32,device='cuda'),torch.tensor(m.faces,dtype=torch.int64,device='cuda')) for n,m in dict(body=body,**models).items()};rows=[]
for record in json.load(open('/home/howardji/data_ext/astra_mug_handle/observations/index.json')):
 cam,t=record['cam'],record['t'];src=Path('/home/howardji/data_ext/astra_mug_handle/observations')/record['file'];reg=src.parent.parent/'wrist_registered'/src.name
 if 'wrist' in cam:
  if not reg.exists():continue
  src=reg
 z=np.load(src);rgb=z['rgb'];D=z['D'];T=z['Tco'];R=Renderer(z['K'],*D.shape);hsv=cv2.cvtColor(rgb,cv2.COLOR_BGR2HSV);M=(hsv[:,:,0]<26)&(hsv[:,:,1]>160)&(hsv[:,:,2]>40)&(rgb[:,:,2]>1.65*np.maximum(rgb[:,:,1],1));M &= z['mask']
 render={}
 with torch.inference_mode():
  for n,(V,F) in tensors.items():
   tc=torch.tensor(T,dtype=torch.float32,device='cuda');s,d=R.raster((V@tc[:3,:3].T+tc[:3,3])[None],F);render[n]=(s[0].cpu().numpy()>0,d[0].cpu().numpy())
 bm,bz=render['body'];outside=~cv2.dilate(bm.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
 roi=np.zeros(D.shape,np.uint8)
 for n,(m,d) in render.items():
  if n!='body':roi |= m.astype(np.uint8)
 roi=cv2.dilate(roi,np.ones((21,21),np.uint8)).astype(bool);valid=outside&roi;target=M&valid
 if target.sum()<20:continue
 row=dict(cam=cam,t=t,registered_wrist='wrist' in cam,observed_handle_px=int(target.sum()),models={})
 for n in models:
  mask,depth=render[n];validn=valid&~((D>.08)&(depth>D+.008)&~M);pred=mask&validn;gt=target&validn
  overlap=pred&gt&(D>.08);err=np.abs(depth[overlap]-D[overlap])*1000
  dt=cv2.distanceTransform((~gt).astype(np.uint8),cv2.DIST_L2,5);dp=cv2.distanceTransform((~pred).astype(np.uint8),cv2.DIST_L2,5)
  row['models'][n]=dict(iou=float((pred&gt).sum()/max(1,(pred|gt).sum())),depth_med_mm=float(np.median(err)) if len(err) else None,chamfer_px=float((np.mean(dt[pred])+np.mean(dp[gt]))/2) if pred.any() and gt.any() else None)
 rows.append(row)
summary={}
for cam in ['top','side','left_wrist','right_wrist']:
 part=[r for r in rows if r['cam']==cam];summary[cam]={}
 for n in models:
  summary[cam][n]={k:float(np.median([r['models'][n][k] for r in part if r['models'][n][k] is not None])) for k in ['iou','depth_med_mm','chamfer_px']};summary[cam][n]['frames']=len(part)
Path(a.out).write_text(json.dumps(dict(summary=summary,rows=rows,note='Offline orange RGB/depth handle-only consistency, frozen body and poses; wrist body registered separately, fixed cameras retain calibrated poses. Not independent ground truth.'),indent=2));print(json.dumps(summary,indent=2))
