"""Compare frozen-body handle candidates in raw multi-view RGB crops."""
import sys,json,argparse
from pathlib import Path
import numpy as np,cv2,trimesh,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from realpc.mug_handle import build_handle,split_mug
from realpc.diffcalib import Renderer
p=argparse.ArgumentParser();p.add_argument('--baseline-lib',default=str(ROOT/'assets/objects_v3/hang'));p.add_argument('--params');p.add_argument('--registered-wrists',action='store_true');p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
old=trimesh.load(Path(a.baseline_lib)/'mug.ply',process=False);meta=json.load(open(Path(a.baseline_lib)/'mug.json'));body,handle=split_mug(old);kw=json.load(open(a.params)) if a.params else {};new=trimesh.util.concatenate([body,build_handle(meta,**kw)])
rows=[]
for cam,t in [('top',0),('side',200),('side',280),('side',460),('right_wrist',100),('right_wrist',260),('right_wrist',400),('left_wrist',150),('left_wrist',340)]:
 folder='wrist_registered' if a.registered_wrists and 'wrist' in cam else 'observations';z=np.load(f'/home/howardji/data_ext/astra_mug_handle/{folder}/{cam}_{t:04d}.npz');rgb=z['rgb'];K=z['K'];T=z['Tco'];R=Renderer(K,*rgb.shape[:2]);tiles=[rgb.copy()]
 for mesh in [old,new]:
  with torch.inference_mode():
   V=torch.tensor(np.asarray(mesh.vertices)@T[:3,:3].T+T[:3,3],dtype=torch.float32,device='cuda');F=torch.tensor(mesh.faces,dtype=torch.int64,device='cuda');s,d=R.raster(V[None],F);mask=s[0].cpu().numpy()>0
  im=rgb.copy();cs,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,cs,-1,(0,255,255),1);tiles.append(im)
 ts=[]
 for label,im in zip(['raw','previous handle','ear spline'],tiles):
  scale=min(320/im.shape[1],280/im.shape[0]);im=cv2.resize(im,None,fx=scale,fy=scale);tile=np.full((310,340,3),245,np.uint8);tile[28:28+len(im),:im.shape[1]]=im;cv2.putText(tile,f'{cam} {t} {label}'+(' body-fit' if a.registered_wrists and 'wrist' in cam else ''),(6,20),0,.45,(0,0,0),1);ts.append(tile)
 row=np.hstack(ts);rows.append(row);cv2.imwrite(str(out/f'{cam}_{t:04d}.jpg'),row)
cv2.imwrite(str(out/'before_after.jpg'),np.vstack(rows));new.export(out/'mug.ply')
