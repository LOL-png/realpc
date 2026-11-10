"""Generate causal cloth masks via the same OnlineClothMasks.step used live."""
import os,sys,argparse,json
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from realpc.io import Episode,DepthStream
from realpc.cloth_inputs import ClothRig,OnlineClothMasks,unproject,project
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--stop',type=int,default=453);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
ep=Episode('/home/howardji/data_ext/agentwm_probe/hf/greycloth/episode_0000');rig=ClothRig(ep,ROOT);cams=['top','side','left_wrist','right_wrist'];caps={c:cv2.VideoCapture(ep.video(c)) for c in cams};ds={c:DepthStream(str(Path(ep.path)/(c+'_depth.mkv'))) for c in cams};masker=OnlineClothMasks('/home/howardji/realpc/.model_cache/sam2/sam2.1_hiera_small.pt');saved={c:[] for c in cams};initP=None; (out/'frames').mkdir(exist_ok=True)
for t in range(a.stop):
 q=rig.joints(t);LP,G,T=rig.poses(q); frame={}; frame.update({'G_'+c:G[c] for c in G});frame.update({'q_'+c:q[c] for c in q});rp=rig.arms.points(q,poses=LP);frame['robot']=rp;frame.update({'tips_'+c:rp[rig.tipids[c]] for c in G})
 for c in cams:
  ok,im=caps[c].read();assert ok;_,raw=ds[c].next();D=raw.astype(np.float32)*float(ep.meta['depth_scale'][c]);h,w=im.shape[:2];scale=640/w;im=cv2.resize(im,(640,round(h*scale)));D=cv2.resize(D,(640,round(h*scale)),interpolation=0);K=ep.intrinsics(c).copy();K[:2]*=scale;seed=None
  if t==0:
   if c=='top':
    # Explicit current-t0 shirt prompt, not a future mask or recorded trajectory.
    # Grey table and grey cloth cannot be separated by a global HSV threshold.
    seed=np.zeros(D.shape,np.uint8)
    cv2.fillPoly(seed,[np.array([[184,181],[253,128],[310,112],[351,125],[398,106],[464,129],[547,162],[494,190],[436,176],[461,213],[509,263],[424,257],[352,264],[282,262],[292,218],[325,187],[269,197],[237,216]],np.int32)],1)
    seed=seed.astype(bool)
   else:
    uv,z=project(initP,K,T[c]);xy=np.round(uv).astype(int);keep=(z>.08)&(xy[:,0]>=0)&(xy[:,0]<640)&(xy[:,1]>=0)&(xy[:,1]<D.shape[0]);seed=np.zeros(D.shape,np.uint8);seed[xy[keep,1],xy[keep,0]]=1;seed=cv2.dilate(seed,np.ones((7,7),np.uint8)).astype(bool)
  m=masker.step(c,im,seed);saved[c].append(m);frame.update({c+'_rgb':im,c+'_depth':D,c+'_K':K,c+'_T':T[c],c+'_mask':m})
  if t==0:
   cv2.imwrite(str(out/(c+'_seed.jpg')),np.where(m[:,:,None],(.6*im+.4*np.array([0,180,255])).astype(np.uint8),im))
   if c=='top':initP,_=unproject(D,K,T[c],2,m)
 np.savez_compressed(out/'frames'/f'{t:04d}.npz',**frame)
 if t%50==0:print('cloth mask tick',t,flush=True)
for c in cams:
 np.savez_compressed(out/(c+'.npz'),masks=np.array(saved[c]),start=0,causal=True);caps[c].release();ds[c].close()
print('saved causal masks',out,flush=True)
