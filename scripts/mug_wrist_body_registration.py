"""Offline body-only wrist registration for shape evidence; never updates runtime calibration."""
import sys,json
from pathlib import Path
import numpy as np,torch,cv2,trimesh
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from realpc.mug_handle import split_mug
from realpc.diffcalib import Renderer,se3_exp
body,_=split_mug(trimesh.load(ROOT/'assets/objects_v3/hang/mug.ply',process=False));V=torch.tensor(body.vertices,dtype=torch.float32,device='cuda');F=torch.tensor(body.faces,dtype=torch.int64,device='cuda');torch.set_num_threads(2)
out=Path('/home/howardji/data_ext/astra_mug_handle/wrist_registered');out.mkdir(exist_ok=True);log=[]
for cam,t in [('right_wrist',100),('right_wrist',260),('right_wrist',340),('right_wrist',400),('left_wrist',150),('left_wrist',340)]:
 src=Path(f'/home/howardji/data_ext/astra_mug_handle/observations/{cam}_{t:04d}.npz');z=dict(np.load(src));rgb=z['rgb'];scale=min(1,240/max(rgb.shape[:2]));h,w=round(rgb.shape[0]*scale),round(rgb.shape[1]*scale);im=cv2.resize(rgb,(w,h));D=cv2.resize(z['D'],(w,h),interpolation=0);K=z['K'].copy();K[:2]*=scale;R=Renderer(K,h,w);T=torch.tensor(z['Tco'],device='cuda',dtype=torch.float32);vc=V@T[:3,:3].T+T[:3,3]
 hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV);M=((hsv[:,:,0]<26)&(hsv[:,:,1]>150)&(hsv[:,:,2]>40)).astype(np.float32)
 with torch.no_grad():bm,bz=R.raster(vc[None],F)
 mask=bm[0].cpu().numpy()>0;roi=cv2.dilate(mask.astype(np.uint8),np.ones((21,21),np.uint8))>0
 # Unknown foreground obscures the body rather than deleting orange surface.
 valid=roi&~((M==0)&(D>.08)&(D<float(np.median(vc[:,2].cpu().numpy()))-.03))
 m=torch.tensor(M,device='cuda');valid=torch.tensor(valid,device='cuda');dd=torch.tensor(D,device='cuda');x=torch.nn.Parameter(torch.zeros(6,device='cuda'));opt=torch.optim.Adam([x],lr=.035);best=(1e9,None)
 for it in range(85):
  opt.zero_grad();delta=se3_exp(torch.cat([x[:3]*.15,x[3:]*.03]));q=vc@delta[:3,:3].T+delta[:3,3];sil,depth=R.raster(q[None],F,sigma=2e-5,fpp=6);s,d=sil[0],depth[0];loss=1-(s*m*valid).sum()/((s+m-s*m)*valid).sum().clamp_min(1)
  good=valid&(m>.5)&(dd>.08)&(d>0)&(abs(dd-d)<.04)
  if good.sum()>20:loss+=.2*torch.nn.functional.smooth_l1_loss(d[good]/.01,dd[good]/.01,beta=1)
  loss+=.004*x.square().sum();value=float(loss.detach())
  if it==0:before=value
  if value<best[0]:best=(value,delta.detach().cpu().numpy())
  loss.backward();opt.step()
 z['Tco_original']=z['Tco'].copy();z['Tco']=best[1]@z['Tco'];np.savez_compressed(out/src.name,**z);r=dict(cam=cam,t=t,before=before,after=best[0],delta=best[1].tolist());log.append(r);print(cam,t,round(before,3),round(best[0],3),flush=True)
(out/'registration.json').write_text(json.dumps(dict(note='Offline per-image body-only registration for shape evidence. Does not alter shared camera calibration or runtime poses.',views=log),indent=2))
