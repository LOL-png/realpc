#!/usr/bin/env python
"""Small joint-two-camera canonical mesh fit on causal t0 masks; fixed calibration.
Only the same mesh transform/scales are optimized for both cameras. No per-view pose.
"""
import os,sys,json,argparse
from pathlib import Path
import numpy as np,cv2,torch,trimesh
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from realpc.diffcalib import Renderer,se3_exp,soft_iou
from realpc.calib_rig import RigCalib
from realpc.io import Episode,read_depth_frames
p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--names',nargs='+');p.add_argument('--steps',type=int,default=65);p.add_argument('--out',required=True);p.add_argument('--seg',default='/home/howardji/data_ext/realpc_v4/seg');a=p.parse_args()
torch.set_num_threads(2);cal=RigCalib.load(ROOT/'configs/calib_yam_rig_v3.json');ep=Episode('/home/howardji/data_ext/agentwm_probe/hf/'+a.task+'/episode_0000');lib=ROOT/'assets/objects_v3'/a.task
views=[]
for cam in ['top','side']:
 z=np.load(a.seg+'/'+a.task+'/episode_0000_'+cam+'.npz');L=z['labels'][0];names=list(z['names']);W=320;H=round(L.shape[0]*W/L.shape[1]);lab=cv2.resize(L,(W,H),interpolation=cv2.INTER_NEAREST)
 K=ep.intrinsics(cam).copy();K[:2]*=W/(1280 if cam=='top' else 640);R=Renderer(K,H,W)
 D=read_depth_frames(ep.path+'/'+cam+'_depth.mkv',[0])[0].astype(np.float32)*float(ep.meta['depth_scale'][cam]);D=cv2.resize(D,(W,H),interpolation=cv2.INTER_NEAREST)
 views.append((cam,R,lab,names,torch.tensor(D,device='cuda'),torch.tensor(cal.T_cam_base(cam,'left'),dtype=torch.float32,device='cuda')))
log={}
for fn in sorted(lib.glob('*.ply')):
 name=fn.stem
 if a.names and name not in a.names:continue
 m=trimesh.load(fn,process=True)
 if len(m.faces)>8000:m=m.simplify_quadric_decimation(face_count=8000)
 meta=json.load(open(fn.with_suffix('.json')));T=np.array(meta['T_world_obj_ref']);T=torch.tensor(T,dtype=torch.float32,device='cuda')
 V=torch.tensor(np.array(m.vertices),dtype=torch.float32,device='cuda');F=torch.tensor(np.array(m.faces),device='cuda',dtype=torch.long)
 x=torch.nn.Parameter(torch.zeros(9,device='cuda'));opt=torch.optim.Adam([x],lr=.015);best=(1e9,None);initial=None
 for it in range(a.steps+1):
  opt.zero_grad();delta=torch.cat([x[:3]*.3,x[3:6]*.03]);M=se3_exp(delta);scale=torch.exp(x[6:]*.2)
  Vn=(V*scale)@M[:3,:3].T+M[:3,3];Vw=Vn@T[:3,:3].T+T[:3,3];loss=0;ious=[]
  for cam,R,lab,names,D,Tcw in views:
   lid=names.index(name)+1;mask=torch.tensor(lab==lid,device='cuda',dtype=torch.float32);valid=torch.tensor((lab==0)|(lab==lid),device='cuda',dtype=torch.float32)
   sil,z=R.raster((Vw@Tcw[:3,:3].T+Tcw[:3,3])[None],F,sigma=1e-6,fpp=8);sil=sil[0];z=z[0]
   iou=soft_iou(sil,mask,valid);ious.append(float(iou.detach()));loss=loss+1-iou
   good=(mask>.5)&(D>.1)&(z>0)&(torch.abs(z-D)<.06)
   if good.sum()>20:loss=loss+.15*torch.nn.functional.smooth_l1_loss(z[good]/.01,D[good]/.01,beta=1)
  loss=loss+.015*(x[:3]**2).sum()+.015*(x[6:]**2).sum()
  val=float(loss.detach())
  if initial is None:initial=ious
  if val<best[0]:best=(val,x.detach().clone())
  if it==a.steps:break
  loss.backward();opt.step()
  if it%20==0:print(name,it,round(val,4),[round(q,3) for q in ious],flush=True)
 with torch.no_grad():
  x=best[1];M=se3_exp(torch.cat([x[:3]*.3,x[3:6]*.03])).cpu().numpy();sc=torch.exp(x[6:]*.2).cpu().numpy()
 orig=trimesh.load(fn,process=True);orig.vertices=(np.array(orig.vertices)*sc)@M[:3,:3].T+M[:3,3];orig.export(fn)
 meta.update(version='v3 smooth prior + shared-two-camera t0 fit',mesh_fit_transform=M.tolist(),mesh_fit_scale=sc.tolist(),fit_masks='causal SAM2 t0 only; no Q28 evaluation masks',extents_m=orig.extents.tolist(),watertight=bool(orig.is_watertight));json.dump(meta,open(fn.with_suffix('.json'),'w'),indent=2)
 log[name]={'initial_iou':initial,'last_iou':ious,'best_loss':best[0],'transform':M.tolist(),'scale':sc.tolist()};print(name,'DONE',log[name],flush=True)
Path(a.out).parent.mkdir(parents=True,exist_ok=True);json.dump(log,open(a.out,'w'),indent=2)
