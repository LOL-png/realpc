"""Offline multi-view strap-handle fit. Cup vertices and calibrated poses frozen."""
import sys,json,argparse
from pathlib import Path
import numpy as np,cv2,torch,trimesh
from scipy.interpolate import CubicSpline
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from realpc.mug_handle import DEFAULT,cup_frame,split_mug,sweep_arrays,profile_basis
from realpc.diffcalib import Renderer
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--profile',choices=['straight','ear'],default='ear');p.add_argument('--wrists',action='store_true');p.add_argument('--steps',type=int,default=180);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
meta=json.load(open(ROOT/'assets/objects_v3/hang/mug.json'));old=trimesh.load(ROOT/'assets/objects_v3/hang/mug.ply',process=False);body,_=split_mug(old);B,o=cup_frame(meta)
tt=lambda v:torch.as_tensor(v,dtype=torch.float32,device='cuda')
B,o=tt(B),tt(o);bodyV,bodyF=tt(body.vertices),torch.as_tensor(body.faces,dtype=torch.int64,device='cuda');_,F=sweep_arrays();F=torch.as_tensor(F,dtype=torch.int64,device='cuda')
samples=65;section=24;basis=tt(profile_basis(samples,a.profile));th=np.arange(section)*2*np.pi/section;cs=tt(np.sign(np.cos(th))*abs(np.cos(th))**.5);sn=tt(np.sign(np.sin(th))*abs(np.sin(th))**.5);taper=tt(np.minimum(1,np.minimum(np.arange(samples)+1,np.arange(samples,0,-1))/3))
base=tt(DEFAULT);param=torch.nn.Parameter(torch.zeros(13,device='cuda'));opt=torch.optim.Adam([param],lr=.035)
def model():
 q=torch.tanh(param);zt=.039+.006*q[0];zb=-.025+.012*q[1];outer=.061+.012*q[2];shoulder=.004+.003*q[3];span=zt-zb;lowouter=outer+.004*q[6]
 wall=lambda z:.03744+(.0408-.03744)*(z+.0425)/.08375-.0007
 rtop=wall(zt);rbot=wall(zb);zero=param[0]*0
 control=torch.stack([torch.stack([rtop,zt]),torch.stack([.050+.006*q[10],zt+shoulder]),torch.stack([outer-.003,zt+shoulder*.4]),torch.stack([outer,zt-span*(.20+.08*q[4])]),torch.stack([(outer+lowouter)/2+.002*q[12],(zt+zb)/2]),torch.stack([lowouter,zb+span*(.22+.08*q[5])]),torch.stack([lowouter-.005,zb+.004]),torch.stack([rbot+.006+.003*q[11],zb]),torch.stack([rbot,zb])])
 if a.profile=='straight':control=torch.cat([control[:4],((control[3]+control[5])/2)[None],control[5:]],0)
 else:
  # Smooth top bulge; progressively narrower lower lobe, no straight segment.
  reach=outer-rtop;lower=.48+.13*q[6]
  control=torch.stack([torch.stack([rtop,zt]),torch.stack([rtop+.42*reach,zt+shoulder]),torch.stack([outer-.10*reach,zt-.07*span]),torch.stack([outer,zt-.25*span]),torch.stack([rbot+.90*reach,zt-.48*span]),torch.stack([rbot+lower*reach,zt-.72*span]),torch.stack([rbot+.28*reach,zt-.88*span]),torch.stack([rbot+.10*reach,zt-.97*span]),torch.stack([rbot,zb])])
 angle=-.13+.6*q[7];thin=.0025+.0015*torch.sigmoid(param[8]);wide=.004+.004*torch.sigmoid(param[9])
 path=basis@control;d=torch.cat([path[1:2]-path[:1],(path[2:]-path[:-2])/2,path[-1:]-path[-2:-1]]);d=d/d.norm(dim=1,keepdim=True).clamp_min(1e-8);normal=torch.stack([-d[:,1],d[:,0]],1)
 taper_ratio=(.70+.15*q[12]) if a.profile=="ear" else param[0]*0+1
 local_taper=taper*(1-(1-taper_ratio)*torch.linspace(0,1,samples,device="cuda")**1.2)
 xz=path[:,None]+normal[:,None]*cs[None,:,None]*thin*local_taper[:,None,None];v=torch.stack([xz[:,:,0],wide*sn[None,:]*local_taper[:,None],xz[:,:,1]],-1).reshape(-1,3);c,s=torch.cos(angle),torch.sin(angle);v=torch.stack([c*v[:,0]-s*v[:,1],s*v[:,0]+c*v[:,1],v[:,2]],1)
 return v@B.T+o,control,angle,thin,wide
views=[]
view_spec=[('top',[0,80,180,200,360,420,480]),('side',[0,80,200,240,320,400,460])]
if a.wrists:view_spec += [('right_wrist',[100,260,340,400]),('left_wrist',[150,340])]
for cam,ticks in view_spec:
 for t in ticks:
  folder='wrist_registered' if 'wrist' in cam else 'observations';f=np.load(f'/home/howardji/data_ext/astra_mug_handle/{folder}/{cam}_{t:04d}.npz');rgb=f['rgb'];scale=min(1,220/max(rgb.shape[:2]));h,w=round(rgb.shape[0]*scale),round(rgb.shape[1]*scale);rgb=cv2.resize(rgb,(w,h));D=cv2.resize(f['D'],(w,h),interpolation=0);K=f['K'].copy();K[:2]*=scale;T=tt(f['Tco']);R=Renderer(K,h,w)
  hsv=cv2.cvtColor(rgb,cv2.COLOR_BGR2HSV);M=(hsv[:,:,0]<26)&(hsv[:,:,1]>160)&(hsv[:,:,2]>40)&(rgb[:,:,2]>1.65*np.maximum(rgb[:,:,1],1));M &= cv2.resize(f['mask'].astype(np.uint8),(w,h),interpolation=0)>0
  with torch.no_grad():bodymask,bodydepth=R.raster((bodyV@T[:3,:3].T+T[:3,3])[None],bodyF)
  bm=bodymask[0].cpu().numpy()>0
  # Score only the exposed handle region, not the much larger fixed cup body.
  near=np.array([[x,y,z] for x in [.03,.095] for y in [-.035,.035] for z in [-.05,.06]]);Q=(near@B.cpu().numpy().T+o.cpu().numpy())@T[:3,:3].cpu().numpy().T+T[:3,3].cpu().numpy();uv=Q[:,:2]/Q[:,2:]*[K[0,0],K[1,1]]+[K[0,2],K[1,2]];roi=np.zeros((h,w),np.uint8);cv2.fillConvexPoly(roi,cv2.convexHull(np.rint(uv).astype(np.int32)),1)
  valid=(roi>0)&~cv2.erode(bm.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
  # Wooden pegs and robot foreground must not be fitted as missing orange.
  occ=(D>.08)&(D<float(T[2,3])-.035)&~M;valid &= ~occ
  if (M&valid).sum()<10:continue
  views.append(dict(cam=cam,t=t,R=R,T=T,D=tt(D),M=tt(M),valid=tt(valid),bodymask=bodymask[0],bodydepth=bodydepth[0],rgb=rgb))
for v in views:
 mask=(v['M'].cpu().numpy()>.5)&(v['valid'].cpu().numpy()>.5)&~cv2.dilate((v['bodymask'].cpu().numpy()>.5).astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
 yy,xx=np.nonzero(mask);uv=np.c_[xx,yy];uv=uv[np.linspace(0,len(uv)-1,min(350,len(uv))).astype(int)];v['target_uv']=tt(uv)
views=[v for v in views if len(v['target_uv'])>=8]
print('views',[(v['cam'],v['t']) for v in views],flush=True);best=(1e9,None);logs=[]
for it in range(a.steps+1):
 opt.zero_grad();V,control,angle,thin,wide=model();loss=0;vals=[]
 for v in views:
  T=v['T'];Q=V[::2]@T[:3,:3].T+T[:3,3];uv=v['R'].project(Q);H,W=v['M'].shape
  ij=uv.detach().round().long();inside=(ij[:,0]>=0)&(ij[:,0]<W)&(ij[:,1]>=0)&(ij[:,1]<H);ii=ij[:,0].clamp(0,W-1);jj=ij[:,1].clamp(0,H-1)
  bd=v['bodydepth'][jj,ii];depth=v['D'][jj,ii];good=inside&(v['valid'][jj,ii]>.5)&((bd<=0)|(Q[:,2]<bd-.0002))&~((depth>.08)&(depth<Q[:,2]-.01)&(v['M'][jj,ii]<.5))
  pts=uv[good];obs=v['target_uv'];meters_per_pixel=float(T[2,3])/float(v['R'].K[0,0])
  if len(pts)<4:loss+=V.sum()*0+10;continue
  distances=torch.cdist(pts[None],obs[None])[0];robust=lambda d:torch.sqrt(1+d*d)-1
  il=(robust(distances.min(1).values).mean()+robust(distances.min(0).values).mean())*meters_per_pixel*1000
  # Actual depth at orange pixels is additional, weaker surface evidence.
  zerr=abs(Q[good,2]-depth[good]);dgood=(depth[good]>.08)&(v['M'][jj[good],ii[good]]>.5)&(zerr<.025)
  dl=(zerr[dgood]*1000).mean() if dgood.any() else il*0
  loss+=(.5 if 'wrist' in v['cam'] else 1.)*(il+(.05 if 'wrist' in v['cam'] else .5)*dl);vals.append(float(il.detach()))
 loss/=len(views)
 # Gentle prior; no circular arc constraint. Penalise zigzags and buried endpoint drift.
 loss+=.002*(param**2).mean()
 val=float(loss.detach())
 if val<best[0]:best=(val,param.detach().clone())
 if it%20==0:print(it,round(val,4),'angle',float(angle)*180/np.pi,'thickness/width',float(thin)*2000,float(wide)*2000,flush=True)
 if it==a.steps:break
 loss.backward();torch.nn.utils.clip_grad_norm_([param],5);opt.step()
with torch.no_grad():
 param.copy_(best[1]);V,c,angle,thin,wide=model();params=dict(control=c.cpu().numpy().tolist(),angle=float(angle),half_thickness=float(thin),half_width=float(wide),profile=a.profile,taper_ratio=float(.70+.15*torch.tanh(param[12])) if a.profile=="ear" else 1.)
(out/'params.json').write_text(json.dumps(params,indent=2));(out/'fit.json').write_text(json.dumps(dict(loss=best[0],training_views=[dict(cam=v['cam'],t=v['t']) for v in views],source='current RGB color + depth; no Q28 evaluation masks; fixed cup/poses'),indent=2));print(params,flush=True)
