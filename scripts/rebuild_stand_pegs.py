"""Four measured individual peg fits; preserve the fitted base/pole.
Across-episode depth is used for the static asset only, never online tracking.
"""
import sys,json,shutil
from pathlib import Path
import numpy as np,trimesh
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.build_smooth_v3 import cylinder
out=Path('/home/howardji/data_ext/astra_cloth/stand01');out.mkdir(parents=True,exist_ok=True)
f=ROOT/'assets/objects_v3/hang/stand.ply';meta_path=f.with_suffix('.json')
for p in [f,meta_path]:shutil.copy2(p,out/('before_'+p.name))
m=trimesh.load(f,process=True);parts=list(m.split());pole=max(parts,key=lambda x:x.extents[2]);base=max(parts,key=lambda x:x.volume)
if base is pole:base=max([p for p in parts if p is not pole],key=lambda x:x.volume)
V=np.asarray(pole.vertices);center=V.mean(0);_,_,vh=np.linalg.svd(V-center);axis=vh[0];axis*=np.sign(axis[2])
def root_at(z):return center+axis*((z-center[2])/axis[2])
P=np.load('/home/howardji/data_ext/astra_cloth/stand00/stand_multiview.npz')['P'];_,ids=np.unique(np.floor(P/.0015).astype(int),axis=0,return_index=True);P=P[ids]
settings=[('lower_minus_x',0,-1,.045,np.pi),('lower_plus_x',0,1,.050,0),('upper_minus_y',1,-1,.132,-np.pi/2),('upper_plus_y',1,1,.130,np.pi/2)];fitted=[];rows=[]
for name,dim,sign,z0,az0 in settings:
 k=(P[:,dim]*sign>.022)&(P[:,dim]*sign<.103)&(abs(P[:,1-dim])<.030)&(abs(P[:,2]-z0)<.030)
 Q=P[k]
 # RANSAC-like trimmed PCA removes residual orange/edge flying pixels.
 for _ in range(4):
  cen=np.median(Q,0);_,_,vh=np.linalg.svd(Q-cen,full_matrices=False);u=vh[0];dist=np.linalg.norm(np.cross(Q-cen,u),axis=1);Q=Q[dist<max(.008,np.quantile(dist,.72))]
 def unpack(x):
  z,az,el,r=x;u=np.array([np.cos(az)*np.cos(el),np.sin(az)*np.cos(el),np.sin(el)]);return root_at(z),u,r
 def residual(x):
  a,u,r=unpack(x);d=Q-a;perp=d-(d@u)[:,None]*u;rr=np.linalg.norm(perp,axis=1)-r
  return np.r_[rr,(x[3]-.0046)*2]
 x0=[z0,az0,.10,.0046];lo=[z0-.015,az0-.35,-.12,.0035];hi=[z0+.015,az0+.35,.45,.007]
 fit=least_squares(residual,x0,bounds=(lo,hi),loss='soft_l1',f_scale=.0015,max_nfev=150);a,u,r=unpack(fit.x);proj=(Q-a)@u;length=float(np.quantile(proj,.985));b=a+u*length
 fitted.append(cylinder(a-u*.003,b,r,[171,134,83],n=48))
 rows.append({'name':name,'root':a.tolist(),'tip':b.tolist(),'length_m':length,'radius_m':float(r),'azimuth_deg':float(np.rad2deg(fit.x[1])),'elevation_deg':float(np.rad2deg(fit.x[2])),'root_height_m':float(a[2]),'support_voxels':len(Q),'surface_error_median_mm':float(np.median(abs(residual(fit.x)[:-1]))*1000)})
new=trimesh.util.concatenate([base,pole]+fitted);new.export(f);meta=json.load(open(meta_path));meta.update(version='v3.1 four individual stand pegs, multiview depth fit',peg_count=4,peg_levels=2,pegs=rows,peg_evidence='hang full-episode top/side and both wrist views: 4 exposed ends at 2 heights; distinct angled arms, not 2 straight crossbars',fit_source='Canonical stand depth from top/side every20 frames; visual peg census with wrist views. Static asset fitting only.',extents_m=new.extents.tolist(),volume_m3=float(new.volume),watertight=bool(new.is_watertight),n_vertices=len(new.vertices),n_faces=len(new.faces));json.dump(meta,open(meta_path,'w'),indent=2);json.dump(rows,open(out/'peg_fit.json','w'),indent=2)
print(json.dumps(rows,indent=2));print('watertight',new.is_watertight)
