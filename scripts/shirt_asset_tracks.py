"""Offline shared rest-shape fit from persistent observed two-view tracks.
Prior mesh is used only to map track births into material coordinates; observed
track/depth distances, not old mesh edges, constrain all 453 frames jointly.
"""
import argparse,sys,json,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import RBFInterpolator
from scipy.optimize import least_squares
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from realpc.cloth_causal import at_image,depth_points
from realpc.shirt_template import outline,make_panels

def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--tracks',required=True);p.add_argument('--seed',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);seed=Path(a.seed);info=json.loads((seed/'template_info.json').read_text());z=np.load(seed/'mesh.npz');X,V,F=z['X'],z['V0'],z['F'];n=info['panel_vertices'];angle=info['fit_parameters'][0]
    R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]);uv=(V[:,:2]-V[:n,:2].mean(0))@R
    poly=outline()*(np.ptp(uv[:n],axis=0)/np.ptp(outline(),axis=0));poly+=uv[:n].min(0)-poly.min(0);basis=RBFInterpolator(poly,np.eye(15),kernel='thin_plate_spline')
    groups={};rows=[]
    # Nearest-face material coordinates are only assigned on birth packets.
    import trimesh
    for t in range(len(X)):
        with np.load(Path(a.inputs)/'frames'/f'{t:04d}.npz') as f,np.load(Path(a.tracks)/f'{t:04d}.npz') as cot:
            for cam in ('top','side'):
                birth=int(cot[cam+'_cot_birth']);key=(cam,birth);pix=cot[cam+'_cot_uv'];d,inside=at_image(f[cam+'_depth'],pix);mask,_=at_image(f[cam+'_mask'],pix)
                Q=depth_points(pix,d,f[cam+'_K'],f[cam+'_T']);valid=inside&mask&(d>.08)&cot[cam+'_cot_visible']
                if key not in groups:
                    mesh=trimesh.Trimesh(X[t],F,process=False);nearest,dist,fi=trimesh.proximity.closest_point_naive(mesh,Q)
                    bc=trimesh.triangles.points_to_barycentric(X[t,F[fi]],nearest);material=(uv[F[fi]]*bc[:,:,None]).sum(1)
                    B=basis(material);pairs=np.array(list(cKDTree(material).query_pairs(.065)));ll=np.linalg.norm(material[pairs[:,0]]-material[pairs[:,1]],axis=1)
                    keep=(ll>.02)&(dist[pairs]<.015).all(1)&valid[pairs].all(1);pairs=pairs[keep]
                    groups[key]=dict(B=B[pairs[:,0]]-B[pairs[:,1]],pairs=pairs,lengths=[])
                g=groups[key];pairs=g['pairs'];ll=np.linalg.norm(Q[pairs[:,0]]-Q[pairs[:,1]],axis=1);good=valid[pairs].all(1)&(ll>.005)&(ll<.10)
                g['lengths'].append(np.where(good,ll,np.nan));rows.append(dict(t=t,camera=cam,birth=birth,pairs=int(good.sum())))
        if t%100==0:print('tracks',t,flush=True)
    BB=[];TT=[];WW=[]
    for g in groups.values():
        M=np.asarray(g['lengths']);count=np.isfinite(M).sum(0);ok=count>=8
        if not ok.any():continue
        target=np.nanquantile(M[:,ok],.85,axis=0);spread=target-np.nanquantile(M[:,ok],.15,axis=0)
        BB.append(g['B'][ok]);TT.append(target);WW.append(np.sqrt(np.minimum(count[ok]/30,1))*.006/np.maximum(spread,.006))
    B=np.concatenate(BB);target=np.concatenate(TT);weight=np.concatenate(WW);mirror=np.array([1,0,14,13,12,11,10,9,8,7,6,5,4,3,2]);center=poly.mean(0)
    def residual(q):
        p=q.reshape(15,2);delta=np.linalg.norm(B@p,axis=1)-target;sym=p-center-(p[mirror]-center)*[-1,1]
        return np.r_[delta*weight,.15*(p-poly).ravel(),.4*sym.ravel(),.1*((p-np.roll(p,1,axis=0))-(poly-np.roll(poly,1,axis=0))).ravel()]
    fit=least_squares(residual,poly.ravel(),bounds=((poly-.08).ravel(),(poly+.08).ravel()),loss='soft_l1',f_scale=.004,max_nfev=100)
    q=fit.x.reshape(15,2);q-=q.mean(0);v,f,n,pair=make_panels(q,cell=.020,layers=1)
    np.savez_compressed(out/'asset.npz',V=v,F=f,outline=q,panel_pair=pair,layers=1)
    before=np.linalg.norm(B@poly,axis=1)-target;after=np.linalg.norm(B@q,axis=1)-target
    result=dict(frames=len(X),cameras=['top','side'],groups=len(groups),measurements=len(B),frames_with_pairs=len({r['t'] for r in rows if r['pairs']}),per_frame=rows,span_m=np.ptp(q,axis=0).tolist(),before_weighted_rmse_mm=float(np.sqrt(np.mean((weight*before)**2))*1000),after_weighted_rmse_mm=float(np.sqrt(np.mean((weight*after)**2))*1000),before_abs_median_mm=float(np.median(abs(before))*1000),after_abs_median_mm=float(np.median(abs(after))*1000),asset_sha256=hashlib.sha256((out/'asset.npz').read_bytes()).hexdigest(),limitations=['Track births mapped through imperfect prior mesh','Depth can switch visible layers after occlusion','Hidden cloth shape remains underconstrained'],seed=str(seed))
    (out/'build.json').write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items() if k!='per_frame'},flush=True)
if __name__=='__main__':main()
