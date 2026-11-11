"""Offline rest-pattern calibration from all-frame, two-view visible geometry.

An existing imperfect trajectory supplies material correspondences, NOT target
edge lengths. Only depth-supported visible endpoint pairs supply measurements.
The saved asset contains a flat pattern, never a pose sequence. This first
alternation is a robust intrinsic surrogate, not a global silhouette optimum.
"""
import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import RBFInterpolator
from scipy.optimize import least_squares
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from realpc.shirt_template import outline,make_panels
from realpc.cloth_inputs import project
from realpc.cloth_causal import at_image,depth_points

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--inputs',required=True);ap.add_argument('--seed',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    root=Path(a.seed);info=json.loads((root/'template_info.json').read_text());z=np.load(root/'mesh.npz');n=info['panel_vertices'];X=z['X'][:,:n];V=z['V0'][:n];angle=info['fit_parameters'][0]
    R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]);uv=(V[:,:2]-V[:,:2].mean(0))@R
    base=outline();sc=np.ptp(uv,axis=0)/np.ptp(base,axis=0);poly=base*sc;poly-=poly.mean(0)
    # Translation of a triangulated pattern's vertex mean differs from outline mean.
    poly+=uv.min(0)-poly.min(0)
    W=RBFInterpolator(poly,np.eye(15),kernel='thin_plate_spline')(uv)
    pairs=np.array(list(cKDTree(uv).query_pairs(.065)));length=np.linalg.norm(uv[pairs[:,0]]-uv[pairs[:,1]],axis=1);pairs=pairs[length>.025]
    # Fixed deterministic subset controls runtime and does not select future motion.
    pairs=pairs[::max(1,len(pairs)//6000)];measure=[];audit=[]
    for t in range(len(X)):
        with np.load(Path(a.inputs)/'frames'/f'{t:04d}.npz') as f:
            for c in ('top','side'):
                pix,dep=project(X[t],f[c+'_K'],f[c+'_T']);d,inside=at_image(f[c+'_depth'],pix);mask,_=at_image(f[c+'_mask'],pix)
                valid=inside&mask&(d>.08)&(abs(d-dep)<.008)
                Q=depth_points(pix,d,f[c+'_K'],f[c+'_T']);good=valid[pairs].all(1)
                lengths=np.linalg.norm(Q[pairs[:,0]]-Q[pairs[:,1]],axis=1)
                # Reject apparent material jumps; folded pairs remain lower bounds.
                good &= (lengths>.005)&(lengths<.10)
                measure.append(np.where(good,lengths,np.nan));audit.append(dict(t=t,camera=c,pairs=int(good.sum())))
        if t%100==0:print('collect',t,flush=True)
    M=np.array(measure);count=np.isfinite(M).sum(0);keep=count>=12
    # Upper robust chord envelope approximates intrinsic length when locally flat.
    # Lower chords in folded frames cannot be treated as garment contraction.
    target=np.nanquantile(M[:,keep],.85,axis=0);spread=np.nanquantile(M[:,keep],.85,axis=0)-np.nanquantile(M[:,keep],.15,axis=0)
    B=W[pairs[keep,0]]-W[pairs[keep,1]];weight=np.sqrt(np.minimum(count[keep]/60,1))*.006/np.maximum(spread,.006)
    mirror=np.array([1,0,14,13,12,11,10,9,8,7,6,5,4,3,2]);center=poly.mean(0)
    def residual(q):
        p=q.reshape(15,2);pred=np.linalg.norm(B@p,axis=1)
        sym=p-center-(p[mirror]-center)*[-1,1]
        return np.r_[(pred-target)*weight,(p-poly).ravel()*.2,sym.ravel()*.3,((p-np.roll(p,1,axis=0))-(poly-np.roll(poly,1,axis=0))).ravel()*.15]
    fit=least_squares(residual,poly.ravel(),bounds=((poly-.06).ravel(),(poly+.06).ravel()),loss='soft_l1',f_scale=.004,max_nfev=100)
    fitted=fit.x.reshape(15,2);fitted-=fitted.mean(0)
    vv,ff,nn,pair=make_panels(fitted,cell=.018,layers=1)
    np.savez_compressed(out/'asset.npz',V=vv,F=ff,outline=fitted,panel_pair=pair,layers=1)
    np.savez_compressed(out/'measurements.npz',pairs=pairs,values=M,keep=keep,target=target,weights=weight)
    summary=dict(frames=len(X),cameras=['top','side'],per_frame=audit,seed=str(root),inputs=str(Path(a.inputs)),method='visible depth chords, robust upper envelope, symmetric 15-control rest-pattern fit; one alternation',limitations=['correspondences inherit seed errors','hidden material not measured','not a global silhouette optimum'],pairs=int(keep.sum()),frames_with_pairs=len({r['t'] for r in audit if r['pairs']}),before_median_abs_mm=float(np.median(abs(np.linalg.norm(B@poly,axis=1)-target))*1000),after_median_abs_mm=float(np.median(abs(np.linalg.norm(B@fit.x.reshape(15,2),axis=1)-target))*1000),before_span_m=np.ptp(poly,axis=0).tolist(),after_span_m=np.ptp(fitted,axis=0).tolist(),asset_sha256=hashlib.sha256((out/'asset.npz').read_bytes()).hexdigest())
    (out/'build.json').write_text(json.dumps(summary,indent=2));print({k:v for k,v in summary.items() if k!='per_frame'},flush=True)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,7));ax.plot(*np.vstack([poly,poly[0]]).T,label='previous rest pattern');ax.triplot(vv[:,0],vv[:,1],ff,lw=.3,color='teal');ax.plot(*np.vstack([fitted,fitted[0]]).T,label='all-frame asset');ax.axis('equal');ax.legend();ax.set(xlabel='m',ylabel='m',title='Fixed rest pattern (offline fit, all453 frames / two views)');fig.savefig(out/'template.png',dpi=160)
if __name__=='__main__':main()
