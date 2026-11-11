"""Offline all-frame two-view intrinsic metric fit on a clean shirt topology.
Produces a non-flat canonical asset, not a recovered manufactured cut pattern.
Only the asset is consumed online; no frame-indexed deformation is exported.
"""
import sys,json,argparse,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix,eye,vstack
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from realpc.cloth_causal import at_image,depth_points
from realpc.cloth_inputs import project
from realpc.cloth_physics import IPCContact

def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--seed',required=True);p.add_argument('--canonical',required=True);p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    z=np.load(a.seed);c=np.load(a.canonical);F=c['F'];V=c['X'][0].copy();n=len(V);X=z['X'][:,:n];assert np.array_equal(z['F'][:len(F)],F)
    E=np.unique(np.sort(np.vstack([F[:,[0,1]],F[:,[1,2]],F[:,[0,2]]]),1),axis=0);values=[];rows=[]
    for t in range(len(X)):
        with np.load(Path(a.inputs)/'frames'/f'{t:04d}.npz') as f:
            for cam in ('top','side'):
                uv,dep=project(X[t],f[cam+'_K'],f[cam+'_T']);d,inside=at_image(f[cam+'_depth'],uv);mask,_=at_image(f[cam+'_mask'],uv);ok=inside&mask&(d>.08)&(abs(d-dep)<.006)
                Q=depth_points(uv,d,f[cam+'_K'],f[cam+'_T']);ll=np.linalg.norm(Q[E[:,0]]-Q[E[:,1]],axis=1);good=ok[E].all(1)&(ll>.003)&(ll<.06)
                values.append(np.where(good,ll,np.nan));rows.append(dict(t=t,camera=cam,edges=int(good.sum())))
        if t%100==0:print('metric',t,flush=True)
    M=np.array(values);count=np.isfinite(M).sum(0);target=np.linalg.norm(V[E[:,0]]-V[E[:,1]],axis=1);good=count>=15
    target[good]=np.nanmedian(M[:,good],axis=0);spread=np.full(len(E),.02);spread[good]=np.nanquantile(M[:,good],.75,axis=0)-np.nanquantile(M[:,good],.25,axis=0)
    weights=np.sqrt(np.minimum(count/80,1))*.003/np.maximum(spread,.003);weights[~good]=0
    rr=np.repeat(np.arange(len(E)),6);cc=np.c_[E[:,0,None]*3+np.arange(3),E[:,1,None]*3+np.arange(3)].ravel()
    prior=.35
    def fun(x):
        q=x.reshape(n,3);ll=np.linalg.norm(q[E[:,0]]-q[E[:,1]],axis=1)
        return np.r_[(ll-target)*weights,prior*(q-V).ravel()]
    def jac(x):
        q=x.reshape(n,3);d=q[E[:,0]]-q[E[:,1]];g=weights[:,None]*d/np.maximum(np.linalg.norm(d,axis=1),1e-12)[:,None]
        return vstack([coo_matrix((np.c_[g,-g].ravel(),(rr,cc)),shape=(len(E),n*3)).tocsr(),eye(n*3)*prior],format='csr')
    fit=least_squares(fun,V.ravel(),jac=jac,bounds=((V-.015).ravel(),(V+.015).ravel()),max_nfev=100,ftol=1e-8)
    Q=fit.x.reshape(n,3);contact=IPCContact(V,E,F)
    alpha=1.
    while contact.intersects(V+alpha*(Q-V)):alpha*=.5
    Q=V+alpha*(Q-V)
    # Asset construction may change rest geometry. Repair only submillimeter
    # self gaps; accepted final geometry is explicitly audited, not animated.
    contact.min_distance=.0004
    for _ in range(80):
        trial=contact.project_candidate(Q,Q)
        if contact.intersects(trial):trial=(Q+trial)/2
        if contact.intersects(trial):break
        Q=trial
    cs=contact.ipc.NormalCollisions();cs.build(contact.mesh,Q,.002)
    gap=float(np.sqrt(cs.compute_minimum_distance(contact.mesh,Q)))
    meta=Path(a.seed).with_name('template_info.json')
    angle=json.loads(meta.read_text())['fit_parameters'][0]
    R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]);center=Q[:,:2].mean(0);U=np.c_[(Q[:,:2]-center)@R,Q[:,2]-Q[:,2].min()]
    # Existing clean topology's ordered boundary, no scan remeshing.
    edges,counts=np.unique(np.sort(np.vstack([F[:,[0,1]],F[:,[1,2]],F[:,[0,2]]]),1),axis=0,return_counts=True);boundary=edges[counts==1];adj={}
    for i,j in boundary:adj.setdefault(int(i),[]).append(int(j));adj.setdefault(int(j),[]).append(int(i))
    chain=[min(adj)];prev=-1
    while True:
        nxt=next(j for j in adj[chain[-1]] if j!=prev)
        if nxt==chain[0]:break
        prev=chain[-1];chain.append(nxt)
    poly=U[chain,:2]
    landmark_ids=json.loads(meta.read_text())['initial_landmark_ids'] if meta.exists() else chain[::max(1,len(chain)//8)][:8]
    np.savez_compressed(out/'asset.npz',V=U,F=F,outline=poly,panel_pair=np.arange(n),layers=1,landmark_ids=landmark_ids,canonical_nonflat=True)
    before=np.linalg.norm(V[E[:,0]]-V[E[:,1]],axis=1)-target;after=np.linalg.norm(Q[E[:,0]]-Q[E[:,1]],axis=1)-target
    report=dict(frames=len(X),cameras=['top','side'],per_frame=rows,edges_observed=int(good.sum()),vertices=n,fit_step_kept=alpha,min_self_gap_mm=gap*1000,intersects=contact.intersects(Q),weighted_rmse_before_mm=float(np.sqrt(np.mean((before*weights)**2))*1000),weighted_rmse_after_mm=float(np.sqrt(np.mean((after*weights)**2))*1000),canonical_displacement_median_mm=float(np.median(np.linalg.norm(Q-V,axis=1))*1000),asset_sha256=hashlib.sha256((out/'asset.npz').read_bytes()).hexdigest(),note='Non-flat canonical mesh on clean single-sheet topology. Metric fitted across all frames; initial pose prior from a previous fit. Not a recovered sewing pattern; hidden layers remain unmeasured.')
    (out/'build.json').write_text(json.dumps(report,indent=2));print({k:v for k,v in report.items() if k!='per_frame'},flush=True)
if __name__=='__main__':main()
