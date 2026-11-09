"""Current-cloud cold localization: size-gated components + PCA pose hypotheses.
No episode-index prior, previous tracking, or future cloud is used.
"""
import itertools
import numpy as np
from scipy.spatial import cKDTree
from . import objlib


def register_components(Pl,Cl,P,C,voxel=.004,grippers=(),contact_points=None,initial_poses=()):
    O=objlib.o3d();reg=O.pipelines.registration
    src=objlib.to_pcd(Pl,Cl).voxel_down_sample(voxel);dst=objlib.to_pcd(P,C).voxel_down_sample(voxel)
    S=np.asarray(src.points);Q=np.asarray(dst.points);Cs=np.asarray(src.colors);Cq=np.asarray(dst.colors)
    if len(S)<20 or len(Q)<20:return np.eye(4),0.,float('inf'),{'failure':'insufficient points'}
    labels=np.array(dst.cluster_dbscan(eps=.012,min_points=12,print_progress=False))
    se=np.sort(np.ptp(S,axis=0));diam=np.linalg.norm(se);candidates=[]
    for k in set(labels):
        if k<0:continue
        ids=np.flatnonzero(labels==k);A=Q[ids]
        if len(A)<25:continue
        ext=np.sort(np.ptp(A,axis=0))
        if np.linalg.norm(ext)>diam*1.8 or ext[-1]<se[-1]*.23:continue
        if diam<.13 and ext[-1]>.12:continue
        # Partial surfaces may be flat, so compare only the two largest spans.
        score=np.abs(np.log((ext[-2:]+.008)/(se[-2:]+.008))).sum()
        candidates.append((score,ids))
    if not candidates:candidates=[(0.,np.arange(len(Q)))]
    candidates=sorted(candidates,key=lambda x:x[0])[:3]
    ms=S.mean(0);_,Vs=np.linalg.eigh(np.cov(S.T));best=None; ranked=[]
    for _,ids in candidates:
        A=Q[ids];mc=A.mean(0);_,Vd=np.linalg.eigh(np.cov(A.T));target=objlib.to_pcd(A,Cq[ids]);tree=cKDTree(A)
        nearby=[g for g in grippers if min(tree.query(f)[0].min() for f in g)<.06]
        poses=[np.asarray(H,float) for H in initial_poses]
        for perm in itertools.permutations(range(3)):
            for signs in itertools.product([-1.,1.],repeat=3):
                R=Vd[:,perm]@np.diag(signs)@Vs.T
                if np.linalg.det(R)<.99:continue
                T=np.eye(4);T[:3,:3]=R;T[:3,3]=mc-R@ms;poses.append(T)
        for T in poses:
            fit=reg.registration_icp(src,target,.025,T,reg.TransformationEstimationPointToPoint(),reg.ICPConvergenceCriteria(max_iteration=25));T=np.asarray(fit.transformation);W=S@T[:3,:3].T+T[:3,3];ds,ji=tree.query(W);dt,_=cKDTree(W).query(A)
            color=np.linalg.norm(Cs-Cq[ids][ji],axis=1)
            cost=np.mean(np.minimum(ds,.03))+.7*np.mean(np.minimum(dt,.03))+.003*np.mean(np.minimum(color,.5))
            coverage=(ds<.008).mean()
            cost += .03*max(0.,.25-coverage)/.25
            if nearby:
                contact_W=W if contact_points is None else np.asarray(contact_points)@T[:3,:3].T+T[:3,3]
                wt=cKDTree(contact_W)
                contact=min(max(wt.query(f)[0].min() for f in g) for g in nearby)
                cost += .8*min(contact,.04)
            ranked.append((cost,T.copy()))
            if best is None or cost<best[0]:best=(cost,T,(ds<.008).mean(),np.sqrt(np.mean(ds[ds<.008]**2)) if np.any(ds<.008) else np.inf,len(A))
    return best[1],float(best[2]),float(best[3]),{'components':len(candidates),'chosen_points':best[4],'bidirectional_cost_mm':best[0]*1000,'hypotheses':[T.tolist() for _,T in sorted(ranked,key=lambda x:x[0])[:12]]}
