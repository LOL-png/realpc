"""Geometric audit: nonadjacent triangle crossings (not a collision solver).
Coplanar overlaps are separately ambiguous and are not counted by the ray test.
"""
import numpy as np
from scipy.spatial import cKDTree


def intersections(X,F):
    tri=X[F];center=tri.mean(1);rad=np.linalg.norm(tri-center[:,None],axis=2).max(1)
    pairs=cKDTree(center).query_pairs(2*rad.max(),output_type='ndarray')
    if not len(pairs):return np.empty((0,2),int)
    keep=np.linalg.norm(center[pairs[:,0]]-center[pairs[:,1]],axis=1)<=rad[pairs].sum(1)
    pairs=pairs[keep];pairs=pairs[~(F[pairs[:,0],:,None]==F[pairs[:,1],None,:]).any((1,2))]
    hits=[]
    for start in range(0,len(pairs),20000):
        pp=pairs[start:start+20000];A=tri[pp[:,0]];B=tri[pp[:,1]];ok=np.zeros(len(pp),bool)
        for one,two in [(A,B),(B,A)]:
            e1=two[:,1]-two[:,0];e2=two[:,2]-two[:,0]
            for j in range(3):
                origin=one[:,j];direction=one[:,(j+1)%3]-origin
                h=np.cross(direction,e2);det=np.sum(e1*h,1);inv=np.divide(1.,det,out=np.zeros(len(det)),where=abs(det)>1e-12)
                s=origin-two[:,0];u=np.sum(s*h,1)*inv;q=np.cross(s,e1);v=np.sum(direction*q,1)*inv;t=np.sum(e2*q,1)*inv
                ok|=(abs(det)>1e-12)&(u>1e-6)&(v>1e-6)&(u+v<1-1e-6)&(t>1e-6)&(t<1-1e-6)
        hits.append(pp[ok])
    return np.concatenate(hits) if hits else np.empty((0,2),int)
