"""Automatic current-frame depth seed for the single garment above a table."""
import numpy as np,cv2
from scipy.spatial import cKDTree
from .cloth_inputs import unproject


def depth_seed(frame):
    D=frame['top_depth'];P,uv=unproject(D,frame['top_K'],frame['top_T'],1)
    workspace=(P[:,0]>.15)&(P[:,0]<.9)&(P[:,1]>-.7)&(P[:,1]<.15)&(P[:,2]>.04)&(P[:,2]<.3)
    P=P[workspace];uv=uv[workspace]
    if len(P)<500:raise ValueError('Insufficient tabletop depth for automatic cloth initialization')
    rng=np.random.default_rng(0);test=P[rng.choice(len(P),min(4000,len(P)),replace=False)];A=np.c_[test[:,:2],np.ones(len(test))];best=0;coef=None
    for _ in range(500):
        ix=rng.choice(len(test),3,replace=False)
        try:c=np.linalg.solve(A[ix],test[ix,2])
        except np.linalg.LinAlgError:continue
        if np.linalg.norm(c[:2])>.15:continue
        good=abs(test[:,2]-A@c)<.0025
        if good.sum()>best:best=good.sum();coef=c
    if coef is None:raise ValueError('No table plane supported')
    A=np.c_[P[:,:2],np.ones(len(P))];near=abs(P[:,2]-A@coef)<.003;coef=np.linalg.lstsq(A[near],P[near,2],rcond=None)[0]
    height=P[:,2]-A@coef;good=(height>.007)&(height<.2)&(cKDTree(frame['robot']).query(P)[0]>.012)
    M=np.zeros(D.shape,np.uint8);M[uv[good,1],uv[good,0]]=1;M=cv2.morphologyEx(M,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8));nc,L,stats,_=cv2.connectedComponentsWithStats(M)
    if nc<2:raise ValueError('No elevated garment component')
    return L==(1+stats[1:,4].argmax()),coef
