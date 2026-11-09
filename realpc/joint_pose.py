"""Fast current-frame two-camera contour alignment, with hard-render acceptance.
No future samples, independent evaluation masks, or per-camera pose parameters.
"""
import numpy as np
import cv2
import torch
from scipy.optimize import least_squares
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def refine_contours(o, T, views, robot_z, lid, alternatives=(), rounds=3):
    data=[]
    for v,rz in zip(views,robot_z):
        if v.labels is None:continue
        h,w=v.R.H,v.R.W;lab=cv2.resize(v.labels,(w,h),interpolation=cv2.INTER_NEAREST);m=lab==lid
        if m.sum()<40:continue
        ign=((lab>0)&(lab!=lid))|((rz<1e8)&~m)
        valid=~ign; sdf=cv2.distanceTransform((~m).astype(np.uint8),cv2.DIST_L2,5)-cv2.distanceTransform(m.astype(np.uint8),cv2.DIST_L2,5)
        boundary=m&~cv2.erode(m.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
        safe=~cv2.dilate(ign.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
        yy,xx=np.nonzero(boundary&safe);target=np.c_[xx,yy][::max(1,len(xx)//250)]
        data.append((v,m,valid,safe,sdf,target,min(1.,m.sum()/300)))
    if not data:return None,0.,0.
    def render(Tn):
        V=o.verts_t@torch.tensor(Tn[:3,:3].T,dtype=torch.float32,device='cuda')+torch.tensor(Tn[:3,3],dtype=torch.float32,device='cuda')
        zs=[d[0].zbuf(V,o.faces_t).cpu().numpy() for d in data];ious=[]
        for z,(_,m,valid,*_) in zip(zs,data):
            p=(z>0)&valid;g=m&valid;ious.append(float((p&g).sum()/max(1,(p|g).sum())))
        return np.average(ious,weights=[d[-1] for d in data]),ious,zs
    original,per0,zs=render(T);best=original;T=T.copy()
    for alt in alternatives:
        score,per,z=render(alt)
        if score>best:T=alt.copy();best=score;per0=per;zs=z
    base=T.copy();bestT=T.copy();start_per=per0
    if best>=.91:return bestT,original,float(best)
    units=np.array([.15,.15,.15,.02,.02,.02])
    for _ in range(rounds):
        pairs=[]
        for z,(v,m,valid,safe,sdf,target,wc) in zip(zs,data):
            fg=z>0;bound=fg&~cv2.erode(fg.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
            bound[[0,-1],:]=False;bound[:,[0,-1]]=False
            yy,xx=np.nonzero(bound&safe)
            if len(xx)<12:continue
            sel=np.arange(0,len(xx),max(1,len(xx)//300));xx=xx[sel];yy=yy[sel];depth=z[yy,xx];K=v.R.K.cpu().numpy()
            pc=np.c_[(xx-K[0,2])*depth/K[0,0],(yy-K[1,2])*depth/K[1,1],depth]
            pw=pc@v.Twc[:3,:3].T+v.Twc[:3,3];local=(pw-T[:3,3])@T[:3,:3]
            # Reciprocal term constrains missed appendages as well as excess contour.
            dist,j=cKDTree(np.c_[xx,yy]).query(target);ok=dist<45
            pairs.append((local,local[j[ok]],target[ok],v.Tcw,K,sdf,wc))
        if not pairs:break
        def transform(x):
            delta=x*units;D=np.eye(4);D[:3,:3]=Rotation.from_rotvec(delta[:3]).as_matrix();D[:3,3]=delta[3:];return T@D
        def residual(x):
            M=transform(x);res=[]
            for P,Q,target,Tcw,K,sdf,wc in pairs:
                def project(P):
                    W=P@M[:3,:3].T+M[:3,3];C=W@Tcw[:3,:3].T+Tcw[:3,3];return C[:,:2]/np.maximum(C[:,2:],.05)*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]
                uv=project(P);r=map_coordinates(sdf,[uv[:,1],uv[:,0]],order=1,mode='nearest')
                res.append(r/3*np.sqrt(wc/max(1,len(P))))
                if len(Q):res.append(((project(Q)-target)/5*np.sqrt(.4*wc/len(Q))).ravel())
            res.append(.07*x)
            return np.concatenate(res)
        fit=least_squares(residual,np.zeros(6),bounds=(-1.2,1.2),diff_step=1e-3,max_nfev=14,loss='soft_l1',f_scale=1.)
        candidate=transform(fit.x);score,per,znew=render(candidate)
        if score>best+.0005 and all(q>=p-.08 for q,p in zip(per,start_per)):
            best=score;bestT=candidate.copy();T=candidate;zs=znew
        else:break
    return bestT,original,float(best)
