"""Current-observation cardboard-tray localization; no episode pose lookup.

Used both while building the fixed mesh library and by StreamConverter at run
time. Colour selects candidate material; connected geometry rejects the block.
"""
import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

def cardboard_points(P,C):
    hsv=cv2.cvtColor((np.clip(C,0,1)[None]*255).astype(np.uint8),cv2.COLOR_RGB2HSV)[0]
    h,s,v=hsv.T
    tan=(h>=5)&(h<=35)&(s>45)&(s<230)&(v>45)
    red=((h<8)|(h>170))&(s>85)&(v>35)
    good=(tan|red)&(P[:,2]>.015)&(P[:,2]<.4)
    Q=P[good];colors=C[good]
    if len(Q)<80:raise ValueError('No cardboard support in current frame')
    _,ii=np.unique(np.floor(Q/.005).astype(int),axis=0,return_index=True);D=Q[ii]
    pairs=cKDTree(D).query_pairs(.016,output_type='ndarray')
    graph=coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(len(D),len(D)))
    _,lab=connected_components(graph,directed=False);counts=np.bincount(lab)
    candidates=[]
    for j in np.argsort(counts)[::-1][:5]:
        points=D[lab==j];span=np.ptp(points,axis=0)
        if counts[j]>60 and max(span[:2])>.15 and max(span[:2])<.65:candidates.append(j)
    if not candidates:raise ValueError('No tray-sized cardboard component')
    component=D[lab==candidates[0]];keep=cKDTree(component).query(Q)[0]<.012
    return Q[keep],colors[keep]

def fit_box(P,C):
    """Return floor-centred rigid pose, outer dimensions, selected observations."""
    Q,colors=cardboard_points(P,C)
    low=Q[Q[:,2]<np.percentile(Q[:,2],65)]
    # Explicit seeded RANSAC avoids Open3D's thread-dependent plane tie breaks.
    rng=np.random.default_rng(0);tri=low[rng.integers(0,len(low),size=(500,3))]
    normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);length=np.linalg.norm(normal,axis=1);good=length>1e-8
    normal=normal[good]/length[good,None];origin=tri[good,0];good=abs(normal[:,2])>.85;normal,origin=normal[good],origin[good]
    if not len(normal):raise ValueError('No horizontal cardboard floor support')
    offset=-np.einsum('ij,ij->i',normal,origin);sample=low[::max(1,len(low)//4000)]
    counts=(abs(sample@normal.T+offset)<.003).sum(0);j=int(np.argmax(counts));n=normal[j];d=float(offset[j])
    for _ in range(3):
        inliers=np.flatnonzero(abs(low@n+d)<.003);points=low[inliers];center=points.mean(0);_,_,Vt=np.linalg.svd(points-center,full_matrices=False);n=Vt[-1];d=-float(n@center)
    if n[2]<0:n=-n;d=-d
    if n[2]<.85:raise ValueError('Cardboard floor plane is not horizontal')
    ex=np.array([1.,0.,-n[0]/n[2]]);ex/=np.linalg.norm(ex);ey=np.cross(n,ex)
    uv=np.c_[Q@ex,Q@ey].astype(np.float32)
    # Isolated depth flying pixels cannot set the rectangle's dimensions.
    density=cKDTree(Q).query(Q,k=5)[0][:,-1];uvfit=uv[density<.010]
    rect=cv2.minAreaRect(uvfit);corners=cv2.boxPoints(rect);edges=np.roll(corners,-1,axis=0)-corners;long=edges[np.linalg.norm(edges,axis=1).argmax()];long/=np.linalg.norm(long)
    x=ex*long[0]+ey*long[1]
    if x[0]<0:x=-x
    y=np.cross(n,x);R=np.column_stack([x,y,n]);local=Q@R
    lo,hi=np.percentile(local[:,:2],[.4,99.6],axis=0);center=(lo+hi)/2
    floor=-d;z=local[:,2]-floor;height=float(np.percentile(z,99.5))
    T=np.eye(4);T[:3,:3]=R;T[:3,3]=R@np.r_[center,floor]
    dims=np.r_[hi-lo,height]
    if not (.15<dims[0]<.5 and .10<dims[1]<.4 and .025<height<.18):raise ValueError('Implausible tray dimensions '+str(dims))
    canonical=(Q-T[:3,3])@R
    return T,dims,canonical,colors,{'points':len(Q),'floor_inliers':len(inliers),'dimensions_m':dims.tolist()}

def surface_residual(P,dims):
    x,y,h=dims
    # Distance to the five observed interior/outer planes; limited to the box
    # footprint by upstream material-component selection.
    return np.minimum.reduce([np.abs(P[:,2]),np.abs(np.abs(P[:,0])-x/2),np.abs(np.abs(P[:,1])-y/2)])

def refine_box_pose(T,Q,dims,fit_dimensions=False):
    """Robust four-wall fit with fixed library dimensions; reject flying pixels.
    Q is a current-frame world cloud of cardboard, not a stored episode pose.
    """
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    dims=np.asarray(dims,float);T=T.copy();L=(Q-T[:3,3])@T[:3,:3];W=L[(L[:,2]>.012)&(L[:,2]<dims[2]+.012)][::3]
    if len(W)<100:return T,{'wall_fit':False}
    def residual(x):
        R=Rotation.from_rotvec([0,0,x[0]]).as_matrix();P=(W-np.r_[x[1:3],0])@R
        size=dims[:2]+(x[3:] if fit_dimensions else 0)
        dx=np.abs(P[:,0])-size[0]/2;dy=np.abs(P[:,1])-size[1]/2
        return np.minimum(np.abs(dx),np.abs(dy)) + np.maximum(dx,0)+np.maximum(dy,0)
    seeds=[np.r_[x,0.,0.,np.zeros(2) if fit_dimensions else []] for x in np.deg2rad([-15,-7,0,7,15])]
    bounds=([-.6,-.04,-.04,-.035,-.035],[.6,.04,.04,.015,.015]) if fit_dimensions else ([-.6,-.04,-.04],[.6,.04,.04])
    fits=[least_squares(residual,x,bounds=bounds,loss='soft_l1',f_scale=.002,max_nfev=30) for x in seeds]
    fit=min(fits,key=lambda f:np.mean(np.minimum(residual(f.x),.015)**2));D=np.eye(4);D[:3,:3]=Rotation.from_rotvec([0,0,fit.x[0]]).as_matrix();D[:2,3]=fit.x[1:3]
    return T@D,{'wall_fit':True,'fitted_dimensions_m':np.r_[dims[:2]+(fit.x[3:] if fit_dimensions else 0),dims[2]].tolist(),'wall_median_mm':float(np.median(residual(fit.x))*1000)}

def update_box_pose(T,P,C,dims,wall_planes=None):
    """Current-only passive fixture update: planar slide/yaw, robust wall/floor data.
    Box cannot attach to a gripper. Identity is gated by its preceding pose.
    """
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    hsv=cv2.cvtColor((np.clip(C,0,1)[None]*255).astype(np.uint8),cv2.COLOR_RGB2HSV)[0];h,s,v=hsv.T
    material=(((h>=5)&(h<=35)&(s>45)&(s<230))|(((h<8)|(h>170))&(s>85)))&(v>40)
    L=(P-T[:3,3])@T[:3,:3];dims=np.asarray(dims);keep=material&(np.abs(L[:,0])<dims[0]/2+.045)&(np.abs(L[:,1])<dims[1]/2+.045)&(L[:,2]>.014)&(L[:,2]<dims[2]+.015)
    W=L[keep];W=W[::max(1,len(W)//2500)]
    if len(W)<100:return T,{'accepted':False,'points':len(W)}
    def distances(x):
        R=Rotation.from_rotvec([0,0,x[0]]).as_matrix();Q=(W-np.r_[x[1:],0])@R
        if wall_planes:
            ds=[]
            for w in wall_planes:
                axis=w['axis'];a,b,c=w['coeff'];ds.append(abs(Q[:,axis]-a*Q[:,1-axis]-b*Q[:,2]-c)/np.sqrt(1+a*a+b*b))
            return np.minimum.reduce(ds)
        return np.minimum(abs(abs(Q[:,0])-dims[0]/2),abs(abs(Q[:,1])-dims[1]/2))
    def fun(x):return np.r_[distances(x)/np.sqrt(len(W)),.025*x[0],.10*x[1:]]
    fit=least_squares(fun,np.zeros(3),bounds=([-.18,-.025,-.025],[.18,.025,.025]),loss='soft_l1',f_scale=.003/np.sqrt(len(W)),max_nfev=20)
    before=float(np.mean(np.minimum(distances(np.zeros(3)),.015)));after=float(np.mean(np.minimum(distances(fit.x),.015)));accept=after<before-.00025
    if accept:
        D=np.eye(4);D[:3,:3]=Rotation.from_rotvec([0,0,fit.x[0]]).as_matrix();D[:2,3]=fit.x[1:];T=T@D
    return T,{'accepted':bool(accept),'points':len(W),'before_mm':before*1000,'after_mm':after*1000}

def refine_box_silhouette(obj,T,views,robot_z):
    """Fit current cardboard silhouette in both views; automatic material seeds.
    The convex outer tray envelope is used, with observed occluders ignored.
    """
    import copy,torch
    from .joint_pose import refine_contours
    adapted=[];rzs=[]
    V=obj.verts_t@torch.tensor(T[:3,:3].T,device='cuda',dtype=torch.float32)+torch.tensor(T[:3,3],device='cuda',dtype=torch.float32)
    for v,rz in zip(views,robot_z):
        if v.name not in ('top','side') or v.rgb is None:continue
        H,W=v.R.H,v.R.W;im=cv2.resize(v.rgb,(W,H));depth=cv2.resize(v.D,(W,H),interpolation=0);z=v.zbuf(V,obj.faces_t).cpu().numpy();pred=z>0
        region=cv2.dilate(pred.astype(np.uint8),np.ones((13,13),np.uint8)).astype(bool)
        hh,ss,vv=cv2.split(cv2.cvtColor(im,cv2.COLOR_BGR2HSV));material=(((hh>=5)&(hh<=35)&(ss>45)&(ss<230))|(((hh<8)|(hh>170))&(ss>85)))&(vv>40)&region
        # Remove foreground color (the block) when observed in front of the tray.
        material &= ~((depth>.1)&pred&(depth<z-.020))
        yy,xx=np.nonzero(material)
        if len(xx)<70:continue
        hull=cv2.convexHull(np.c_[xx,yy].astype(np.int32));mask=np.zeros((H,W),np.uint8);cv2.fillConvexPoly(mask,hull,1)
        ignored=(rz<1e8)|((depth>.1)&pred&(depth<z-.020));mask[ignored]=254
        q=copy.copy(v);q.labels=mask;adapted.append(q);rzs.append(rz)
    if not adapted:return T,{'silhouette_views':0}
    Tn,before,after=refine_contours(obj,T,adapted,rzs,1,rounds=2,planar=True)
    accepted=Tn is not None and after>before+.005
    result=Tn if accepted else T;per={}
    V=obj.verts_t@torch.tensor(result[:3,:3].T,device='cuda',dtype=torch.float32)+torch.tensor(result[:3,3],device='cuda',dtype=torch.float32)
    for v,rz in zip(adapted,rzs):
        gt=v.labels==1;valid=(v.labels!=254);pred=(v.zbuf(V,obj.faces_t).cpu().numpy()>0)&valid;gt &= valid
        per[v.name]={'iou':float((pred&gt).sum()/max(1,(pred|gt).sum())),'visible_pixels':int(gt.sum())}
    return result,{'silhouette_views':len(adapted),'sil_before':float(before),'sil_after':float(after),'accepted':bool(accepted),'per_camera':per}
