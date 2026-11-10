"""Position constraints for cloth. Finite-iteration residuals are reported.
This is an approximate projection solver, not a certified continuous collision solver.
"""
import numpy as np
from scipy.spatial import cKDTree


def panel_separation(X,F,n,gap=.0015,pinned=()):
    # Material-paired front/back panels: keep a positive local signed thickness.
    top=F[(F<n).all(1)];N=np.zeros((n,3))
    normal=np.cross(X[top[:,1]]-X[top[:,0]],X[top[:,2]]-X[top[:,0]])
    for i in range(3):np.add.at(N,top[:,i],normal)
    N/=np.maximum(np.linalg.norm(N,axis=1),1e-12)[:,None]
    distance=np.sum((X[:n]-X[n:])*N,axis=1);delta=np.maximum(gap-distance,0)
    inv=np.ones(len(X));inv[np.asarray(pinned,int)]=0;den=np.maximum(inv[:n]+inv[n:],1)
    X[:n]+=N*(delta*inv[:n]/den)[:,None];X[n:]-=N*(delta*inv[n:]/den)[:,None]
    return X


def self_contact(X,old,V0,F,pinned=(),gap=.001):
    """Nonlocal vertex/triangle barriers using previous-step sidedness.
    Broad-phase centroid candidates, exact barycentric plane test. Local seams
    excluded by rest distance; contact projections respect fixed grasp vertices.
    """
    tri=X[F];center=tri.mean(1);_,ids=cKDTree(center).query(X,k=min(16,len(F)))
    vv=np.repeat(np.arange(len(X)),ids.shape[1]);ff=ids.ravel();a,b,c=tri[ff].transpose(1,0,2);p=X[vv]
    e=b-a;g=c-a;N=np.cross(e,g);norm=np.linalg.norm(N,axis=1);N/=np.maximum(norm,1e-12)[:,None]
    r=p-a;dist=np.sum(r*N,axis=1);proj=r-dist[:,None]*N
    aa=np.sum(e*e,1);bb=np.sum(e*g,1);cc=np.sum(g*g,1);d=np.sum(proj*e,1);ee=np.sum(proj*g,1);den=aa*cc-bb*bb
    u=(cc*d-bb*ee)/np.maximum(den,1e-20);v=(aa*ee-bb*d)/np.maximum(den,1e-20)
    n=len(X)//2
    same_panel=((vv<n)&(F[ff]<n).all(1))|((vv>=n)&(F[ff]>=n).all(1))
    seam=(F[ff]<n).any(1)&(F[ff]>=n).any(1)
    local=(np.linalg.norm(V0[vv]-V0[F[ff]].mean(1),axis=1)<.025)&(same_panel|seam)
    oldtri=old[F[ff]];oldN=np.cross(oldtri[:,1]-oldtri[:,0],oldtri[:,2]-oldtri[:,0]);oldN/=np.maximum(np.linalg.norm(oldN,axis=1),1e-12)[:,None]
    prev=np.sum((old[vv]-oldtri[:,0])*oldN,axis=1)
    # A changed side of an infinite triangle plane is NOT a collision: the
    # vertex may have moved around its boundary. Require old barycentric support.
    oe=oldtri[:,1]-oldtri[:,0];og=oldtri[:,2]-oldtri[:,0];orr=old[vv]-oldtri[:,0]
    oa=np.sum(oe*oe,1);ob=np.sum(oe*og,1);oc=np.sum(og*og,1);od=np.sum(orr*oe,1);of=np.sum(orr*og,1);oden=np.maximum(oa*oc-ob*ob,1e-20)
    ou=(oc*od-ob*of)/oden;ov=(oa*of-ob*od)/oden
    crossed=(prev*dist<0)&(ou>=0)&(ov>=0)&(ou+ov<=1)&(np.abs(prev)<.01)&(np.abs(dist)<.005)
    sign=np.where(crossed,np.where(prev>=0,1.,-1.),np.where(dist>=0,1.,-1.))
    valid=(u>=0)&(v>=0)&(u+v<=1)&(~local)&(norm>1e-10)&(sign*dist<gap)&((np.abs(dist)<gap)|crossed)&(~(F[ff]==vv[:,None]).any(1))
    vi=vv[valid];fi=ff[valid];normal=N[valid]*sign[valid,None];depth=gap-sign[valid]*dist[valid];bc=np.c_[1-u[valid]-v[valid],u[valid],v[valid]]
    inv=np.ones(len(X));inv[np.asarray(pinned,int)]=0
    mass=inv[vi]+np.sum(inv[F[fi]]*bc*bc,axis=1);corr=normal*(depth/np.maximum(mass,1e-12))[:,None]
    delta=np.zeros_like(X);count=np.zeros(len(X));np.add.at(delta,vi,inv[vi,None]*corr);np.add.at(count,vi,1)
    for k in range(3):np.add.at(delta,F[fi,k],-inv[F[fi,k],None]*bc[:,k,None]*corr);np.add.at(count,F[fi,k],1)
    delta/=np.maximum(count,1)[:,None]
    length=np.linalg.norm(delta,axis=1);delta*=np.minimum(1,.002/np.maximum(length,1e-12))[:,None]
    X+=delta
    return X,int(valid.sum())


def visible_vertices(X,F,K,T,shape):
    """Vertices belonging to actually frontmost rasterized material faces."""
    import torch
    from .diffcalib import Renderer
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes
    scale=.5;h,w=shape;k=K.copy();k[:2]*=scale
    R=Renderer(k,round(h*scale),round(w*scale))
    Q=(X-T[:3,3])@T[:3,:3]
    with torch.inference_mode():
        v=torch.as_tensor(Q,dtype=torch.float32,device='cuda');faces=torch.as_tensor(F,dtype=torch.int64,device='cuda')
        mesh=Meshes(verts=[R._ndc(v[None])[0]],faces=[faces])
        pix,_,_,_=rasterize_meshes(mesh,image_size=(R.H,R.W),faces_per_pixel=1,cull_backfaces=False,perspective_correct=False,max_faces_per_bin=100000)
        ids=torch.unique(pix);ids=ids[ids>=0];vi=torch.unique(faces[ids]).cpu().numpy()
    visible=np.zeros(len(X),bool);visible[vi]=True;return visible


def joint_step(X,V0,E,A,rhs,pi,pt,band=.02,stiffness=2000.,contact=None,max_stretch=0.):
    """One sparse Gauss-Newton step jointly fitting observations and edge strain.
    Fixed contact centers are eliminated; all other contact targets remain soft.
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla
    n=len(X);e=X[E[:,0]]-X[E[:,1]];length=np.linalg.norm(e,axis=1);rest=np.linalg.norm(V0[E[:,0]]-V0[E[:,1]],axis=1)
    target=np.clip(length,rest*(1-band),rest*(1+band));direction=e/np.maximum(length,1e-12)[:,None]
    row=np.repeat(np.arange(len(E)),6);col=np.c_[E[:,0,None]*3+np.arange(3),E[:,1,None]*3+np.arange(3)].ravel();val=np.c_[direction,-direction].ravel()
    J=sp.coo_matrix((val,(row,col)),shape=(len(E),3*n)).tocsr();weights=stiffness*(.01/np.maximum(rest,.001))**2
    # Keep a tangent stiffness inside the permitted strain band as well.
    H=sp.kron(A,sp.eye(3),format='csr')+J.T@sp.diags(weights)@J+sp.eye(3*n)*.1
    g=(A@X-rhs).ravel()+J.T@(weights*(length-target))
    if max_stretch:
        activate=rest*(1+max_stretch*.25);gap=rest*max_stretch*.75
        t=np.maximum((length-activate)/gap,0);safe=np.maximum(1-t,1e-8);strength=10.
        first=np.where(t>0,strength*t/safe/gap,0);second=np.where(t>0,strength/(safe*gap)**2,0)
        g+=J.T@first;H=H+J.T@sp.diags(second)@J
    if contact is not None:
        cg,ch=contact.derivatives(X);g+=cg;H=H+ch
    fixed=(np.asarray(pi)[:,None]*3+np.arange(3)).ravel();free=np.setdiff1d(np.arange(3*n),fixed)
    delta=np.zeros(3*n);delta[fixed]=(pt-X[pi]).ravel()
    delta[free]=sla.spsolve(H[free][:,free].tocsc(),-g[free]-H[free][:,fixed]@delta[fixed])
    delta=delta.reshape(-1,3);norm=np.linalg.norm(delta,axis=1);delta*=min(1.,.025/max(float(norm.max()),1e-12))
    result=X+delta
    if max_stretch:result,_=extension_filter(X,result,V0,E,max_stretch)
    if contact is not None:result=contact.filter(X,result)
    direction=result-X
    def energy(Y):
        ll=np.linalg.norm(Y[E[:,0]]-Y[E[:,1]],axis=1);rr=ll-np.clip(ll,rest*(1-band),rest*(1+band))
        value=.5*np.sum(Y*(A@Y))-np.sum(rhs*Y)+.5*np.sum(weights*rr*rr)
        if len(pi):value+=5e5*np.sum((Y[pi]-pt)**2)
        if max_stretch:
            tt=np.maximum((ll-activate)/gap,0)
            if tt.max()>=1:return np.inf
            value+=strength*np.sum(-np.log1p(-tt)-tt)
        if contact is not None:value+=contact.energy(Y)
        return value
    before=energy(X)
    for _ in range(16):
        if energy(result)<=before+1e-10:break
        direction*=.5;result=X+direction
    else:result=X.copy()

    return result


class IPCContact:
    """IPC Toolkit barrier + continuous collision filtering, initialized uncut."""
    def __init__(self,X,E,F,dhat=.001,stiffness=1e9):
        import ipctk
        self.ipc=ipctk;self.mesh=ipctk.CollisionMesh(np.asarray(X,float),np.asarray(E,np.int32),np.asarray(F,np.int32))
        if ipctk.has_intersections(self.mesh,np.asarray(X,float)):raise ValueError('IPC requires an intersection-free initial mesh')
        self.barrier=ipctk.BarrierPotential(dhat,stiffness);self.dhat=dhat;self.last_alpha=1.;self.last_contacts=0
    def derivatives(self,X):
        cs=self.ipc.NormalCollisions();cs.build(self.mesh,X,self.dhat);self.last_contacts=len(cs)
        g=self.barrier.gradient(cs,self.mesh,X);H=self.barrier.hessian(cs,self.mesh,X,self.ipc.PSDProjectionMethod.CLAMP)
        return g,H
    def filter(self,X,Y):
        alpha=float(self.ipc.compute_collision_free_stepsize(self.mesh,X,Y))
        self.last_alpha=alpha
        return X+min(1.,.9*alpha if alpha<1 else 1.)*(Y-X)
    def energy(self,X):
        cs=self.ipc.NormalCollisions();cs.build(self.mesh,X,self.dhat)
        return float(self.barrier(cs,self.mesh,X))
    def intersects(self,X):return bool(self.ipc.has_intersections(self.mesh,X))


def extension_filter(X,Y,V0,E,limit=.02):
    """Convex edge-length upper bounds; initial state must be feasible."""
    e=X[E[:,0]]-X[E[:,1]];de=(Y-X)[E[:,0]]-(Y-X)[E[:,1]];bound=np.linalg.norm(V0[E[:,0]]-V0[E[:,1]],axis=1)*(1+limit)
    aa=np.sum(de*de,1);bb=2*np.sum(e*de,1);cc=np.sum(e*e,1)-bound*bound
    if np.max(cc)>1e-9:raise ValueError('Extension filter starts outside its feasible set')
    cc=np.minimum(cc,0);disc=np.sqrt(np.maximum(bb*bb-4*aa*cc,0))
    roots=np.divide(-bb+disc,2*aa,out=np.full(len(E),np.inf),where=aa>1e-20)
    pos=(bb>=0)&(aa>1e-20);roots[pos]=-2*cc[pos]/np.maximum(bb[pos]+disc[pos],1e-30)
    alpha=float(min(1.,roots.min()));return X+(alpha*.8 if alpha<1 else 1.)*(Y-X),alpha
