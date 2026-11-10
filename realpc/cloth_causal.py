"""Causal material-mesh cloth tracker. The same step API serves live and replay.

Only current RGB/depth/masks/FK and retained past state enter step(). Masks are
supplied by OnlineClothMasks; no recorded/future cloth poses enter this module.
"""
import cv2
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import dijkstra
from .cloth_inputs import project, unproject
from .cloth_mesh import ClothMesh, tsdf_surface, decimate

CAMS = ('top','side','left_wrist','right_wrist')

def at_image(a, uv, default=0):
    xy=np.rint(uv).astype(int); h,w=a.shape[:2]
    valid=(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
    out=np.full(len(uv),default,dtype=a.dtype);out[valid]=a[xy[valid,1],xy[valid,0]]
    return out,valid

def depth_points(uv,d,K,T):
    C=np.c_[(uv[:,0]-K[0,2])*d/K[0,0],(uv[:,1]-K[1,2])*d/K[1,1],d]
    return C@T[:3,:3].T+T[:3,3]

def front_face_anchors(X,F,uv,K,T):
    """Two-sided ray/triangle test: bind image features to the FRONT material
    layer, not the closest 3D vertex from an arbitrary overlapping fold."""
    rays=np.c_[(uv[:,0]-K[0,2])/K[0,0],(uv[:,1]-K[1,2])/K[1,1],np.ones(len(uv))]@T[:3,:3].T
    tri=X[F];e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0];s=T[:3,3]-tri[:,0]
    h=np.cross(rays[:,None],e2[None]);det=np.einsum('fj,nfj->nf',e1,h)
    inv=np.divide(1.,det,out=np.zeros_like(det),where=np.abs(det)>1e-10)
    u=np.einsum('fj,nfj->nf',s,h)*inv;q=np.cross(s,e1);vv=np.einsum('nj,fj->nf',rays,q)*inv;depth=np.einsum('fj,fj->f',e2,q)[None]*inv
    valid=(np.abs(det)>1e-10)&(u>=0)&(vv>=0)&(u+vv<=1)&(depth>.08);depth[~valid]=np.inf
    fi=depth.argmin(1);dd=depth[np.arange(len(uv)),fi];bary=np.c_[1-u[np.arange(len(uv)),fi]-vv[np.arange(len(uv)),fi],u[np.arange(len(uv)),fi],vv[np.arange(len(uv)),fi]]
    ids=F[fi,np.argmax(bary,axis=1)];return ids,dd

class CausalClothTracker:
    def __init__(self, cell=.010, arap=2., data=2., track=8., pin=35., wrists=False, dense=False, coverage=0., predict=False, plane=False, strain_limit=0., wrist_flow=False, template='tsdf', grasp_layers=False, cot_weight=0., cot_anchor='nearest', motion_scale=1., robust_depth=False):
        self.cell=cell;self.wa=arap;self.wd=data;self.wt=track;self.wp=pin;self.wrists=wrists
        self.cm=None;self.tracks={};self.gray={};self.pins={};self.closed={};self.counter=0;self.tick=-1
        self.events=[];self.last_metrics={};self.dense=dense;self.prev_views=None;self.coverage=coverage;self.predict=predict;self.prev_X=None;self.prev_G={};self.plane=plane;self.strain_limit=strain_limit;self.wrist_flow=wrist_flow;self.template=template;self.grasp_layers=grasp_layers;self.cot_weight=cot_weight;self.cot_anchor=cot_anchor;self.cot={};self.cot_checks=[];self.motion_scale=motion_scale;self.robust_depth=robust_depth

    def cot_constraints(self,f,v,X):
        ids=[];targets=[];weights=[];self.cot_checks=[]
        assert int(f['max_observed_tick'])==self.tick and bool(f['zero_lookahead'])
        for c in CAMS:
            z=v[c];birth=int(f[c+'_cot_birth']);uv=f[c+'_cot_uv'];vis=f[c+'_cot_visible'];dep,_=at_image(z['D'],uv);Q=depth_points(uv,dep,z['K'],z['T'])
            if c not in self.cot or self.cot[c]['birth']!=birth:
                seed=f[c+'_cot_seed_uv'];d0,_=at_image(z['D'],seed);P=depth_points(seed,d0,z['K'],z['T']);dist,j=cKDTree(X).query(P)
                if self.cot_anchor=='front':
                    j,dmesh=front_face_anchors(X,self.cm.F,seed,z['K'],z['T']);dist=np.abs(d0-dmesh)
                good=(d0>.08)&(dist<(.020 if 'wrist' in c or self.cot_anchor=='front' else .012));off=np.einsum('nji,nj->ni',self.R[j],P-X[j])
                self.cot[c]=dict(birth=birth,id=j,off=off,good=good,Q=Q.copy(),last_valid=np.zeros(len(Q),bool))
            s=self.cot[c];valid=vis&(dep>.08)&s['good'];held=np.arange(len(Q))%5==0;offset=np.einsum('nij,nj->ni',self.R[s['id']],s['off']);target=Q-offset
            fit=valid&~held
            if 'wrist' in c:
                if not self.wrist_flow:fit[:]=False
                else:
                    fit &= s['last_valid'];target=X[s['id']]+Q-s['Q']
            # Bounded recovery after occlusion, while preserving material IDs.
            delta=target-X[s['id']];length=np.linalg.norm(delta,axis=1);target=X[s['id']]+delta*np.minimum(1,.035*self.motion_scale/np.maximum(length,1e-8))[:,None]
            ids.extend(s['id'][fit]);targets.extend(target[fit]);weights.extend(np.full(fit.sum(),self.cot_weight*(.5 if 'wrist' in c else 1.)))
            chk=valid&held;self.cot_checks.append((c,s['id'][chk],s['off'][chk],uv[chk],int(vis.sum()),int(valid.sum()),int(fit.sum()),self.tick-birth,len(uv)))
            s['Q']=Q.copy();s['last_valid']=valid.copy()
        return np.asarray(ids,int),np.asarray(targets).reshape(-1,3),np.asarray(weights)

    def vertex_flow(self,v,X):
        """Advect visible material vertices before nearest-surface association.
        Visibility is checked in the PAST frame: using only current visibility
        would incorrectly hide cloth that just lifted toward the camera.
        """
        ids=[];targets=[];weights=[]
        if self.prev_views is None:return np.array([],int),np.empty((0,3)),np.empty(0)
        for c in CAMS[:2]:
            old=self.prev_views[c];new=v[c];uv,z=project(X,old['K'],old['T']);dep,ok=at_image(old['D'],uv)
            ii=np.flatnonzero(ok&(dep>.08)&(np.abs(dep-z)<.012))
            if not len(ii):continue
            p=uv[ii].astype(np.float32).reshape(-1,1,2)
            nxt,st,err=cv2.calcOpticalFlowPyrLK(old['gray'],new['gray'],p,None,winSize=(25,25),maxLevel=4)
            back,sb,_=cv2.calcOpticalFlowPyrLK(new['gray'],old['gray'],nxt,None,winSize=(25,25),maxLevel=4)
            nd,inside=at_image(new['D'],nxt[:,0]);valid=st[:,0].astype(bool)&sb[:,0].astype(bool)&inside&(nd>.08)&(np.linalg.norm(back[:,0]-p[:,0],axis=1)<1.)&(err[:,0]<25)
            Q0=depth_points(uv[ii],dep[ii],old['K'],old['T']);Q1=depth_points(nxt[:,0],nd,new['K'],new['T']);delta=Q1-Q0
            valid &= np.linalg.norm(delta,axis=1)<.065*self.motion_scale
            ids.extend(ii[valid]);targets.extend(Q1[valid]);weights.extend(np.full(valid.sum(),3.))
        return np.asarray(ids,int),np.asarray(targets).reshape(-1,3),np.asarray(weights)

    def views(self, f):
        views={};robot=cKDTree(f['robot'])
        for c in CAMS:
            D=f[c+'_depth'].copy();M=f[c+'_mask'].copy();K=f[c+'_K'];T=f[c+'_T'];rgb=f[c+'_rgb']
            # SAM can include grippers, notably in wrist closeups. Reject only
            # measured points near the known robot and dark non-cloth pixels.
            P,uv=unproject(D,K,T,1,M)
            good=(P[:,2]>.025)&(P[:,2]<.55)&(P[:,0]>-.1)&(P[:,0]<1.1)&(P[:,1]>-.85)&(P[:,1]<.45)
            good &= robot.query(P,workers=1)[0]>.009
            hsv=cv2.cvtColor(rgb,cv2.COLOR_BGR2HSV);col=hsv[uv[:,1],uv[:,0]]
            good &= (col[:,2]>45)&(col[:,1]<100)
            M[:]=False;M[uv[good,1],uv[good,0]]=True;D[~M]=0
            if self.robust_depth:
                median=cv2.medianBlur(D,3);edge=(np.abs(D-median)>.015)|(median<=.08)
                D[edge]=0;M[edge]=False
            P,uv=unproject(D,K,T,2,M)
            ux=uv+np.array([1,0]);uy=uv+np.array([0,1]);dx,_=at_image(D,ux);dy,_=at_image(D,uy)
            Nx=depth_points(ux,dx,K,T)-P;Ny=depth_points(uy,dy,K,T)-P;N=np.cross(Nx,Ny);ln=np.linalg.norm(N,axis=1)
            valid=(dx>.08)&(dy>.08)&(np.linalg.norm(Nx,axis=1)<.03)&(np.linalg.norm(Ny,axis=1)<.03)&(ln>1e-9)
            N/=np.maximum(ln,1e-9)[:,None];N[~valid]=0
            views[c]=dict(D=D,K=K,T=T,M=M,P=P,N=N,rgb=rgb,gray=cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY))
        return views

    def initialize(self,v):
        P=np.concatenate([v[c]['P'] for c in ('top','side')]);lo=P.min(0)-.015;hi=P.max(0)+.015
        vv=[(v[c]['D'],v[c]['K'],v[c]['T'],1.,v[c]['P']) for c in ('top','side')]
        if self.template=='top':
            # A single observed cloth skin avoids TSDF welding nearby initial
            # wrinkle layers. Hidden cloth is still an approximation, not GT.
            z=v['top'];D=cv2.medianBlur(z['D'].astype(np.float32),5);h,w=D.shape
            yy,xx=np.mgrid[0:h:4,0:w:4];uv=np.c_[xx.ravel(),yy.ravel()];dd=D[yy,xx].ravel();valid=z['M'][yy,xx].ravel()&(dd>.08)
            grid=np.arange(len(dd)).reshape(xx.shape);a=grid[:-1,:-1].ravel();b=grid[:-1,1:].ravel();c=grid[1:,:-1].ravel();d=grid[1:,1:].ravel()
            F=np.vstack([np.c_[a,b,c],np.c_[b,d,c]]);F=F[valid[F].all(1)];V=depth_points(uv,dd,z['K'],z['T'])
            edge=np.stack([V[F[:,0]]-V[F[:,1]],V[F[:,1]]-V[F[:,2]],V[F[:,2]]-V[F[:,0]]],1)
            F=F[(np.linalg.norm(edge,axis=2)<.08).all(1)];used=np.unique(F);remap=np.full(len(V),-1,int);remap[used]=np.arange(len(used));V,F=V[used],remap[F]
        else:
            V,F,_=tsdf_surface(vv,lo,hi,voxel=.004,trunc=.012,support=.008)
            V,F=decimate(V,F,self.cell)
        self.cm=ClothMesh(V,F,n_samples=8000)
        self.e0=V[self.cm.E[:,0]]-V[self.cm.E[:,1]]
        E=self.cm.E;el=np.linalg.norm(self.e0,axis=1)
        self.graph=sp.coo_matrix((np.r_[el,el],(np.r_[E[:,0],E[:,1]],np.r_[E[:,1],E[:,0]])),shape=(len(V),len(V))).tocsr()
        self.R=np.tile(np.eye(3),(len(V),1,1))
        # Independent-edge batches allow fast sequential position-based strain
        # projection without updating a vertex twice within the same batch.
        used=[set() for _ in range(len(V))];colors=[]
        for aa,bb in E:
            color=0
            while color in used[aa] or color in used[bb]:color+=1
            used[aa].add(color);used[bb].add(color);colors.append(color)
        colors=np.array(colors);self.edge_batches=[np.flatnonzero(colors==c) for c in range(colors.max()+1)]

    def project_strain(self,X,pi,pt):
        E=self.cm.E;rest=np.linalg.norm(self.e0,axis=1);inv=np.ones(len(X));inv[pi]=0
        for _ in range(4):
            for batch in self.edge_batches:
                aa,bb=E[batch].T;e=X[aa]-X[bb];length=np.linalg.norm(e,axis=1)
                target=np.clip(length,rest[batch]*(1-self.strain_limit),rest[batch]*(1+self.strain_limit))
                scale=(length-target)/np.maximum(length,1e-9)/np.maximum(inv[aa]+inv[bb],1)
                correction=scale[:,None]*e;X[aa]-=inv[aa,None]*correction;X[bb]+=inv[bb,None]*correction
            X[pi]=pt
        return X

    def rotations(self,X):
        E=self.cm.E;d=X[E[:,0]]-X[E[:,1]];cov=np.zeros((len(X),3,3));a=self.e0[:,:,None]*d[:,None,:]
        np.add.at(cov,E[:,0],a);np.add.at(cov,E[:,1],a)
        U,_,Vh=np.linalg.svd(cov);R=Vh.transpose(0,2,1)@U.transpose(0,2,1)
        neg=np.linalg.det(R)<0;Vh[neg,-1]*=-1;return Vh.transpose(0,2,1)@U.transpose(0,2,1)

    def update_pins(self,f,X):
        pi=[];pt=[]
        for a in ('left','right'):
            close=float(f['q_'+a][-1])<.25
            if not close:
                if a in self.pins:self.events.append(dict(t=self.tick,arm=a,event='release'))
                self.pins.pop(a,None);self.closed[a]=False
            elif not self.closed.get(a,False):
                self.closed[a]=True;d=cKDTree(f['tips_'+a]).query(X)[0];i=int(d.argmin())
                if d[i]<.035:
                    if self.grasp_layers:
                        # A closed pinch can capture multiple folded material
                        # layers. Freeze their union ONCE, never reselect later.
                        seeds=np.flatnonzero(d<max(.012,float(d[i])+.004))
                        dist=dijkstra(self.graph,directed=False,indices=seeds,limit=.012).min(0);idx=np.flatnonzero(dist<.012)
                    else:
                        dist=dijkstra(self.graph,directed=False,indices=i,limit=.018);idx=np.flatnonzero(dist<.018)
                    G=f['G_'+a];local=(X[idx]-G[:3,3])@G[:3,:3];self.pins[a]=(idx,local)
                    self.events.append(dict(t=self.tick,arm=a,event='acquire',vertices=idx.tolist(),distance_mm=float(d[i]*1000)))
                else:self.events.append(dict(t=self.tick,arm=a,event='miss',distance_mm=float(d[i]*1000)))
            if a in self.pins:
                idx,local=self.pins[a];G=f['G_'+a];pi.extend(idx);pt.extend(local@G[:3,:3].T+G[:3,3])
        return np.asarray(pi,int),np.asarray(pt).reshape(-1,3)

    def track_features(self,v,X):
        ids=[];targets=[];weights=[];checks=[];counts={}
        cams=CAMS if self.wrists or self.wrist_flow else CAMS[:2]
        for c in cams:
            z=v[c];prev=self.tracks.get(c);now=[]
            if prev is not None and len(prev['uv']):
                p=prev['uv'].astype(np.float32).reshape(-1,1,2)
                nxt,st,err=cv2.calcOpticalFlowPyrLK(self.gray[c],z['gray'],p,None,winSize=(21,21),maxLevel=4,criteria=(3,30,.01))
                back,sb,_=cv2.calcOpticalFlowPyrLK(z['gray'],self.gray[c],nxt,None,winSize=(21,21),maxLevel=4)
                uv=nxt[:,0];dep,inside=at_image(z['D'],uv);valid=st[:,0].astype(bool)&sb[:,0].astype(bool)&inside&(dep>.08)&(np.linalg.norm(back[:,0]-p[:,0],axis=1)<.8)&(err[:,0]<22)
                Q=depth_points(uv,dep,z['K'],z['T']);valid &= np.linalg.norm(Q-prev['Q'],axis=1)<.065*self.motion_scale
                for key in prev:now.append((key,prev[key][valid]))
                now=dict(now);now['uv']=uv[valid];now['Q']=Q[valid];now['age']+=1
                offset=np.einsum('nij,nj->ni',self.R[now['id']],now['off']);target=now['Q']-offset
                if self.wrist_flow and 'wrist' in c:
                    # Relative motion cancels a fixed wrist-depth registration
                    # offset; absolute wrist geometry does NOT attract mesh.
                    target=X[now['id']]+Q[valid]-prev['Q'][valid]
                fit=~now['held'];ids.extend(now['id'][fit]);targets.extend(target[fit]);weights.extend(np.full(fit.sum(),self.wt*(.5 if 'wrist' in c else 1.)))
                held=np.flatnonzero(now['held'])
                checks.append((c,now['id'][held],now['off'][held],now['uv'][held],now['age'][held]))
            else:now={'uv':np.empty((0,2)), 'Q':np.empty((0,3)), 'id':np.empty(0,int),'off':np.empty((0,3)), 'held':np.empty(0,bool),'age':np.empty(0,int)}
            # Replenish only where CURRENT depth agrees with the material mesh.
            mask=(cv2.erode(z['M'].astype(np.uint8),np.ones((5,5),np.uint8))*255)
            for uv in now['uv']:cv2.circle(mask,tuple(np.rint(uv).astype(int)),9,0,-1)
            uv=cv2.goodFeaturesToTrack(z['gray'],maxCorners=max(1,450-len(now['uv'])),qualityLevel=.015,minDistance=8,mask=mask)
            if uv is not None and len(now['uv'])<450:
                uv=uv[:,0];dep,_=at_image(z['D'],uv);Q=depth_points(uv,dep,z['K'],z['T']);d,j=cKDTree(X).query(Q);threshold=.020 if self.wrist_flow and 'wrist' in c else .009;ok=(d<threshold)&(dep>.08)
                uv,Q,j=uv[ok],Q[ok],j[ok];off=np.einsum('nji,nj->ni',self.R[j],Q-X[j]);n=len(j)
                new=dict(uv=uv,Q=Q,id=j,off=off,held=(np.arange(self.counter,self.counter+n)%5==0),age=np.zeros(n,int));self.counter+=n
                for k in now:now[k]=np.concatenate([now[k],new[k]])
            self.tracks[c]=now;self.gray[c]=z['gray'];counts[c]=dict(active=len(now['id']),held=int(now['held'].sum()),median_age=float(np.median(now['age'])) if len(now['age']) else 0)
        return np.asarray(ids,int),np.asarray(targets).reshape(-1,3),np.asarray(weights),checks,counts

    def step(self,t,frame,iters=5):
        assert t==self.tick+1, 'Frames must be supplied once in causal order'
        self.tick=t;v=self.views(frame)
        if self.cm is None:self.initialize(v)
        X=self.cm.X.copy();Xp=X.copy();n=len(X);E=self.cm.E
        pi,pt=self.update_pins(frame,X)
        ti,tt,tw,checks,counts=self.track_features(v,X)
        if self.dense:
            vi,vt,vw=self.vertex_flow(v,X);ti=np.r_[ti,vi];tt=np.vstack([tt,vt]);tw=np.r_[tw,vw]
        if self.cot_weight:
            ci,ct,cw=self.cot_constraints(frame,v,X);ti=np.r_[ti,ci];tt=np.vstack([tt,ct]);tw=np.r_[tw,cw]
        if self.predict and self.prev_X is not None:
            velocity=.8*(Xp-self.prev_X);length=np.linalg.norm(velocity,axis=1);velocity*=np.minimum(1,.025*self.motion_scale/np.maximum(length,1e-8))[:,None]
            # Contact prediction acts on the SAME material neighborhood, never
            # whichever layer happens to approach the gripper this frame.
            motion=np.zeros_like(X);mass=np.zeros(n)
            for a,(idx,local) in self.pins.items():
                if a not in self.prev_G:continue
                gd=dijkstra(self.graph,directed=False,indices=idx,limit=.35).min(0);influence=np.exp(-(gd/.18)**2)
                dg=frame['G_'+a]@np.linalg.inv(self.prev_G[a]);delta=Xp@dg[:3,:3].T+dg[:3,3]-Xp
                motion+=influence[:,None]*delta;mass+=influence
            contact=motion/np.maximum(mass,1)[:,None]
            blend=np.minimum(mass,1);velocity=(1-blend[:,None])*velocity+contact
            X=Xp+velocity
        prior=X.copy()
        cams=CAMS if self.wrists else CAMS[:2];trees={c:cKDTree(v[c]['P']) for c in cams if len(v[c]['P'])}
        for iteration in range(iters):
            R=self.rotations(X);edge=np.einsum('nij,nj->ni',.5*(R[E[:,0]]+R[E[:,1]]),self.e0);b=np.zeros((n,3));np.add.at(b,E[:,0],edge);np.add.at(b,E[:,1],-edge)
            w=np.full(n,.15);rhs=.15*prior+self.wa*b
            for c,tree in trees.items():
                z=v[c];uv,depth=project(X,z['K'],z['T']);dd,inside=at_image(z['D'],uv)
                # Hidden layer vertices must not chase the visible fold surface.
                visible=inside&(dd>.08)&(depth<dd+.012)&(depth>dd-.045)
                d,j=tree.query(X);ok=visible&(d<.035);ww=self.wd*(.35 if 'wrist' in c else 1.)
                target=z['P'][j[ok]]
                if self.plane:
                    normal=z['N'][j[ok]];delta=target-X[ok];target=X[ok]+.1*delta+.9*np.sum(normal*delta,axis=1)[:,None]*normal
                w[ok]+=ww;rhs[ok]+=ww*target
                if self.coverage:
                    # Symmetric surface fit: a mesh can sit on the cloth while
                    # leaving large measured regions uncovered. Normalize by
                    # vertex occupancy, rather than letting dense views win.
                    eligible=inside&(depth>.08)&((dd<.08)|(depth<dd+.012))
                    jj=np.flatnonzero(eligible)
                    if len(jj):
                        ds,ii=cKDTree(X[jj]).query(z['P']);good=ds<.045;ii=jj[ii[good]];Y=z['P'][good]
                        sums=np.zeros((n,3));np.add.at(sums,ii,Y);num=np.bincount(ii,minlength=n);hit=num>0
                        w[hit]+=self.coverage;rhs[hit]+=self.coverage*sums[hit]/num[hit,None]
            if len(ti):np.add.at(w,ti,tw);np.add.at(rhs,ti,tw[:,None]*tt)
            if len(pi):np.add.at(w,pi,self.wp);np.add.at(rhs,pi,self.wp*pt)
            X=sla.spsolve((self.wa*self.cm.L+sp.diags(w)).tocsc(),rhs)
            if self.strain_limit:X=self.project_strain(X,pi,pt)
        self.cm.X=X;self.R=self.rotations(X);metrics=self.measure(v,checks,counts);metrics['pins']=len(pi);self.prev_views=v
        if self.dense:metrics['vertex_flow_constraints']=len(vi)
        self.prev_X=Xp;self.prev_G={a:frame['G_'+a].copy() for a in ('left','right')};self.last_metrics=metrics;return X.copy(),metrics,v

    def measure(self,v,checks,counts):
        S=self.cm.samples();tree=cKDTree(S);m={'t':self.tick,'tracks':counts,'cameras':{}}
        allP=[]
        for c in CAMS:
            z=v[c];P=z['P'];allP.append(P)
            if not len(P):continue
            d=tree.query(P)[0];uv,zz=project(S,z['K'],z['T']);dd,ok=at_image(z['D'],uv);visible=ok&(dd>.08)&(zz<dd+.012)
            dz=np.abs(zz[visible]-dd[visible]);m['cameras'][c]=dict(surface_to_mesh_med_mm=float(np.median(d)*1000),surface_to_mesh_p90_mm=float(np.percentile(d,90)*1000),surface_coverage_10mm=float((d<.01).mean()),visible_depth_med_mm=float(np.median(dz)*1000) if len(dz) else None,visible_depth_p90_mm=float(np.percentile(dz,90)*1000) if len(dz) else None,visible_samples=int(visible.sum()))
        P=np.concatenate(allP[:2]);d=cKDTree(P).query(S)[0]
        m.update(mesh_to_surface_med_mm=float(np.median(d)*1000),mesh_to_surface_p90_mm=float(np.percentile(d,90)*1000),mesh_support_10mm=float((d<.01).mean()))
        er=self.cm.edge_ratio();m['edge_ratio_p5_p50_p95']=np.percentile(er,[5,50,95]).tolist();m['heldout_tracks']={}
        for c,ids,off,uv,age in checks:
            P=self.cm.X[ids]+np.einsum('nij,nj->ni',self.R[ids],off);pred,_=project(P,v[c]['K'],v[c]['T']);e=np.linalg.norm(pred-uv,axis=1)
            long=e[age>=30]
            m['heldout_tracks'][c]=dict(n=len(e),median_px=float(np.median(e)) if len(e) else None,p90_px=float(np.percentile(e,90)) if len(e) else None,median_age=float(np.median(age)) if len(e) else None,long_n=len(long),long_median_px=float(np.median(long)) if len(long) else None,long_p90_px=float(np.percentile(long,90)) if len(long) else None)
        m['cotracker']={}
        for c,ids,off,uv,nvis,nvalid,nfit,age,total in self.cot_checks:
            P=self.cm.X[ids]+np.einsum('nij,nj->ni',self.R[ids],off);pred,_=project(P,v[c]['K'],v[c]['T']);err=np.linalg.norm(pred-uv,axis=1)
            m['cotracker'][c]=dict(queries=total,model_visible=nvis,depth_associated=nvalid,fitted=nfit,held_n=len(err),age=age,held_median_px=float(np.median(err)) if len(err) else None,held_p90_px=float(np.percentile(err,90)) if len(err) else None)
        return m
