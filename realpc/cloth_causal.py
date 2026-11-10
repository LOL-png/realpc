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
    @classmethod
    def from_config(cls, config):
        """One configuration boundary shared by live/replay and conversion."""
        import inspect
        keys=inspect.signature(cls).parameters
        return cls(**{k:v for k,v in config.items() if k in keys})

    def __init__(self, cell=.010, arap=2., data=2., track=8., pin=35., wrists=False, dense=False, coverage=0., predict=False, plane=False, strain_limit=0., wrist_flow=False, template='tsdf', grasp_layers=False, cot_weight=0., cot_anchor='nearest', motion_scale=1., robust_depth=False, material_slack=1., layer_visibility=False, paired_grasp=False, length_slack=1., point_grasp=False, bending=0., gravity=0., joint_physics=False, strain_stiffness=2000., ipc_contact=False, max_stretch=0., idle_gate=False, endpoint_projection=False, dynamics=False, silhouette=False, flat_rest=False, coverage_radius=.045, init_iters=36, boundary_grasp=False, cot_rays_only=False, support_occlusion=False, table_contact=False, native_init_iters=15, dynamics_gain=1., back_on_table=False, visible_grasp=False, landmark_grasp=False, init_landmarks=False, native_init_gain=None):
        self.native_init_gain=native_init_gain;self.init_landmarks=init_landmarks;self.landmark_grasp=landmark_grasp;self.visible_grasp=visible_grasp;self.back_on_table=back_on_table;self.dynamics_gain=dynamics_gain;self.native_init_iters=native_init_iters;self.table_contact=table_contact;self.support_occlusion=support_occlusion;self.cot_rays_only=cot_rays_only;self.ray_observations=[];self.boundary_grasp=boundary_grasp;self.grasp_seed={};self.init_iters=init_iters;self.coverage_radius=coverage_radius;self.flat_rest=flat_rest;self.silhouette=silhouette;self.dynamics=dynamics;self.endpoint_projection=endpoint_projection;self.idle_gate=idle_gate;self.max_stretch=max_stretch;self.ipc_contact=ipc_contact;self.strain_stiffness=strain_stiffness;self.joint_physics=joint_physics;self.bending=bending;self.gravity=gravity;self.point_grasp=point_grasp;self.length_slack=length_slack;self.paired_grasp=paired_grasp;self.paired_done=set();self.layer_visibility=layer_visibility;self.material_slack=material_slack;self.cell=cell;self.wa=arap;self.wd=data;self.wt=track;self.wp=pin;self.wrists=wrists
        self.cm=None;self.tracks={};self.gray={};self.pins={};self.closed={};self.counter=0;self.tick=-1
        self.events=[];self.last_metrics={};self.dense=dense;self.prev_views=None;self.coverage=coverage;self.predict=predict;self.prev_X=None;self.prev_G={};self.plane=plane;self.strain_limit=strain_limit;self.wrist_flow=wrist_flow;self.template=template;self.grasp_layers=grasp_layers;self.cot_weight=cot_weight;self.cot_anchor=cot_anchor;self.cot={};self.cot_checks=[];self.motion_scale=motion_scale;self.robust_depth=robust_depth

    def __getstate__(self):
        state=self.__dict__.copy()
        for k in ('projection_solve','projection_coupling','projection_key','ipc_state'):state.pop(k,None)
        return state

    def cot_constraints(self,f,v,X):
        ids=[];targets=[];weights=[];self.cot_checks=[];self.ray_observations=[]
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
            if self.support_occlusion:
                _,predicted_depth=project(X[s['id']]+offset,z['K'],z['T'])
                fit &= z['mobile'][s['id']] | (np.abs(predicted_depth-dep)<.015)
            if 'wrist' in c:
                if not self.wrist_flow:fit[:]=False
                else:
                    fit &= s['last_valid'];target=X[s['id']]+Q-s['Q']
            # Bounded recovery after occlusion, while preserving material IDs.
            delta=target-X[s['id']];length=np.linalg.norm(delta,axis=1);target=X[s['id']]+delta*np.minimum(1,.035*self.motion_scale/np.maximum(length,1e-8))[:,None]
            if self.cot_rays_only:
                rays=np.c_[(uv[:,0]-z['K'][0,2])/z['K'][0,0],(uv[:,1]-z['K'][1,2])/z['K'][1,1],np.ones(len(uv))]@z['T'][:3,:3].T
                rays/=np.linalg.norm(rays,axis=1)[:,None]
                self.ray_observations.append((s['id'][fit],s['off'][fit],np.broadcast_to(z['T'][:3,3],(fit.sum(),3)).copy(),rays[fit],np.full(fit.sum(),self.cot_weight*(.5 if 'wrist' in c else 1.))))
            else:
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
            ii=np.flatnonzero(ok&(dep>.08)&(np.abs(dep-z)<.012)&old.get('mesh_visible',np.ones(len(X),bool)))
            if not len(ii):continue
            p=uv[ii].astype(np.float32).reshape(-1,1,2)
            nxt,st,err=cv2.calcOpticalFlowPyrLK(old['gray'],new['gray'],p,None,winSize=(25,25),maxLevel=4)
            back,sb,_=cv2.calcOpticalFlowPyrLK(new['gray'],old['gray'],nxt,None,winSize=(25,25),maxLevel=4)
            nd,inside=at_image(new['D'],nxt[:,0]);valid=st[:,0].astype(bool)&sb[:,0].astype(bool)&inside&(nd>.08)&(np.linalg.norm(back[:,0]-p[:,0],axis=1)<1.)&(err[:,0]<25)
            Q0=depth_points(uv[ii],dep[ii],old['K'],old['T']);Q1=depth_points(nxt[:,0],nd,new['K'],new['T']);delta=Q1-Q0
            valid &= np.linalg.norm(delta,axis=1)<.065*self.motion_scale
            if self.support_occlusion:valid &= new['mobile'][ii] | (np.abs(nd-z[ii])<.015)
            ids.extend(ii[valid]);targets.extend(Q1[valid]);weights.extend(np.full(valid.sum(),3.))
        return np.asarray(ids,int),np.asarray(targets).reshape(-1,3),np.asarray(weights)

    def views(self, f):
        views={};robot=cKDTree(f['robot'])
        for c in CAMS:
            D=f[c+'_depth'].copy();M=f[c+'_mask'].copy();K=f[c+'_K'];T=f[c+'_T'];rgb=f[c+'_rgb']
            if self.template=='shirt':
                uv_robot,z_robot=project(f['robot'],K,T);xy=np.rint(uv_robot).astype(int);h,w=D.shape
                keep=(z_robot>.08)&(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
                R=np.zeros((h,w),np.uint8);R[xy[keep,1],xy[keep,0]]=1
                R=cv2.dilate(R,np.ones((5,5),np.uint8));M &= R==0
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
            views[c]=dict(D=D,K=K,T=T,M=M,P=P,N=N,rgb=rgb,gray=cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY),raw_D=f[c+'_depth'],raw_M=f[c+'_mask'],robot_pixels=R if self.template=='shirt' else np.zeros_like(M))
        return views

    def initialize(self,v):
        P=np.concatenate([v[c]['P'] for c in ('top','side')]);lo=P.min(0)-.015;hi=P.max(0)+.015
        vv=[(v[c]['D'],v[c]['K'],v[c]['T'],1.,v[c]['P']) for c in ('top','side')]
        initial_X=None
        if self.template=='shirt':
            from .shirt_template import fit_current
            V,F,initial_X,self.template_info=fit_current(v,self.cell,flat_rest=self.flat_rest,table_plane=getattr(self,'table_plane',None),back_on_table=self.back_on_table)
        elif self.template=='top':
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
        self.clean_template=V.copy() if initial_X is not None else None
        # Lock the fitted garment metric once at initialization. The clean topology
        # is retained; online fitting never resets rest lengths to hide stretch.
        if initial_X is not None:
            if not self.flat_rest:V=initial_X.copy()
            # Explicit garment-size sensitivity; fixed before the sequence.
            angle=self.template_info['fit_parameters'][0]
            axis=np.array([np.cos(angle),np.sin(angle),0.])
            delta=(V-V.mean(0))@axis
            V+=(self.material_slack-1)*delta[:,None]*axis
            dclean=(self.clean_template-self.clean_template.mean(0))@axis
            self.clean_template+=(self.material_slack-1)*dclean[:,None]*axis
            self.template_info['material_width_scale']=self.material_slack
            axis=np.array([-np.sin(angle),np.cos(angle),0.])
            delta=(V-V.mean(0))@axis;V+=(self.length_slack-1)*delta[:,None]*axis
            delta=(self.clean_template-self.clean_template.mean(0))@axis;self.clean_template+=(self.length_slack-1)*delta[:,None]*axis
            self.template_info['material_length_scale']=self.length_slack
        self.cm=ClothMesh(V,F,n_samples=8000)
        if initial_X is not None:self.cm.X=initial_X
        self.e0=V[self.cm.E[:,0]]-V[self.cm.E[:,1]]
        if self.template=='shirt':
            # Cotangent mean-curvature operator of the clean flat panels.
            # Boundary rows omitted: open hems/cuffs must not shrink.
            flat=self.clean_template;tri=flat[F];rows=[];cols=[];vals=[]
            for k in range(3):
                a=tri[:,(k+1)%3]-tri[:,k];b=tri[:,(k+2)%3]-tri[:,k]
                cot=np.sum(a*b,axis=1)/np.maximum(np.linalg.norm(np.cross(a,b),axis=1),1e-10)
                i=F[:,(k+1)%3];j=F[:,(k+2)%3]
                rows.extend([i,j,i,j]);cols.extend([j,i,i,j]);vals.extend([-cot,-cot,cot,cot])
            L=sp.coo_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=(len(V),len(V))).tocsr()
            edges=np.sort(np.concatenate([F[:,[0,1]],F[:,[1,2]],F[:,[0,2]]]),axis=1);ue,ct=np.unique(edges,axis=0,return_counts=True)
            boundary=np.unique(ue[ct==1]);mask=np.ones(len(V));mask[boundary]=0
            pair=np.asarray(self.template_info['panel_pair']);sewn=pair==np.arange(len(V));near_seam=np.unique(ue[sewn[ue].any(1)]);mask[near_seam]=0
            norm=np.asarray(abs(L).sum(1)).ravel();self.bend_op=sp.diags(mask/np.maximum(norm,1e-8))@L
            self.bend_matrix=self.bend_op.T@self.bend_op

        E=self.cm.E;el=np.linalg.norm(self.e0,axis=1)
        self.graph=sp.coo_matrix((np.r_[el,el],(np.r_[E[:,0],E[:,1]],np.r_[E[:,1],E[:,0]])),shape=(len(V),len(V))).tocsr()
        self.R=self.rotations(self.cm.X)
        # Independent-edge batches allow fast sequential position-based strain
        # projection without updating a vertex twice within the same batch.
        used=[set() for _ in range(len(V))];colors=[]
        for aa,bb in E:
            color=0
            while color in used[aa] or color in used[bb]:color+=1
            used[aa].add(color);used[bb].add(color);colors.append(color)
        colors=np.array(colors);self.edge_batches=[np.flatnonzero(colors==c) for c in range(colors.max()+1)]

    def project_strain(self,X,pi,pt):
        if self.template=='shirt':
            # Global edge projection, with exactly eliminated grasp DOFs.
            # Local Gauss-Seidel alone cannot propagate a lift across a shirt.
            E=self.cm.E;rest=np.linalg.norm(self.e0,axis=1)
            fixed=np.unique(pi);free=np.setdiff1d(np.arange(len(X)),fixed)
            if len(pi):X[pi]=pt
            key=tuple(fixed)
            if getattr(self,'projection_key',None)!=key:
                weights=(np.median(rest)/np.maximum(rest,.001))**2
                ii=np.r_[E[:,0],E[:,1],E[:,0],E[:,1]];jj=np.r_[E[:,0],E[:,1],E[:,1],E[:,0]]
                A=sp.coo_matrix((np.r_[weights,weights,-weights,-weights],(ii,jj)),shape=(len(X),len(X))).tocsr()+sp.eye(len(X))*.0003
                self.projection_weights=weights
                self.projection_solve=sla.factorized(A[free][:,free].tocsc())
                self.projection_coupling=A[free][:,fixed];self.projection_key=key
            target_X=X.copy()
            for iteration in range(160):
                e=X[E[:,0]]-X[E[:,1]];length=np.linalg.norm(e,axis=1)
                target=np.clip(length,rest*(1-self.strain_limit),rest*(1+self.strain_limit))
                rhs=.0003*target_X
                correction=e*(self.projection_weights*target/np.maximum(length,1e-12))[:,None]
                np.add.at(rhs,E[:,0],correction);np.add.at(rhs,E[:,1],-correction)
                X[free]=self.projection_solve(rhs[free]-self.projection_coupling@X[fixed])
                if iteration%10==9 and np.max(np.abs(length/rest-1))<self.strain_limit+.003:break
            return X
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

    def update_pins(self,f,X,views=None):
        pi=[];pt=[]
        if not hasattr(self,'grasp_seed'):self.grasp_seed={}
        for a in ('left','right'):
            close=float(f['q_'+a][-1])<.25
            if not close:
                if a in self.pins:self.events.append(dict(t=self.tick,arm=a,event='release'))
                self.pins.pop(a,None);self.grasp_seed.pop(a,None);self.closed[a]=False
                if hasattr(self,'paired_done'):self.paired_done.discard(a)
            elif not self.closed.get(a,False):
                self.closed[a]=True;tips=f['tips_'+a]
                if self.template=='shirt':
                    G=f['G_'+a];local_tips=(tips-G[:3,3])@G[:3,:3];distal=local_tips[:,2]>.12
                    if distal.any():tips=tips[distal]
                d=cKDTree(tips).query(X)[0];raw_distance=d.copy();landmark_target=None
                if getattr(self,'visible_grasp',False) and views is not None:
                    from .cloth_physics import visible_vertices
                    seen=np.zeros(len(X),bool)
                    for camera in CAMS[:2]:
                        z=views[camera];uv,depth=project(X,z['K'],z['T']);dd,inside=at_image(z['D'],uv)
                        seen |= inside&(dd>.08)&(np.abs(depth-dd)<.02)&visible_vertices(X,self.cm.F,z['K'],z['T'],z['D'].shape)
                    if np.any(seen&(d<.035)):d=np.where(seen,d,np.inf)
                i=int(d.argmin())
                if self.template=='shirt' and self.boundary_grasp:
                    F=self.cm.F;edges=np.sort(np.vstack([F[:,[0,1]],F[:,[1,2]],F[:,[0,2]]]),axis=1);edges,count=np.unique(edges,axis=0,return_counts=True);boundary=np.unique(edges[count==1]);bi=int(boundary[d[boundary].argmin()])
                    if d[bi]<.030:i=bi
                if getattr(self,'landmark_grasp',False) and views is not None:
                    candidates=np.asarray(self.template_info['initial_landmark_ids'],int)
                    corner=int(candidates[raw_distance[candidates].argmin()])
                    observed=np.concatenate([views[c]['P'] for c in CAMS[:2]])
                    observed_distance=cKDTree(tips).query(observed)[0]
                    nearest=int(observed_distance.argmin())
                    if raw_distance[corner]<.060 and observed_distance[nearest]<.025:
                        i=corner;landmark_target=observed[nearest].copy()
                        d=raw_distance.copy();d[i]=observed_distance[nearest]
                if d[i]<.035:
                    self.grasp_seed[a]=i
                    if self.grasp_layers:
                        # A closed pinch can capture multiple folded material
                        # layers. Freeze their union ONCE, never reselect later.
                        seeds=np.flatnonzero(d<max(.012,float(d[i])+.004))
                        dist=dijkstra(self.graph,directed=False,indices=seeds,limit=.012).min(0);idx=np.flatnonzero(dist<.012)
                    else:
                        dist=dijkstra(self.graph,directed=False,indices=i,limit=.018);idx=np.flatnonzero(dist<.018)
                    G=f['G_'+a];target=X[idx].copy()
                    if landmark_target is not None:target+=landmark_target-X[i]
                    local=(target-G[:3,3])@G[:3,:3];self.pins[a]=(idx,local)
                    self.events.append(dict(t=self.tick,arm=a,event='acquire',seed=i,boundary_prior=bool(self.boundary_grasp),landmark_contact=landmark_target is not None,vertices=idx.tolist(),distance_mm=float(d[i]*1000)))
                else:self.events.append(dict(t=self.tick,arm=a,event='miss',distance_mm=float(d[i]*1000)))
            if a in self.pins and self.template=='shirt' and getattr(self,'paired_grasp',False):
                if not hasattr(self,'paired_done'):self.paired_done=set()
                if a not in self.paired_done:
                    self.paired_done.add(a);idx,local=self.pins[a];n=self.template_info['panel_vertices']
                    pair=np.asarray(self.template_info.get('panel_pair',np.r_[np.arange(n,2*n),np.arange(n)]));other=pair[idx];other=other[np.linalg.norm(X[other]-X[idx],axis=1)<.025]
                    added=np.setdiff1d(other,idx);G=f['G_'+a]
                    self.pins[a]=(np.r_[idx,added],np.vstack([local,(X[added]-G[:3,3])@G[:3,:3]]))
                    self.events.append(dict(t=self.tick,arm=a,event='paired_panel_capture',vertices=added.tolist()))
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
                if self.support_occlusion:
                    pp=X[prev['id']]+np.einsum('nij,nj->ni',self.R[prev['id']],prev['off']);_,predicted_depth=project(pp,z['K'],z['T'])
                    valid &= z['mobile'][prev['id']] | (np.abs(predicted_depth-dep)<.015)
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
                if self.layer_visibility:
                    j,md=front_face_anchors(X,self.cm.F,uv,z['K'],z['T']);ok=(np.abs(dep-md)<threshold)&(dep>.08)
                uv,Q,j=uv[ok],Q[ok],j[ok];off=np.einsum('nji,nj->ni',self.R[j],Q-X[j]);n=len(j)
                new=dict(uv=uv,Q=Q,id=j,off=off,held=(np.arange(self.counter,self.counter+n)%5==0),age=np.zeros(n,int));self.counter+=n
                for k in now:now[k]=np.concatenate([now[k],new[k]])
            self.tracks[c]=now;self.gray[c]=z['gray'];counts[c]=dict(active=len(now['id']),held=int(now['held'].sum()),median_age=float(np.median(now['age'])) if len(now['age']) else 0)
        return np.asarray(ids,int),np.asarray(targets).reshape(-1,3),np.asarray(weights),checks,counts

    def step(self,t,frame,iters=5):
        assert t==self.tick+1, 'Frames must be supplied once in causal order'
        self.tick=t;v=self.views(frame)
        if self.table_contact and not hasattr(self,'table_plane'):
            from .cloth_seed import depth_seed
            _,self.table_plane=depth_seed(frame)
        if self.ipc_contact and t==0:iters=max(iters,self.init_iters)
        if self.dynamics and self.flat_rest and t==0:iters=max(iters,self.init_iters)
        if self.cm is None:self.initialize(v)
        if self.table_contact:self.template_info['table_plane_z_axbyc']=self.table_plane.tolist()
        X=self.cm.X.copy();Xp=X.copy();n=len(X);E=self.cm.E
        if self.ipc_contact and not hasattr(self,'ipc_state'):
            from .cloth_physics import IPCContact
            self.ipc_state=IPCContact(X,E,self.cm.F)
        pi,pt=self.update_pins(frame,X,v)
        hi,ht=pi,pt
        if self.template=='shirt' and self.point_grasp and len(pi):
            hard=[]
            for arm,(ids,_) in self.pins.items():
                center=self.grasp_seed.get(arm,int(ids[np.argmin(np.linalg.norm(self.cm.V0[ids]-self.cm.V0[ids].mean(0),axis=1))]));hard.append(center)
                if self.paired_grasp:
                    pair=np.asarray(self.template_info.get('panel_pair',np.arange(len(X))))
                    if pair[center]!=center and pair[center] in ids:hard.append(pair[center])
            choose=np.array([np.flatnonzero(pi==i)[0] for i in hard]);hi,ht=pi[choose],pt[choose]
        if t==0 and (self.init_landmarks or (self.dynamics and self.flat_rest and not self.joint_physics)):
            hi=np.asarray(self.template_info['initial_landmark_ids'],int)
            # Hem corners constrain garment placement; uncertain neck/cuff
            # silhouette matches must remain free to fit the two depth views.
            if self.init_landmarks:hi=hi[:2]
            ht=X[hi].copy()
        mobile=np.full(n,t==0,bool)
        if self.prev_X is not None:mobile |= np.linalg.norm(Xp-self.prev_X,axis=1)>.003
        if len(pi):mobile |= dijkstra(self.graph,directed=False,indices=pi,limit=.20).min(0)<.20
        for z in v.values():z['mobile']=mobile
        ti,tt,tw,checks,counts=self.track_features(v,X)
        if self.dense:
            vi,vt,vw=self.vertex_flow(v,X);ti=np.r_[ti,vi];tt=np.vstack([tt,vt]);tw=np.r_[tw,vw]
        if self.cot_weight:
            ci,ct,cw=self.cot_constraints(frame,v,X);ti=np.r_[ti,ci];tt=np.vstack([tt,ct]);tw=np.r_[tw,cw]
        idle=False
        if self.idle_gate and self.tick>0 and not len(pi) and len(ti)>20:
            tips=np.vstack([frame['tips_left'],frame['tips_right']]);near=cKDTree(X).query(tips)[0].min()
            residual=float(np.median(np.linalg.norm(tt-X[ti],axis=1)))
            velocity=float(np.median(np.linalg.norm(Xp-self.prev_X,axis=1))) if self.prev_X is not None else 0.
            idle=bool(residual<.004 and velocity<.002)
        if idle:iters=0
        if not idle and self.predict and self.prev_X is not None:
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
        if self.template=='shirt' and not idle:X[:,2]-=self.gravity
        if self.max_stretch and not self.ipc_contact:
            from .cloth_physics import extension_filter
            X,self.extension_alpha=extension_filter(Xp,X,self.cm.V0,E,self.max_stretch)
        prior=X.copy()
        # An inertial prediction is an objective target, not a feasible mesh.
        # Start Newton at the last accepted mesh; all motion passes line search.
        if self.ipc_contact or self.dynamics:X=Xp.copy()
        cams=CAMS if self.wrists else CAMS[:2];trees={c:cKDTree(v[c]['P']) for c in cams if len(v[c]['P'])}
        total_iters=iters+(self.native_init_iters if self.dynamics and self.joint_physics and t==0 else 0)
        for iteration in range(total_iters):
            R=self.rotations(X);edge=np.einsum('nij,nj->ni',.5*(R[E[:,0]]+R[E[:,1]]),self.e0);b=np.zeros((n,3));np.add.at(b,E[:,0],edge);np.add.at(b,E[:,1],-edge)
            w=np.full(n,.15);rhs=.15*prior+self.wa*b
            for c,tree in trees.items():
                z=v[c];uv,depth=project(X,z['K'],z['T']);dd,inside=at_image(z['D'],uv)
                # Hidden layer vertices must not chase the visible fold surface.
                mesh_visible=np.ones(n,bool)
                if self.layer_visibility:
                    from .cloth_physics import visible_vertices
                    mesh_visible=visible_vertices(X,self.cm.F,z['K'],z['T'],z['D'].shape)
                visible=inside&(dd>.08)&(depth<dd+.012)&(depth>dd-.045)&mesh_visible
                d,j=tree.query(X);ok=visible&(d<.035);ww=self.wd*(.35 if 'wrist' in c else 1.)
                target=z['P'][j[ok]]
                if self.plane:
                    normal=z['N'][j[ok]];delta=target-X[ok];target=X[ok]+.1*delta+.9*np.sum(normal*delta,axis=1)[:,None]*normal
                w[ok]+=ww;rhs[ok]+=ww*target
                if self.silhouette:
                    raw,inside_raw=at_image(z['raw_D'],uv);mask,_=at_image(z['raw_M'],uv);robot_pixel,_=at_image(z['robot_pixels'],uv)
                    # Outside the observed silhouette is negative evidence only
                    # when the projected surface is not occluded by the robot.
                    outside=inside_raw&(raw>.08)&(~mask.astype(bool))&(robot_pixel==0)&(depth<raw+.008)&mesh_visible&(d<.08)
                    w[outside]+=5*self.wd;rhs[outside]+=5*self.wd*z['P'][j[outside]]
                if self.coverage:
                    # Symmetric surface fit: a mesh can sit on the cloth while
                    # leaving large measured regions uncovered. Normalize by
                    # vertex occupancy, rather than letting dense views win.
                    eligible=inside&(depth>.08)&mesh_visible
                    if self.support_occlusion:eligible &= (dd<.08)|(depth<dd+.012)|mobile
                    elif not self.layer_visibility:eligible &= (dd<.08)|(depth<dd+.012)
                    jj=np.flatnonzero(eligible)
                    if len(jj):
                        ds,ii=cKDTree(X[jj]).query(z['P']);good=ds<self.coverage_radius;ii=jj[ii[good]];Y=z['P'][good]
                        sums=np.zeros((n,3));np.add.at(sums,ii,Y);num=np.bincount(ii,minlength=n);hit=num>0
                        w[hit]+=self.coverage;rhs[hit]+=self.coverage*sums[hit]/num[hit,None]
            if self.ipc_contact:
                if self.table_contact:
                    normal=np.r_[-self.table_plane[:2],1.];height=X@normal-self.table_plane[2]-.0002;under=height<0;ground=X[under]-height[under,None]*normal/(normal@normal)
                else:
                    under=X[:,2]<self.template_info['floor_z'];ground=X[under].copy();ground[:,2]=self.template_info['floor_z']
                w[under]+=1000;rhs[under]+=1000*ground
            if len(ti):np.add.at(w,ti,tw);np.add.at(rhs,ti,tw[:,None]*tt)
            if len(pi):np.add.at(w,pi,self.wp);np.add.at(rhs,pi,self.wp*pt)
            A=self.wa*self.cm.L+sp.diags(w)
            if self.template=='shirt' and self.bending:
                if not hasattr(self,'bend_matrix'):
                    # Checkpoint experiment: uniform interior curvature fallback.
                    B=sp.diags(1/np.maximum(self.cm.L.diagonal(),1))@self.cm.L;self.bend_matrix=B.T@B
                A=A+self.bending*self.bend_matrix
            if self.template=='shirt' and self.dynamics and (t>0 or not self.joint_physics or iteration>=iters):
                from .cloth_dynamics import ClothDynamics
                # At initialization, joint fitting has already advanced X.
                # Starting dynamics from Xp discards that fitted garment.
                if not hasattr(self,'dynamics_state'):self.dynamics_state=ClothDynamics(self.cm.V0,self.cm.F,X,self.template_info['floor_z'],table_plane=getattr(self,'table_plane',None))
                if iteration==(iters if t==0 and self.joint_physics else 0):dynamic_start=np.array([self.dynamics_state.previous_pins.get(int(i),Xp[i]) for i in hi]).reshape(-1,3)
                target=ht if t==0 else dynamic_start+(ht-dynamic_start)*(iteration+1)/max(iters,1)
                X=self.dynamics_state.advance(A,rhs,hi,target,duration=1/30 if t==0 else 1/30/max(iters,1),gain=(self.native_init_gain if t==0 and self.native_init_gain is not None else self.dynamics_gain))
            elif self.template=='shirt' and self.joint_physics:
                from .cloth_physics import joint_step
                rays=None
                if self.cot_rays_only and self.ray_observations:
                    ri,off,centers,dirs,rw=[np.concatenate([obs[j] for obs in self.ray_observations]) for j in range(5)]
                    rays=(ri,centers-np.einsum('nij,nj->ni',R[ri],off),dirs,rw)
                X=joint_step(X,self.cm.V0,E,A,rhs,hi,ht,self.strain_limit,self.strain_stiffness,self.ipc_state if self.ipc_contact else None,self.max_stretch,self.endpoint_projection,rays=rays,table_plane=getattr(self,'table_plane',None))
            else:X=sla.spsolve(A.tocsc(),rhs)
            if self.strain_limit and not self.joint_physics and not self.dynamics:X=self.project_strain(X,hi,ht)
            if self.template=='shirt' and not self.ipc_contact and not self.dynamics:
                from .cloth_physics import self_contact
                X,self.contact_count=self_contact(X,Xp,self.cm.V0,self.cm.F,hi)
                X[:,2]=np.maximum(X[:,2],self.template_info['floor_z'])
                if len(hi):X[hi]=ht
        if self.template=='shirt' and self.strain_limit and not self.joint_physics and not self.dynamics:X=self.project_strain(X,hi,ht)
        if self.layer_visibility:
            from .cloth_physics import visible_vertices
            for c in CAMS[:2]:v[c]['mesh_visible']=visible_vertices(X,self.cm.F,v[c]['K'],v[c]['T'],v[c]['D'].shape)
        if self.dynamics and self.flat_rest and t==0 and hasattr(self,'dynamics_state'):self.dynamics_state.data.qvel[:]=0
        self.cm.X=X;self.R=self.rotations(X)
        if t==0:
            # Initialize persistent material correspondences AFTER the garment
            # fit. Pre-fit offsets otherwise cause drift on a stationary shirt.
            self.tracks={};self.gray={};self.cot={};self.counter=0
            _,_,_,checks,counts=self.track_features(v,X)
            if self.cot_weight:self.cot_constraints(frame,v,X)
        metrics=self.measure(v,checks,counts)
        if self.table_contact:metrics['min_table_clearance_mm']=float((X@np.r_[-self.table_plane[:2],1.]-self.table_plane[2]).min()*1000)
        metrics['idle_gate']=idle;metrics['pins']=len(pi);metrics['ipc_intersections']=self.ipc_state.intersects(X) if self.ipc_contact else None;metrics['ipc_contacts']=self.ipc_state.last_contacts if self.ipc_contact else None;metrics['ipc_alpha']=self.ipc_state.last_alpha if self.ipc_contact else None;metrics['grasp_error_mm']=float(np.max(np.linalg.norm(X[pi]-pt,axis=1))*1000) if len(pi) else 0.;metrics['hard_grasp_error_mm']=float(np.max(np.linalg.norm(X[hi]-ht,axis=1))*1000) if len(hi) else 0.;self.prev_views=v
        if self.dense:metrics['vertex_flow_constraints']=len(vi)
        # Initialization is registration, not an observed time interval. Do not
        # extrapolate the t0 fitting displacement as cloth velocity at t1.
        self.prev_X=X.copy() if t==0 else Xp
        self.prev_G={a:frame['G_'+a].copy() for a in ('left','right')};self.last_metrics=metrics;return X.copy(),metrics,v

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
        er=self.cm.edge_ratio();m['edge_ratio_p5_p50_p95']=np.percentile(er,[5,50,95]).tolist();m['max_edge_strain']=float(np.max(np.abs(er-1)));m['contacts_projected']=getattr(self,'contact_count',0);m['heldout_tracks']={}
        for c,ids,off,uv,age in checks:
            P=self.cm.X[ids]+np.einsum('nij,nj->ni',self.R[ids],off);pred,_=project(P,v[c]['K'],v[c]['T']);e=np.linalg.norm(pred-uv,axis=1)
            long=e[age>=30]
            m['heldout_tracks'][c]=dict(n=len(e),median_px=float(np.median(e)) if len(e) else None,p90_px=float(np.percentile(e,90)) if len(e) else None,median_age=float(np.median(age)) if len(e) else None,long_n=len(long),long_median_px=float(np.median(long)) if len(long) else None,long_p90_px=float(np.percentile(long,90)) if len(long) else None)
        m['cotracker']={}
        for c,ids,off,uv,nvis,nvalid,nfit,age,total in self.cot_checks:
            P=self.cm.X[ids]+np.einsum('nij,nj->ni',self.R[ids],off);pred,_=project(P,v[c]['K'],v[c]['T']);err=np.linalg.norm(pred-uv,axis=1)
            m['cotracker'][c]=dict(queries=total,model_visible=nvis,depth_associated=nvalid,fitted=nfit,held_n=len(err),age=age,held_median_px=float(np.median(err)) if len(err) else None,held_p90_px=float(np.percentile(err,90)) if len(err) else None)
        return m
