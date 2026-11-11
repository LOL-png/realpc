"""Experimental sparse SQP: solve data and linearized constraints together.
Exact edge-length/continuous collision checks still gate every accepted step.
OSQP Python API: https://osqp.org/docs/interfaces/python.html
"""
import numpy as np
import scipy.sparse as sp
import osqp
from .cloth_physics import extension_filter

def step(X,V0,E,A,rhs,pi,pt,band=.02,stiffness=2000.,contact=None,
         max_stretch=.02,endpoint_projection=False,rays=None,table_plane=None,
         metric_bounds=True,contact_projection=False,constraint_restore=False,
         compression_soft=False,max_step=.002,conic=False):
    if contact is None or not max_stretch:raise ValueError('QP requires contact and a fixed metric bound')
    n=len(X);rest=np.linalg.norm(V0[E[:,0]]-V0[E[:,1]],axis=1)
    edge=X[E[:,0]]-X[E[:,1]];length=np.linalg.norm(edge,axis=1);directions=edge/length[:,None]
    row=np.repeat(np.arange(len(E)),6);col=np.c_[E[:,0,None]*3+np.arange(3),E[:,1,None]*3+np.arange(3)].ravel()
    J=sp.coo_matrix((np.c_[directions,-directions].ravel(),(row,col)),shape=(len(E),3*n)).tocsr()
    weights=stiffness*(.01/np.maximum(rest,.001))**2;target=np.clip(length,rest*(1-band),rest*(1+band))
    H=sp.kron(A,sp.eye(3),format='csr')+J.T@sp.diags(weights)@J+sp.eye(3*n)*.1
    g=(A@X-rhs).ravel()+J.T@(weights*(length-target))
    cg,ch=contact.derivatives(X);H=H+ch;g+=cg
    ray_H=None
    if rays is not None and len(rays[0]):
        ids,origins,dirs,rw=rays;blocks=rw[:,None,None]*(np.eye(3)[None]-dirs[:,:,None]*dirs[:,None,:]);ii=ids[:,None]*3+np.arange(3)
        ray_H=sp.coo_matrix((blocks.ravel(),(np.broadcast_to(ii[:,:,None],blocks.shape).ravel(),np.broadcast_to(ii[:,None,:],blocks.shape).ravel())),shape=(3*n,3*n)).tocsr()
        rb=np.zeros((n,3));np.add.at(rb,ids,np.einsum('nij,nj->ni',blocks,origins));rb=rb.ravel();H+=ray_H;g+=ray_H@X.ravel()-rb
    # Soft goal plus a bounded hard incremental contact step. The latter may be
    # infeasible; solver failure retains the previous accepted mesh explicitly.
    pi=np.asarray(pi,int);pt=np.asarray(pt).reshape(-1,3).copy();raw_pt=pt.copy()
    if table_plane is not None and len(pi):
        normal=np.r_[-table_plane[:2],1.];height=pt@normal-table_plane[2]-.00025
        pt+=np.maximum(-height,0)[:,None]*normal/(normal@normal)
    pin_dof=(pi[:,None]*3+np.arange(3)).ravel()
    goal=(pt-X[pi]);goal*=np.minimum(1,max_step/np.maximum(np.linalg.norm(goal,axis=1),1e-12))[:,None]
    pin_weight=5e5;diag=np.zeros(3*n);diag[pin_dof]=pin_weight;H+=sp.diags(diag);g[pin_dof]+=pin_weight*(X[pi]-pt).ravel()
    constraints=[J,sp.eye(3*n,format='csr')]
    # Reserve some upper-length margin for curvature (norm linearization is a
    # lower bound). Lower linear bounds imply exact lower bounds along the step.
    # Solve inside the acceptance bounds: an absolute QP residual of 1e-7 m
    # can otherwise sit outside an exact edge-ratio/table check indefinitely.
    low=[rest*(1-max_stretch*.9)-length,np.full(3*n,-max_step)]
    high=[rest*(1+max_stretch*.5)-length,np.full(3*n,max_step)]
    if len(pi):
        selector=sp.coo_matrix((np.ones(len(pin_dof)),(np.arange(len(pin_dof)),pin_dof)),shape=(len(pin_dof),3*n)).tocsr()
        constraints.append(selector);low.append(goal.ravel());high.append(goal.ravel())
    if table_plane is not None:
        normal=np.r_[-table_plane[:2],1.];height=X@normal-table_plane[2]-.00025
        C=sp.coo_matrix((np.tile(normal,n),(np.repeat(np.arange(n),3),np.arange(3*n))),shape=(n,3*n)).tocsr()
        constraints.append(C);low.append(-height);high.append(np.full(n,np.inf))
    cs=contact.ipc.NormalCollisions();cs.build(contact.mesh,X,.001,dmin=0,broad_phase=contact.broad_phase)
    cr=[];cc=[];cv=[];cl=[]
    for c in cs.vv_collisions+cs.ev_collisions+cs.ee_collisions+cs.fv_collisions:
        dof=c.dof(X,contact.mesh.edges,contact.mesh.faces);gap=np.sqrt(max(c.compute_distance(dof),1e-24))
        ids=np.asarray(c.vertex_ids(contact.mesh.edges,contact.mesh.faces));ids=ids[ids>=0]
        grad=np.asarray(c.compute_distance_gradient(dof))/(2*gap);idx=(ids[:,None]*3+np.arange(3)).ravel()
        cr.extend([len(cl)]*len(idx));cc.extend(idx);cv.extend(grad);cl.append(contact.min_distance+2e-5-gap)
    if cl:
        constraints.append(sp.coo_matrix((cv,(cr,cc)),shape=(len(cl),3*n)).tocsr());low.append(np.array(cl));high.append(np.full(len(cl),np.inf))
    C=sp.vstack(constraints,format='csc');lo=np.concatenate(low);hi=np.concatenate(high)
    scale=max(float(np.max(abs(H.diagonal()))),1.)
    if conic:
        from .cloth_conic import solve
        direction,info,ok=solve(H,g,C,lo,hi,X,E,rest,max_stretch)
    else:
        solver=osqp.OSQP();solver.setup(P=sp.triu(H/scale,format='csc'),q=g/scale,A=C,l=lo,u=hi,verbose=False,eps_abs=1e-7,eps_rel=1e-5,max_iter=6000,polishing=True)
        solved=solver.solve();direction=solved.x;ok=solved.info.status_val in (1,2)
        info=dict(qp_status=solved.info.status,qp_iterations=int(solved.info.iter),qp_primal_residual=float(solved.info.prim_res))
    result=X.copy()
    if ok:
        candidate=X+direction.reshape(n,3);candidate,edge_alpha=extension_filter(X,candidate,V0,E,max_stretch);candidate=contact.filter(X,candidate)
        info.update(edge_alpha=float(edge_alpha),ccd_alpha=float(contact.last_alpha))
        def energy(Y):
            ratios=np.linalg.norm(Y[E[:,0]]-Y[E[:,1]],axis=1)/rest
            if ratios.min()<1-max_stretch-1e-8 or ratios.max()>1+max_stretch+1e-8:return np.inf
            if contact.intersects(Y):return np.inf
            if table_plane is not None and (Y@normal-table_plane[2]).min()<.0002-1e-8:return np.inf
            ll=ratios*rest;err=ll-np.clip(ll,rest*(1-band),rest*(1+band))
            value=.5*np.sum(Y*(A@Y))-np.sum(rhs*Y)+.5*np.sum(weights*err**2)+contact.energy(Y)+.5*pin_weight*np.sum((Y[pi]-pt)**2)
            if ray_H is not None:value+=.5*Y.ravel()@(ray_H@Y.ravel())-rb@Y.ravel()
            return float(value)
        old=energy(X)
        for halves in range(12):
            if energy(candidate)<=old+1e-9:result=candidate;break
            candidate=(X+candidate)*.5
        info['line_halves']=halves
    info.update(max_accepted_mm=float(np.max(np.linalg.norm(result-X,axis=1)))*1000,hard_error_mm=float(np.max(np.linalg.norm(result[pi]-pt,axis=1)))*1000 if len(pi) else 0.)
    info['raw_grasp_error_mm']=float(np.max(np.linalg.norm(result[pi]-raw_pt,axis=1)))*1000 if len(pi) else 0.
    if not hasattr(contact,'solve_trace'):contact.solve_trace=[]
    contact.solve_trace.append(info);contact.solve_trace=contact.solve_trace[-100:]
    return result
