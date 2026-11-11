"""Exact convex edge-upper bounds for the experimental cloth SQP step.
Clarabel API: https://clarabel.org/stable/python/getting_started_py/
"""
import numpy as np
import scipy.sparse as sp
import clarabel

def solve(H,g,C,lo,hi,X,E,rest,max_stretch):
    hi=hi.copy();hi[:len(E)]=np.inf  # Replace linear upper bounds by exact cones.
    eq=np.isfinite(lo)&np.isfinite(hi)&(abs(hi-lo)<1e-14)
    low=np.isfinite(lo)&~eq;high=np.isfinite(hi)&~eq
    mats=[C[eq],-C[low],C[high]]
    bounds=[lo[eq],-lo[low],hi[high]]
    cones=[clarabel.ZeroConeT(int(eq.sum())),clarabel.NonnegativeConeT(int(low.sum()+high.sum()))]
    # s_e = [L_e, (X_i+d_i)-(X_j+d_j)] belongs to a Lorentz cone.
    rows=(np.arange(len(E))[:,None]*4+np.arange(1,4)).ravel()
    cols_i=(E[:,0,None]*3+np.arange(3)).ravel()
    cols_j=(E[:,1,None]*3+np.arange(3)).ravel()
    A=sp.coo_matrix((np.r_[-np.ones(len(rows)),np.ones(len(rows))],(np.r_[rows,rows],np.r_[cols_i,cols_j])),shape=(4*len(E),H.shape[0])).tocsc()
    b=np.c_[rest*(1+max_stretch*.95),X[E[:,0]]-X[E[:,1]]].ravel()
    mats.append(A);bounds.append(b);cones.extend(clarabel.SecondOrderConeT(4) for _ in E)
    scale=max(float(np.max(abs(H.diagonal()))),1.)
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=100;settings.max_threads=1
    settings.tol_feas=1e-9;settings.tol_gap_abs=1e-9;settings.tol_gap_rel=1e-8
    solver=clarabel.DefaultSolver(sp.triu(H/scale,format='csc'),g/scale,sp.vstack(mats,format='csc'),np.concatenate(bounds),cones,settings)
    result=solver.solve();status=str(result.status)
    return np.asarray(result.x),dict(qp_status=status,qp_iterations=int(result.iterations),qp_primal_residual=float(result.r_prim),conic=True),status in ('Solved','AlmostSolved')
