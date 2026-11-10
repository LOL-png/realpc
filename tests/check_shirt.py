"""Clean garment topology, contact barrier and current-packet regressions."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from realpc.shirt_template import make_panels,outline
from realpc.cloth_seed import depth_seed
from realpc.cloth_physics import self_contact
from realpc.cloth_causal import CausalClothTracker
V,F,n,pair=make_panels(outline()*.5,.025)
m=trimesh.Trimesh(V,F,process=False);edges,count=np.unique(m.edges_sorted,axis=0,return_counts=True)
assert (count<=2).all() and len(m.split(only_watertight=False))==1
edge=edges[count==1];ids=np.unique(edge);lookup=np.full(len(V),-1,int);lookup[ids]=np.arange(len(ids));edge=lookup[edge]
assert connected_components(coo_matrix((np.ones(len(edge)),(edge[:,0],edge[:,1])),shape=(len(ids),len(ids))),directed=False)[0]==4
# A single sheet is a disk, with no duplicated panel or seam vertices.
sv,sf,sn,spair=make_panels(outline()*.5,.025,layers=1)
sm=trimesh.Trimesh(sv,sf,process=False)
se,sc=np.unique(sm.edges_sorted,axis=0,return_counts=True)
assert len(sv)==sn and np.array_equal(spair,np.arange(sn))
assert (sc<=2).all() and len(sm.split(only_watertight=False))==1
assert len(sv)-len(se)+len(sf)==1 and np.all(sv[:,2]==0)
be=se[sc==1];_,degree=np.unique(be,return_counts=True);assert (degree==2).all()
print('PASS single-sheet disk topology, one outline, no duplicate layer')
from realpc.cloth_physics import IPCContact
contact=IPCContact(V,m.edges_unique,F);assert not contact.intersects(V)
p=Path('/home/howardji/data_ext/astra_cloth/auto_inputs/frames');f=dict(np.load(p/'0000.npz'));M,plane=depth_seed(f)
f['top_mask'][:]=False;N,plane2=depth_seed(f);assert np.array_equal(M,N) and np.array_equal(plane,plane2)
# Two independently supplied streams, identical retained prefix. No lookahead input.
kw=dict(template='shirt',cell=.025,arap=2,data=2,coverage=3,strain_limit=.02,point_grasp=True,bending=10,joint_physics=True,ipc_contact=True,max_stretch=.02)
a=CausalClothTracker(**kw);b=CausalClothTracker(**kw)
for t in range(3):
 f=dict(np.load(p/f'{t:04d}.npz'));x,_,_=a.step(t,f,iters=2)
 if t<2:
  y,_,_=b.step(t,{k:v.copy() for k,v in f.items()},iters=2);assert np.allclose(x,y,rtol=0,atol=1e-10)
try:a.step(2,f)
except AssertionError:pass
else:raise AssertionError('Noncausal/out-of-order packet accepted')
print('PASS one sewn component, four garment openings, manifold edges, rest noncontact, automatic seed independent of hand mask, exact live/replay prefix, packet order')

from realpc.cloth_physics import IPCContact
v=np.array([[0.,0.,0.],[.1,0,0],[0,.1,0],[0,0,.002],[.1,0,.002],[0,.1,.002]])
f=np.array([[0,1,2],[3,4,5]]);e=np.array([[0,1],[1,2],[0,2],[3,4],[4,5],[3,5]])
c=IPCContact(v,e,f);target=v.copy();target[3:,2]-=.004
safe=c.filter(v,target);assert safe[3:,2].min()>0 and not c.intersects(safe) and c.last_alpha<1
print('PASS IPC continuous filter prevents a moving panel crossing another panel')

from realpc.cloth_physics import extension_filter
expanded=v*1.3;bounded,alpha=extension_filter(v,expanded,v,e,.02)
assert alpha<1 and (np.linalg.norm(bounded[e[:,0]]-bounded[e[:,1]],axis=1)<=1.020001*np.linalg.norm(v[e[:,0]]-v[e[:,1]],axis=1)).all()
print('PASS extension filter bounds every edge, including a large proposed motion')

# Coupled plane contact must descend without clipping the whole garment update.
from realpc.cloth_physics import joint_step
import scipy.sparse as sp
x=np.array([[0.,0.,.02],[.02,0.,.022]]);e=np.array([[0,1]]);plane=np.array([.1,0.,.01]);target=x-np.array([0,0,.03])
y=joint_step(x,x,e,sp.eye(2),target,np.array([],int),np.empty((0,3)),stiffness=1000,table_plane=plane)
assert (y@np.array([-.1,0.,1.])-plane[2]>=.0002-1e-10).all()
assert np.sum((y-target)**2)<np.sum((x-target)**2)
# A pixel track must leave the depth along its viewing ray unconstrained.
x=np.array([[.01,.02,1.]])
y=joint_step(x,x,np.empty((0,2),int),sp.eye(1)*1e-6,x*1e-6,np.array([],int),np.empty((0,3)),rays=(np.array([0]),np.zeros((1,3)),np.array([[0.,0.,1.]]),np.array([100.])))
assert np.linalg.norm(y[0,:2])<.001 and abs(y[0,2]-1)<1e-10
print('PASS tilted-table active-set descent and 2D-track ray null space')

from realpc.cloth_dynamics import feasible_pin_targets
pin=np.array([[0.,0.,.009],[0.,0.,.011],[.1,0.,.1]])
fixed,amount=feasible_pin_targets(pin,np.array([0.,0.,.01]))
assert amount>.001 and fixed[:,2].min()>=.0106-1e-12
assert np.allclose(fixed[1]-fixed[0],pin[1]-pin[0])
assert np.array_equal(fixed[2],pin[2])
print('PASS table-feasible grasp clusters preserve layer separation')
