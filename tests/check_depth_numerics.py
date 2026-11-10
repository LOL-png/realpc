"""CPU checks for depth correspondence identity and unobservable ICP motion."""
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from realpc.meshtrack import CamView,MeshTracker

# Unchecked high-resolution neighbours must never enter the ICP measurement.
v=CamView.__new__(CamView)
v.depth_generated=True
v.associated_points={1:np.array([[.1,.2,.3],[.2,.3,.4]])}
v.D=np.full((8,8),.9);v.labels=np.ones((2,2),np.uint8)
assert np.array_equal(v.masked_points(1),v.associated_points[1])
assert len(v.masked_points(2))==0
print('PASS: ICP uses exactly the depth returns that passed rejection')

# A featureless plane supplies z/tilt, not x/y/yaw. Preserve those null modes.
u=np.linspace(-.1,.1,31);x,y=np.meshgrid(u,u)
p=np.c_[x.ravel(),y.ravel(),np.zeros(x.size)]
o=SimpleNamespace(icp_pts=p,icp_nrm=np.tile([0.,0.,1.],(len(p),1)),tree=cKDTree(p))
tracker=SimpleNamespace(cfg={'depth_mesh':True,'huber':.005})
T=np.eye(4);T[:3,3]=[.004,-.003,.006]
fit,res=MeshTracker.icp_masked(tracker,o,T,p)
assert np.allclose(fit[:2,3],T[:2,3],atol=1e-8)
assert abs(fit[2,3])<1e-6
assert np.allclose(fit[:3,:3],np.eye(3),atol=1e-8)
print('PASS: planar ICP fixes observable depth, preserves unobservable lateral/yaw prediction')
