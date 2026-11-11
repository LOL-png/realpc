"""Fixed asset, metric bounds and causal live/replay regressions."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from realpc.cloth_physics import metric_filter,project_extension_endpoint
from realpc.cloth_causal import CausalClothTracker

v=np.array([[0.,0.,0.],[1.,0.,0.]]);e=np.array([[0,1]])
for y in (v*1.5,v*.5,np.array([[0.,0.,0.],[-1.,0.,0.]])):
    q,_=metric_filter(v,y,v,e,.02)
    assert .98-1e-10<=np.linalg.norm(q[1]-q[0])<=1.02+1e-10
    q=project_extension_endpoint(y,v,e,two_sided=True)
    assert abs(np.linalg.norm(q[1]-q[0])-1)<.003
print('PASS upper/lower metric bounds, including opposite-endpoint segment')
asset=sys.argv[1]
config=dict(template='shirt',shirt_asset=asset,flat_rest=True,shirt_layers=1,
    table_contact=True,max_stretch=.02,endpoint_projection=True,
    ipc_contact=True,joint_physics=True,init_iters=2,init_registration_iters=2,
    init_registration_strain=5000,strain_stiffness=5000,strain_limit=.005)
a=CausalClothTracker(**config);b=CausalClothTracker.from_config(config)
root=Path('/home/howardji/data_ext/astra_cloth/auto_inputs/frames')
for t in range(3):
    f=dict(np.load(root/f'{t:04d}.npz'))
    x,m,_=a.step(t,f,iters=2);y,_,_=b.step(t,{k:v.copy() for k,v in f.items()},iters=2)
    assert np.allclose(x,y,rtol=0,atol=1e-10)
    if t==0:rest=a.cm.V0.copy();sf=a.cm.sf.copy();sb=a.cm.sb.copy()
    assert np.array_equal(rest,a.cm.V0) and np.array_equal(sf,a.cm.sf) and np.array_equal(sb,a.cm.sb)
    edges=a.cm.E;ratio=np.linalg.norm(x[edges[:,0]]-x[edges[:,1]],axis=1)/np.linalg.norm(a.e0,axis=1)
    assert ratio.min()>=.98-1e-9 and ratio.max()<=1.02+1e-9
    assert not m['ipc_intersections'] and a.max_stretch==.02
try:a.step(1,f)
except AssertionError:pass
else:raise AssertionError('Out-of-order frame accepted')
print('PASS fixed rest/sample IDs, exact causal live/replay, metric bounds and crossings at t0 and later')
