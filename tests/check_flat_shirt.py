"""The fitted flat garment retains its material metric in both entry paths."""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from realpc.cloth_causal import CausalClothTracker

config=dict(template='shirt',flat_rest=True,material_slack=1.3,length_slack=1.35,
    cell=.025,init_iters=12,arap=2,data=4,coverage=8,coverage_radius=.12,
    cot_weight=32,cot_anchor='front',robust_depth=True,point_grasp=True,
    paired_grasp=True,bending=1,joint_physics=True,strain_stiffness=5000,
    strain_limit=.02,ipc_contact=True,layer_visibility=True,idle_gate=True,
    silhouette=True)
if '--registration' in sys.argv:config.update(init_registration_iters=16,init_registration_strain=200,table_contact=True,back_on_table=True)
if '--clearance' in sys.argv:config.update(contact_clearance=.0002,feasible_predict=True,predict=True)
if '--single-layer' in sys.argv:config.update(shirt_layers=1,paired_grasp=False,table_contact=True)
if '--dynamics' in sys.argv:
    config.update(material_slack=1.15,length_slack=1.25,table_contact=True,
        dynamics=True,dynamics_gain=25.,native_init_iters=3,boundary_grasp=True)
live=CausalClothTracker(**config)
conversion=CausalClothTracker.from_config(dict(config,iters=2,out='unused'))
assert conversion.flat_rest and conversion.material_slack==config['material_slack'] and conversion.ipc_contact
saved=[]
for t in range(3):
    f=dict(np.load(f'/home/howardji/data_ext/astra_cloth/auto_inputs/frames/{t:04d}.npz'))
    f.update(dict(np.load(f'/home/howardji/data_ext/astra_cloth/auto_cot/{t:04d}.npz')))
    x,m,_=live.step(t,f,iters=2);saved.append(x.copy())
    if t==0:
        if '--registration' in sys.argv:
            assert live.strain_stiffness==5000 and live.wa==2 and live.wd==4 and live.coverage==8 and live.bending==1
            assert m['initial_registration']['rest_metric_reset'] is False
        rest=live.cm.V0.copy();faces=live.cm.sf.copy();bary=live.cm.sb.copy()
        if '--single-layer' in sys.argv:
            assert len(x)==live.template_info['panel_vertices'] and live.bend_matrix.nnz>0
        assert np.array_equal(live.prev_X,x), 't0 registration must not create velocity'
    assert np.array_equal(rest,live.cm.V0)
    assert np.array_equal(faces,live.cm.sf) and np.array_equal(bary,live.cm.sb)
    assert not m['ipc_intersections']
    if t<2:
        y,_,_=conversion.step(t,{k:v.copy() for k,v in f.items()},iters=2)
        assert np.allclose(x,y,rtol=0,atol=1e-10), (t,float(np.max(np.abs(x-y))))
        if t==1:conversion_prefix=conversion.cm.X.copy()
assert np.array_equal(conversion_prefix,conversion.cm.X)
print('PASS flat rest metric and sample identities immutable; identical live/conversion prefix with CoTracker; no crossing in three-frame smoke')
