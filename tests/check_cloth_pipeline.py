"""Live and conversion use the same sparse masks and newly generated tracks."""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from realpc.cloth_pipeline import CausalClothPipeline
config=dict(template='shirt',shirt_layers=1,flat_rest=True,material_slack=1.3,length_slack=1.35,
    cell=.03,init_registration_iters=16,init_registration_strain=200,init_registration_contour=150,
    arap=2,data=4,coverage=8,coverage_radius=.12,cot_weight=32,cot_anchor='front',robust_depth=True,
    point_grasp=True,bending=1,joint_physics=True,strain_stiffness=5000,strain_limit=.02,
    ipc_contact=True,layer_visibility=True,idle_gate=True,silhouette=True,table_contact=True,
    feasible_predict=True,predict=True,mask_interval=10,mask_backend='video',iters=2)
observed=[]
for run in range(2):
    pipe=CausalClothPipeline(config);sequence=[]
    for t in range(3):
        f=dict(np.load(f'/home/howardji/data_ext/astra_cloth/auto_inputs/frames/{t:04d}.npz'))
        # The second entry path deliberately receives corrupted cached masks.
        if run:
            for c in ('top','side','left_wrist','right_wrist'):f[c+'_mask']=~f[c+'_mask']
        X,m,v=pipe.step(t,f);sequence.append(X.copy())
        assert pipe.last_packet['max_observed_tick']==t and pipe.last_packet['zero_lookahead']
        assert not m['ipc_intersections']
    summary=pipe.summary();assert summary['masks']['sam_frames']==1 and summary['masks']['sam_camera_calls']==2
    assert not summary['cached_cotracker_used'];observed.append(sequence)
    del pipe
assert np.allclose(observed[0],observed[1],rtol=0,atol=1e-8)
print('PASS real sparse-SAM/CoTracker/mesh live-conversion prefix; cached masks ignored; current-only tracks; 1/3 SAM frames')
