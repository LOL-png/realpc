"""Recorded current-grasp regression; requires an explicit trusted checkpoint."""
import argparse
import copy
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

p=argparse.ArgumentParser()
p.add_argument('--state',required=True)
p.add_argument('--inputs',required=True)
p.add_argument('--expected-id',required=True,type=int)
a=p.parse_args()
with open(a.state,'rb') as f:tr,history,_=pickle.load(f)
t=len(history);frame=dict(np.load(Path(a.inputs)/'frames'/f'{t:04d}.npz'))
tr.tick=t;views=tr.views(frame)
G=frame['G_right'];tips=frame['tips_right'];local=(tips-G[:3,3])@G[:3,:3]
tips=tips[local[:,2]>.12]
observed=np.concatenate([views[c]['P'] for c in ('top','side')])
distance=cKDTree(tips).query(observed)[0];contact=observed[distance.argmin()]
candidates=np.asarray(tr.template_info['initial_landmark_ids'])
rest=tr.cm.V0.copy();past=copy.deepcopy(tr.cot)
for _ in range(2):
    chosen,info=tr.cot_contact_landmark(frame,views,contact,candidates)
    assert chosen==a.expected_id,(chosen,info)
    assert info['source']=='persistent_cotracker'
    assert np.array_equal(tr.cm.V0,rest)
    for c in tr.cot:
        for k in ('Q','id','off','good'):
            assert np.array_equal(tr.cot[c][k],past[c][k])
print('PASS repeatable read-only material vote',json.dumps(info))
held_only=dict(frame)
for c in ('top','side'):
    held_only[c+'_cot_visible']=np.arange(len(frame[c+'_cot_uv']))%5==0
assert tr.cot_contact_landmark(held_only,views,contact,candidates)[0] is None
future=dict(frame,max_observed_tick=t+1)
try:
    tr.cot_contact_landmark(future,views,contact,candidates)
except AssertionError:
    pass
else:
    raise AssertionError('Accepted a future packet')
print('PASS held-out tracks excluded and future packet rejected')
