"""Sparse-mask scheduling must not consume cached/future masks."""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from realpc.cloth_sparse_masks import SparseClothMasks
left=SparseClothMasks('unused',interval=5);right=SparseClothMasks('unused',interval=5)
# Deterministic current-seed oracle exercises scheduling/geometry, not SAM quality.
for m in (left,right):m._sam=lambda image,seed,cam=None:seed.copy()
for t in range(6):
    f=dict(np.load(f'/home/howardji/data_ext/astra_cloth/auto_inputs/frames/{t:04d}.npz'))
    g={k:v.copy() for k,v in f.items()}
    for k in g:
        if k.endswith('_mask'):g[k]=~g[k]
    a,ia=left.step(t,f);b,ib=right.step(t,g)
    assert all(np.array_equal(a[c],b[c]) for c in a)
    assert ia['max_observed_tick']==t and ia['sam_called']==ib['sam_called']
assert left.events[0]['tick']==0 and any(e['tick']==5 for e in left.events)
assert left.summary()['sam_camera_calls']==sum(e['camera_calls'] for e in left.events)
try:left.step(8,f)
except AssertionError:pass
else:raise AssertionError('Nonsequential packet accepted')
print('PASS cached masks ignored, exact current/past replay, sparse scheduling, actual call accounting')
