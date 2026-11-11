"""Recorded stalled-pickup regression; requires an explicit trusted checkpoint."""
import argparse, pickle, sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
p=argparse.ArgumentParser()
p.add_argument('--checkpoint',required=True)
p.add_argument('--inputs',required=True)
p.add_argument('--tracks',required=True)
p.add_argument('--conic',action='store_true')
a=p.parse_args();results=[]
for repeat in range(2):
    with open(a.checkpoint,'rb') as f:tr,history,metrics=pickle.load(f)
    if a.conic:tr.constraint_conic=True;tr.held_step=.005;tr.held_iters=12
    t=len(history);frame=dict(np.load(Path(a.inputs)/'frames'/f'{t:04d}.npz'))
    frame.update(dict(np.load(Path(a.tracks)/f'{t:04d}.npz')))
    assert tr.constraint_qp and tr.qp_active
    rest=tr.cm.V0.copy();sf=tr.cm.sf.copy();sb=tr.cm.sb.copy()
    x,m,_=tr.step(t,frame,iters=5)
    trace=tr.ipc_state.solve_trace
    assert any(v['max_accepted_mm']>.1 for v in trace), 'Stalled on a solved QP'
    assert trace[-1]['raw_grasp_error_mm']<trace[0]['raw_grasp_error_mm']-1
    ratio=np.linalg.norm(x[tr.cm.E[:,0]]-x[tr.cm.E[:,1]],axis=1)/np.linalg.norm(tr.e0,axis=1)
    assert ratio.min()>=.98-1e-9 and ratio.max()<=1.02+1e-9
    assert not m['ipc_intersections']
    assert np.array_equal(rest,tr.cm.V0) and np.array_equal(sf,tr.cm.sf) and np.array_equal(sb,tr.cm.sb)
    results.append(x.copy())
    print('PASS repeat',repeat,'frame',t,'contact center mm',m['hard_grasp_error_mm'],
          'soft patch mm',m['grasp_error_mm'],'edge ratio',ratio.min(),ratio.max(),
          'relaxed pin steps',sum(bool(r.get('pin_equality_relaxed',False)) for r in trace))
assert np.allclose(results[0],results[1],rtol=0,atol=1e-10)
print('PASS deterministic current-packet replay, nonzero constrained progress, immutable asset and physical bounds')
