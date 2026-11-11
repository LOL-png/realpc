"""Offline necessary metric-bound audit; never used by the runtime tracker."""
import argparse,json,pickle,sys
from pathlib import Path
import numpy as np
from scipy.sparse.csgraph import dijkstra
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
p=argparse.ArgumentParser()
p.add_argument('--state',required=True,help='Trusted local acquisition checkpoint')
p.add_argument('--inputs',required=True)
p.add_argument('--out',required=True)
a=p.parse_args()
with open(a.state,'rb') as f:tr,history,_=pickle.load(f)
left,right=tr.pins['left'],tr.pins['right']
dist=dijkstra(tr.graph,directed=False,indices=left[0])[:,right[0]]
li=list(left[0]).index(tr.grasp_seed['left']);ri=list(right[0]).index(tr.grasp_seed['right'])
rows=[];t=len(history)-1
while (Path(a.inputs)/'frames'/f'{t:04d}.npz').exists():
    with np.load(Path(a.inputs)/'frames'/f'{t:04d}.npz') as frame:
        if any(frame['q_'+arm][-1]>=.25 for arm in ('left','right')):break
        P=left[1]@frame['G_left'][:3,:3].T+frame['G_left'][:3,3]
        Q=right[1]@frame['G_right'][:3,:3].T+frame['G_right'][:3,3]
        ratios=np.linalg.norm(P[:,None]-Q,axis=2)/dist
        rows.append(dict(frame=t,all_region_max_ratio=float(ratios.max()),contact_center_ratio=float(ratios[li,ri])))
    t+=1
result=dict(state=a.state,stop_frame=t,per_frame=rows,allowed_ratio=1+tr.max_stretch,
    note='Necessary only: Euclidean held-target distance cannot exceed a material edge path times the strain limit. Passing is not proof of feasible shape. Stops at first release; no old contact target is extrapolated afterward. All-region includes soft contact targets; center tests only selected hard-contact material points.')
for key in ('all_region_max_ratio','contact_center_ratio'):
    result[key]=max(rows,key=lambda r:r[key])
Path(a.out).write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items() if k!='per_frame'})
