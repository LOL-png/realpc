"""Publish paired independent/common-mask metrics and contact sheets; no fitting."""
import json
from pathlib import Path
import numpy as np,cv2
D=Path('/home/howardji/data_ext/astra_rigid/planar_final');O=Path('/home/howardji/point_tracker/reports/ASTRA_Q29_1109');all_rows=[]
for n in [0,10]:
 ep=f'episode_{n:04d}';new=json.load(open(D/('audit' if n==0 else f'audit{n}')/'metrics.json'));old=json.load(open(f'/home/howardji/data_ext/astra_meshes/dataset_v2/qa/sweep/{ep}/metrics.json'))
 for ref in ['independent','tracker']:
  for key in sorted(set(old['per_frame'].get(ref,{}))&set(new['per_frame'].get(ref,{}))):
   a=dict(old['per_frame'][ref][key]);b=dict(new['per_frame'][ref][key]);ticks=sorted(set(a)&set(b));A=np.array([a[t] for t in ticks]);B=np.array([b[t] for t in ticks]);all_rows.append(dict(episode=ep,reference=ref,object_camera=key,frames=len(ticks),before_pass=float(np.mean(A>=.5)),after_pass=float(np.mean(B>=.5)),before_median_iou=float(np.median(A)),after_median_iou=float(np.median(B)),ticks=ticks))
 chosen=([0,176,384,416,560,848] if n==0 else [0,176,240,288,350,584]);ims=[]
 for t in chosen:
  p=D/('audit' if n==0 else f'audit{n}')/f'sweep_{ep}_{t:04d}.jpg';im=cv2.imread(str(p));assert im is not None,p;ims.append(im)
 sheet=np.vstack([np.hstack(ims[i:i+2]) for i in range(0,6,2)]);name='sweep_box_broom_contact_sheet.jpg' if n==0 else 'sweep_episode_0010_contact_sheet.jpg';cv2.imwrite(str(O/name),sheet)
json.dump({'scope':'Full fresh causal replays (ep0 cached current-only masks; ep10 segmentation in loop); SAME timestamps for paired comparison against five-pilot v3. Pass IoU>=.5. Independent Q28 masks only ep0; tracker is the old common causal-SAM reference, not new runtime masks. No manual GT.','paired':all_rows},open(O/'sweep_full_replay_alignment.json','w'),indent=2)
s=['# Full sweep replay alignment','','Fresh causal segmentation: ep0 cached, ep10 inside the loop; comparison to previous v3 pilot at matched timestamps. Pass = silhouette IoU≥0.5. Independent masks are Q28 SAM-derived; `tracker` is the old common causal-SAM reference. Neither is manual ground truth.','','| Episode | Reference | Object/view | Matched frames | Before pass | After pass | Median IoU before → after |','|---|---|---|---:|---:|---:|---:|']
for r in all_rows:s.append(f"| {r['episode']} | {r['reference']} | {r['object_camera']} | {r['frames']} | {r['before_pass']:.1%} | {r['after_pass']:.1%} | {r['before_median_iou']:.3f} → {r['after_median_iou']:.3f} |")
(O/'sweep_full_replay_alignment.md').write_text('\n'.join(s)+'\n')
for r in all_rows:
 if r['object_camera'].startswith('broom'):print({k:v for k,v in r.items() if k!='ticks'})
