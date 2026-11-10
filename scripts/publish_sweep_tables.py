import csv,json
from pathlib import Path
import numpy as np,cv2
D=Path('/home/howardji/data_ext/astra_rigid');O=Path('/home/howardji/point_tracker/reports/ASTRA_Q29_1109')
b=json.load(open(D/'box_audit05/metrics.json'))['records'];h=json.load(open(D/'hard50/metrics.json'))['records'];records=[]
for r in sorted(h,key=lambda x:x['episode']):
 row={'episode':r['episode'],'broom_tick':r['tick'],'broom_baseline':r['baseline']}
 for c in ['top','side']:
  q=r['cameras'][c];row.update({f'broom_{c}_'+k:v for k,v in q.items()});row[f'broom_{c}_eligible']=q['reference_pixels']>=60
  q=[x for x in b if x['episode']==r['episode'] and x['camera']==c];eligible=[x for x in q if x.get('material_silhouette',{}).get('visible_pixels',0)>=100]
  row[f'box_{c}_silhouette_pass']=sum(x['material_silhouette']['iou']>=.5 for x in eligible);row[f'box_{c}_eligible']=len(eligible)
  row[f'box_{c}_median_depth_mm']=float(np.median([x['depth_med_mm'] for x in q if x['depth_med_mm'] is not None]))
  strict=[x for x in q if x['pass'] is not None];row[f'box_{c}_strict_depth_pass']=sum(x['pass'] for x in strict);row[f'box_{c}_strict_depth_eligible']=len(strict)
 records.append(row)
with open(O/'sweep_alignment_all50.csv','w') as f:
 w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
scope='Broom: ONE previously worst audited top-view frame per episode, then current-frame cold reacquisition; NOT a full-episode replay. Baseline v3 for five pilots, legacy v1 for 45. Broom references are causal SAM masks; >=60 pixels required. Box: start/middle/end current-observation fits, same runtime fitting functions; automatic material-hull IoU>=0.5 with >=100 visible pixels. Box was absent before, so no before IoU exists. These are diagnostic views of the final mesh, not 50 full online-seg replays.'
json.dump({'scope':scope,'records':records},open(O/'sweep_alignment_all50.json','w'),indent=2)
s=['# All-50 sweep alignment audit','',scope,'','Broom entries are before → after IoU (not percentages). `occluded` means too few reference pixels to score. Box entries are passing/eligible sampled frames. These silhouettes can pass despite millimetre depth mismatch; the CSV separately includes the stricter depth gate. Meshes/priors fitted on this collection: this is reconstruction QA, not held-out generalization.','','| Episode | Broom tick / baseline | Broom top | Broom side | Box top | Box side |','|---|---:|---:|---:|---:|---:|']
for r in records:
 vals=[]
 for c in ['top','side']:
  vals.append(f"{r[f'broom_{c}_before_iou']:.3f} → {r[f'broom_{c}_after_iou']:.3f}" if r[f'broom_{c}_eligible'] else 'occluded')
 s.append(f"| {r['episode']} | {r['broom_tick']} / {r['broom_baseline']} | {vals[0]} | {vals[1]} | {r['box_top_silhouette_pass']}/{r['box_top_eligible']} | {r['box_side_silhouette_pass']}/{r['box_side_eligible']} |")
(O/'sweep_alignment_all50.md').write_text('\n'.join(s)+'\n')
for name,paths in [('broom_hard_frame_comparison',[D/'hard50'/f'episode_{i:04d}.jpg' for i in [6,10,13,30]]),('box_multiepisode_contact_sheet',[D/'box_audit05'/x for x in ['episode_0000_0000.jpg','episode_0000_0848.jpg','episode_0010_0292.jpg','episode_0020_0402.jpg']])]:
 ims=[]
 for p in paths:
  im=cv2.imread(str(p)); assert im is not None,p
  scale=960/im.shape[1];im=cv2.resize(im,(960,round(im.shape[0]*scale)));ims.append(im)
 cv2.imwrite(str(O/(name+'.jpg')),np.vstack(ims))
for c in ['top','side']:
 q=[x for x in b if x['camera']==c];strict=[x for x in q if x['pass'] is not None]
 print(c,'strict depth',sum(x['pass'] for x in strict),len(strict),'depth median',np.median([x['depth_med_mm'] for x in q if x['depth_med_mm'] is not None]))
