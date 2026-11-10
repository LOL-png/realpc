"""Publish fresh hang replay with matched-frame before/after numbers."""
import json,subprocess
from pathlib import Path
import cv2,numpy as np
RUN=Path('/home/howardji/data_ext/astra_rigid/hang_v2stand')
OUT=Path('/home/howardji/point_tracker/reports/ASTRA_Q29_1109');OUT.mkdir(parents=True,exist_ok=True)
new=json.load(open(RUN/'audit/metrics.json'))
old=json.load(open('/home/howardji/data_ext/astra_meshes/iter25/hang_audit/metrics.json'))
prior=json.load(open('/home/howardji/data_ext/astra_cloth/stand02/audit/metrics.json'))
def compare(reference):
    result={}
    for key,values in reference['per_frame']['independent'].items():
        before=dict(values);after=dict(new['per_frame']['independent'][key]);ticks=sorted(before.keys()&after.keys())
        a=np.array([before[t] for t in ticks]);b=np.array([after[t] for t in ticks])
        result[key]=dict(frames=len(ticks),before_pass=float((a>=.5).mean()),after_pass=float((b>=.5).mean()),before_median=float(np.median(a)),after_median=float(np.median(b)))
    return result
matched=compare(old)
payload={'reference':'Independent Q28 SAM masks, IoU>=0.5; identical evaluated timestamps for each comparison',
    'matched_vs_previous_Q29':matched,'matched_vs_first_four_peg_replay':compare(prior),'new_full_episode':new['summary'],
    'replay':str(RUN/'converted/hang/episode_0000.npz'),'mesh_commit':'39df240','ticks':499}
(OUT/'hang_v2stand_alignment.json').write_text(json.dumps(payload,indent=2))
subprocess.run(['ffmpeg','-y','-v','error','-i',str(RUN/'audit/top_overlay.mp4'),'-i',str(RUN/'audit/side_overlay.mp4'),
    '-filter_complex','[0:v][1:v]vstack=inputs=2[v]','-map','[v]','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'hang_v2stand_both_cameras.mp4')],check=True)
tiles=[]
for t in [0,100,180,280,350,420,496]:tiles.append(cv2.imread(str(RUN/'audit'/f'hang_episode_0000_{t:04d}.jpg')))
info=np.full((480,640,3),245,np.uint8)
lines=['Fresh replay: four individual stand arms','499 ticks, both cameras','Before -> after (matched Q28 masks):']
for key,r in matched.items():lines.append(f"{key}: {100*r['before_pass']:.1f}% -> {100*r['after_pass']:.1f}%")
lines+=['Pass = silhouette IoU >= 0.5','Left: RGB; right: mesh overlay','Scene occlusion affects display only.']
for i,line in enumerate(lines):cv2.putText(info,line,(18,38+i*37),0,.65,(30,30,30),1,cv2.LINE_AA)
tiles.append(info);tiles.append(np.full_like(info,245));sheet=np.vstack([np.hstack(tiles[i:i+3]) for i in [0,3,6]])
cv2.imwrite(str(OUT/'hang_v2stand_contact_sheet.jpg'),sheet)
text=['Fresh 499-frame hang replay using the four-arm stand mesh (commit39df240).','',
    '| Object | Top pass before → after | Side pass before → after |','|---|---:|---:|']
for name in ['mug','stand']:
    a,b=matched[name+'/top'],matched[name+'/side'];text.append(f"| {name} | {a['before_pass']:.1%} → {a['after_pass']:.1%} | {b['before_pass']:.1%} → {b['after_pass']:.1%} |")
text+=['','Same125 sampled timestamps as the previous Q29 replay; pass means IoU≥0.5 against independent Q28 SAM masks. These are alignment proxies, not manual ground truth. The new video contains all499 frames. Calibration and tracker settings unchanged; initial causal mask cache reused.',
    '', '[Both-camera video](hang_v2stand_both_cameras.mp4) · [Contact sheet](hang_v2stand_contact_sheet.jpg) · [Full metrics](hang_v2stand_alignment.json)']
(OUT/'hang_v2stand_alignment.md').write_text('\n'.join(text)+'\n')
print(json.dumps(matched,indent=2))
