"""Export the common camera audit as a per-frame CSV and an honest overview."""
import argparse, csv, json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser()
p.add_argument('--audit',required=True)
p.add_argument('--out',required=True)
a=p.parse_args()
audit=json.loads(Path(a.audit).read_text());out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
keys=['iou','mask_recall','depth_med_mm','depth_p90_mm']
with (out/'per_frame_alignment.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=['model','frame','camera']+keys)
    writer.writeheader()
    for label,rows in audit['per_frame'].items():
        for row in rows:
            for camera in ('top','side'):
                writer.writerow(dict(model=label,frame=row['t'],camera=camera,**row[camera]))
fig,axes=plt.subplots(3,2,figsize=(14,9),sharex=True)
for j,camera in enumerate(('top','side')):
    for i,(key,title) in enumerate([('iou','Mask IoU'),('mask_recall','Visible mask coverage'),('depth_med_mm','Overlap depth median (mm)')]):
        ax=axes[i,j]
        for label,rows in audit['per_frame'].items():
            ax.plot([r['t'] for r in rows],[r[camera][key] for r in rows],label=label,lw=1)
        for lo,hi in [(160,250),(335,audit['frames'])]:ax.axvspan(lo,hi,color='gray',alpha=.1)
        ax.set_title(camera+' — '+title);ax.grid(alpha=.2)
        if i<2:ax.set_ylim(0,1)
        if i==2:ax.set_xlabel('frame')
axes[0,0].legend()
fig.suptitle('Same masks and depth for every model; shaded: fold intervals. Hidden layers are not measured.')
fig.tight_layout();fig.savefig(out/'per_frame_alignment.png',dpi=150)
print(out/'per_frame_alignment.csv')
