"""Matched phase summaries of camera/strain/held-track diagnostics (no fitting)."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--fit',action='append',required=True,help='label=fit_directory');p.add_argument('--camera-audit',required=True);p.add_argument('--out',required=True);a=p.parse_args()
camera=json.loads(Path(a.camera_audit).read_text());result={}
def median(rows,key):
    x=[r.get(key) for r in rows if r.get(key) is not None]
    return float(np.median(x)) if x else None
for spec in a.fit:
    label,d=spec.split('=',1);d=Path(d);m=json.loads((d/'metrics.json').read_text());q=json.loads((d/'quality_report.json').read_text());physical=q['physical_audit']
    r=dict(frames=len(m),max_extension_percent=100*max(v['max_extension'] for v in physical),max_compression_percent=100*max(v['max_compression'] for v in physical),crossing_frames=sum(v['intersects'] for v in physical),physical_frames_audited=len(physical),phases={})
    for phase,lo,hi in [('before_lift',0,160),('first_fold',160,250),('between_folds',250,335),('second_fold',335,len(m))]:
        rows=m[lo:hi]
        if not rows:continue
        rr=dict(camera=camera['summary'][label][phase],heldout_cotracker={},heldout_lk={})
        for c in ('top','side'):
            cot=[v.get('cotracker',{}).get(c,{}) for v in rows];lk=[v.get('heldout_tracks',{}).get(c,{}) for v in rows]
            rr['heldout_cotracker'][c]={key:median(cot,key) for key in ['held_median_px','held_p90_px','held_n','age','queries']}
            rr['heldout_lk'][c]={key:median(lk,key) for key in ['median_px','long_median_px','n','long_n','median_age']}
        active=[v for v in rows if v.get('pins',0)>0]
        rr['hard_grasp_max_mm']=max((v['hard_grasp_error_mm'] for v in active if 'hard_grasp_error_mm' in v),default=None)
        rr['hard_grasp_median_mm']=median(active,'hard_grasp_error_mm')
        rr['active_grasp_frames']=len(active);r['phases'][phase]=rr
    result[label]=r
Path(a.out).write_text(json.dumps(dict(models=result,note='Depth: phase median of per-frame medians on silhouette overlap, with IoU reported. CoTracker: median of per-frame held-out medians; visibility/survival-conditioned, with counts and reseed age. Hard-grasp errors are constraint residuals, not independent material ground truth. Crossing audit is at exported frames, not a proof of physical validity.'),indent=2))
for name,r in result.items():
 f=r['phases']['first_fold'];print(name,'frames',r['frames'],'depth',[round(f['camera'][c]['depth_med_mm'],2) for c in ('top','side')],'worst extension',round(r['max_extension_percent'],2),'tracks',[f['heldout_cotracker'][c]['held_median_px'] for c in ('top','side')],'crossings',r['crossing_frames'])
