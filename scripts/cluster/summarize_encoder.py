"""Report window-weighted means; this evaluates reconstructed extracted points."""
import argparse,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
r=Path(a.root);enc=r/'reports/encoder';data=json.loads((enc/'roundtrip_v2.json').read_text())
groups=['gripper','robot','static','moving','cloth','static_in_moving_tokens']
variants=['fb4096_analytic/default','fb4096_learned/default','fb4096_analytic/segments','fb4096_learned/segments']
rows={}
for variant in variants:
    rows[variant]={}
    for group in groups:
        vals=[d[variant][group] for d in data.values() if group in d.get(variant,{})]
        count=sum(v['windows'] for v in vals)
        if count:rows[variant][group]=dict(mean_mm=sum(v['mean_mm']*v['windows'] for v in vals)/count,windows=count)
def fmt(x):return '-' if x is None else f'{x:.2f}'
lines=['# Dataset v2 encoder round trip','','FB checkpoint zero-shot, 4096 points, 16 history + t0 +32 future frames, stride48. Analytic = Kabsch without residuals. Means are weighted by the number of windows containing each group; they are not pooled point errors.','',
'| Variant | Gripper | Robot | Static | Moving | Cloth | Blending |','|---|---:|---:|---:|---:|---:|---:|']
for variant in variants:lines.append('| '+variant+' | '+' | '.join(fmt(rows[variant].get(g,{}).get('mean_mm')) for g in groups)+' |')
old=json.loads((r/'references/roundtrip_v4.json').read_text())
lines+=['','## Published v1 comparison','','Same episode names, FB learned/default. Point extraction, geometry and validity cohorts changed; these differences are not a controlled model improvement.','',
'| Episode | Group | Published v1 mm | v2 mm |','|---|---|---:|---:|']
for key,d in old.items():
    if key not in data:continue
    for group in groups:
        before=d.get('fb4096_learned/default',{}).get(group)
        after=data[key].get('fb4096_learned/default',{}).get(group)
        if before and after:lines.append(f"| {key} | {group} | {before['mean_mm']:.2f} | {after['mean_mm']:.2f} |")
lines+=['','## Matched pre-fold cloth cohort','','Frames1..159 only; complete default-stride windows t0=64,112. This avoids comparing v1 pre-fold-only validity with the full v2 fold. Geometry and sample identity still differ.','',
'| Variant | v1 cloth mm | v2 cloth mm |','|---|---:|---:|']
matched={label:json.loads((enc/(label+'_cloth_matched.json')).read_text())['greycloth/episode_0000'] for label in ('v1','v2')}
for variant in variants:lines.append('| '+variant+' | '+' | '.join(fmt(matched[label].get(variant,{}).get('cloth',{}).get('mean_mm')) for label in ('v1','v2'))+' |')
lines+=['','Reconstruction error measures compression of the extracted points, not camera alignment, hidden material correctness or closed-loop world-model performance. All cloth remains approximate. Three videos: sweep_roundtrip.mp4, hang_roundtrip.mp4, greycloth_roundtrip.mp4.']
(enc/'REPORT.md').write_text('\n'.join(lines)+'\n')
(enc/'summary.json').write_text(json.dumps(dict(aggregate=rows,episodes=len(data),checkpoint_sha256=hashlib.sha256((r/'checkpoints/ae_wmstats.pt').read_bytes()).hexdigest(),units='mm'),indent=2))
