"""Whole-episode summary retaining spurious visible predictions in each audit."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--depth-root',type=Path,required=True);p.add_argument('--sam-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
result={'method':'IoU >= 0.5 on assessable views. Every timestamp is tested; a view is scored when reference OR rendered foreground has at least 60 pixels. Counts can differ because a spurious visible prediction is scored as a failure even if the other tracker correctly predicts an occluded object. No unmatched predictions are dropped. Robot/other-object occlusion is excluded. Independent Q28 masks except box, which uses the same fresh SAM reference for both runs. These are mask agreements, not ground-truth 6-DoF pose errors.','episodes':{}}
timing_rows=[];alignment_rows=[]
for task in ['sweep','hang']:
    roots={'SAM':a.sam_root,'depth':a.depth_root}
    qa={k:json.loads((v/task/'episode_0000_qa.json').read_text()) for k,v in roots.items()}
    audits={k:json.loads((v/('audit_'+task)/'metrics.json').read_text()) for k,v in roots.items()}
    states={k:json.loads((v/task/'episode_0000_run.json').read_text()) for k,v in roots.items()}
    q=qa['depth'];events=q['depth_pipeline']['sam_events'];speedup=qa['SAM']['seconds']/q['seconds']
    timing_rows.append(f"| {task} | {q['ticks']} | {qa['SAM']['timing']['end_to_end_fps']:.2f} | {q['timing']['end_to_end_fps']:.2f} | {speedup:.2f}× | {len(events)}/{q['ticks']} ({100*len(events)/q['ticks']:.1f}%) |")
    episode={'timing':qa,'source':states,'speedup':speedup,'objects':{}}
    for obj in q['objects']:
        reference='tracker' if obj=='box' else 'independent';row={}
        for cam in ['top','side']:
            key=obj+'/'+cam
            data={k:dict(v['per_frame'][reference].get(key,[])) for k,v in audits.items()}
            ticks=sorted(set(data['SAM'])&set(data['depth']))
            stat={'common_frames':len(ticks),'reference':reference}
            for k in data:
                x=np.array(list(data[k].values()));stat[k]={'frames':len(x),'pass_percent':float(100*np.mean(x>=.5)),'median_iou':float(np.median(x)),'ticks':sorted(data[k])}
            row[cam]=stat
        episode['objects'][obj]=row
        fmt=lambda mode,field,digits: '/'.join(f"{row[c][mode][field]:.{digits}f}" for c in ['top','side'])
        counts=lambda mode: '/'.join(str(row[c][mode]['frames']) for c in ['top','side'])
        alignment_rows.append(f"| {task}/{obj}{'*' if obj=='box' else ''} | {counts('SAM')} → {counts('depth')} | {fmt('SAM','pass_percent',1)} | {fmt('depth','pass_percent',1)} | {fmt('SAM','median_iou',3)} → {fmt('depth','median_iou',3)} |")
    result['episodes'][task]=episode
(a.out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
table='\n'.join(['| Episode | Frames | SAM fps | Depth fps | Speedup | SAM event frames in depth mode |','|---|---:|---:|---:|---:|---:|']+timing_rows)
table+='\n\n'+'\n'.join(['| Object | Scored frames SAM → depth (top/side) | SAM pass % top/side | Depth pass % top/side | Median IoU SAM → depth (top/side) |','|---|---:|---:|---:|---|']+alignment_rows)
table+='\n\n'+result['method']+' Box (*) has no independent Q28 mask. Timing uses serialized runs on an RTX 4070 Ti SUPER with OMP/MKL=2; includes initialization, input and compact conversion, excludes process startup and optional QA-mask export. Videos use recording time, not extraction speed.\n'
(a.out/'TABLES.md').write_text(table);print(table)
