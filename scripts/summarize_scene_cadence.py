"""Common-reference alignment and uncontended speed, including partial results."""
import argparse,csv,json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True)
a=p.parse_args();root=Path(a.root);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
objects=['mug','stand','broom','dustpan','debris','box','cloth'];results={};csvrows=[];series={}
for n in [1,2,3,10]:
    folder=root/f'n{n}';entry=dict(tasks={},objects={},complete=False);series[n]={}
    for task in ['hang','sweep','greycloth']:
        qa=folder/'data'/task/'episode_0000_qa.json'
        if not qa.exists():continue
        q=json.loads(qa.read_text());seg=q['pipeline']['segmentation']
        assert q['pipeline']['causal'] and q['pipeline']['shared_live_conversion_step']
        assert seg['one_shared_stream'] and seg['causal']
        if n in (1,2,3):assert seg['sam_frames']==(q['ticks']+n-1)//n
        entry['tasks'][task]=dict(frames=q['ticks'],seconds=q['seconds'],end_to_end_fps=q['ticks']/q['seconds'],
            step_fps=q['pipeline']['fps'],sam_frames=seg['sam_frames'],sam_fraction=seg['sam_frame_rate'],
            config_sha256=q['config_sha256'],implementation_sha256=q['implementation_sha256'],load_verified=q['load_verified'])
        audit=folder/'audit'/task/'episode_0000'/('camera_audit.json' if task=='greycloth' else 'metrics.json')
        if not audit.exists():continue
        d=json.loads(audit.read_text())
        if task=='greycloth':
            rows=d['per_frame'][f'n{n}'];item=dict(reference='fixed_every_frame_cloth_proxy',cameras={},phases=d['summary'][f'n{n}'])
            series[n]['cloth']={}
            for c in ['top','side']:
                values=[r[c]['iou'] for r in rows]
                item['cameras'][c]=dict(frames=len(values),mean_iou=float(np.mean(values)),median_iou=float(np.median(values)),
                    pass_iou_05=float(np.mean(np.array(values)>=.5)),mean_mask_coverage=float(np.mean([r[c]['mask_recall'] for r in rows])),
                    depth_median_mm=float(np.median([r[c]['depth_med_mm'] for r in rows if r[c]['depth_med_mm'] is not None])))
                series[n]['cloth'].update({(c,r['t']):r[c]['iou'] for r in rows})
            entry['objects']['cloth']=item
        else:
            for name in (['mug','stand'] if task=='hang' else ['broom','dustpan','debris','box']):
                mode='fixed_n10_reference' if name=='box' else 'independent'
                item=dict(reference='fixed_n10_consistency_only' if name=='box' else 'Q28_evaluation_only',cameras={})
                series[n][name]={}
                for c in ['top','side']:
                    raw=d.get('support_per_frame',{}).get(mode,{}).get(name+'/'+c,[])
                    rows=[r for r in raw if r['gt_pixels']>=60]
                    vals=np.array([r['iou'] for r in rows])
                    depths=[r for r in d.get('depth_per_frame',{}).get(name+'/'+c,[]) if r['median_mm'] is not None]
                    item['cameras'][c]=dict(frames=len(vals),mean_iou=float(vals.mean()) if len(vals) else None,
                        median_iou=float(np.median(vals)) if len(vals) else None,pass_iou_05=float(np.mean(vals>=.5)) if len(vals) else None,
                        small_reference_false_positive_frames=sum(r['gt_pixels']<60 and r['pred_pixels']>=60 for r in raw),
                        depth_frames=len(depths),depth_median_mm=float(np.median([r['median_mm'] for r in depths])) if depths else None,
                        depth_pass_fraction=float(np.mean([r['pass_depth'] for r in depths])) if depths else None)
                    series[n][name].update({(c,r['t']):r['iou'] for r in rows})
                entry['objects'][name]=item
    if len(entry['objects'])==7:
        # Equal objects, then equal camera means; no speed term and no varying
        # run's segmentation mask is used as its own ground truth.
        entry['macro_common_iou']=float(np.mean([np.mean([v['mean_iou'] for v in entry['objects'][name]['cameras'].values() if v['mean_iou'] is not None]) for name in objects if name!='box']))
        entry['complete']=True
    for name,item in entry['objects'].items():
        task='hang' if name in ['mug','stand'] else 'greycloth' if name=='cloth' else 'sweep'
        for cam,stats in item['cameras'].items():
            csvrows.append(dict(interval=n,object=name,camera=cam,reference=item['reference'],**stats,
                end_to_end_fps=entry['tasks'][task]['end_to_end_fps'],step_fps=entry['tasks'][task]['step_fps'],sam_fraction=entry['tasks'][task]['sam_fraction']))
    results[n]=entry
complete=[n for n in results if results[n]['complete']]
if len(complete)==4:
    for task in ['hang','sweep','greycloth']:
        assert len({results[n]['tasks'][task]['implementation_sha256'] for n in complete})==1, 'Implementation changed: '+task
    configurations=[]
    for n in complete:
        cfg=json.loads((root/f'n{n}'/'config.json').read_text())
        assert cfg['segmentation'].pop('interval')==n
        configurations.append(cfg)
    assert all(c==configurations[0] for c in configurations), 'Changes beyond SAM cadence'
    for name in objects:
        cohorts=[set(series[n][name]) for n in complete]
        assert all(x==cohorts[0] for x in cohorts),'Different reference-visible cohorts: '+name
ranking=sorted(complete,key=lambda n:results[n]['macro_common_iou'],reverse=True)
report=dict(rates=results,accuracy_ranking=ranking,
    note='Ranking excludes box consistency (no independent box mask) and requires separate depth/visual review. Cloth masks are a fixed proxy. One full validation episode per task; not a held-out generalization claim. More frequent SAM does not validate hidden cloth or physical deformation.')
(out/'summary.json').write_text(json.dumps(report,indent=2))
if csvrows:
    fields=list(dict.fromkeys(k for row in csvrows for k in row))
    with (out/'per_object_alignment.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(csvrows)
def fmt(x):return 'pending' if x is None else f'{x:.3f}'
lines=['# Unified SAM accuracy comparison','',report['note'],'',
       'Primary IoU uses reference-visible pixels (>=60 at audit resolution) with the SAME cohort at every cadence. Rigid reference: Q28; cloth: fixed current-frame masks/depth. Box scores are consistency only and excluded from the macro ranking.','',
       '| Object (mean IoU top / side) | Every1 | Every2 | Every3 | Every10 |','|---|---|---|---|---|']
for name in objects:
    cells=[]
    for n in [1,2,3,10]:
        item=results[n]['objects'].get(name)
        cells.append(' / '.join(fmt(item['cameras'][c]['mean_iou']) for c in ['top','side']) if item else 'pending')
    lines.append('| '+name+' | '+' | '.join(cells)+' |')
lines+=['','| Object (depth median mm top / side) | Every1 | Every2 | Every3 | Every10 |','|---|---|---|---|---|']
for name in objects:
    cells=[]
    for n in [1,2,3,10]:
        item=results[n]['objects'].get(name)
        cells.append(' / '.join(fmt(item['cameras'][c]['depth_median_mm']) for c in ['top','side']) if item else 'pending')
    lines.append('| '+name+' | '+' | '.join(cells)+' |')
lines+=['','| End-to-end fps | Every1 | Every2 | Every3 | Every10 |','|---|---|---|---|---|']
for task in ['hang','sweep','greycloth']:
    lines.append('| '+task+' | '+' | '.join(fmt(results[n]['tasks'].get(task,{}).get('end_to_end_fps')) for n in [1,2,3,10])+' |')
lines+=['','Cloth fold phase medians (top / side; fixed reference, hidden layers unmeasured):','',
        '| Interval | First-fold IoU | Second-fold IoU | First-fold depth mm | Second-fold depth mm |',
        '|---|---|---|---|---|']
for n in [1,2,3,10]:
    item=results[n]['objects'].get('cloth');cells=[]
    for key,phase in [('iou','first_fold'),('iou','second_fold'),('depth_med_mm','first_fold'),('depth_med_mm','second_fold')]:
        cells.append(' / '.join(fmt(item['phases'][phase][c][key]) for c in ['top','side']) if item else 'pending')
    lines.append('| '+str(n)+' | '+' | '.join(cells)+' |')
lines+=['','Actual SAM fractions, per-view IoU>=0.5 pass rates, depth, and step-only fps are in [CSV](per_object_alignment.csv) and [JSON](summary.json). Robot/table geometry does not depend on SAM cadence.',
        'A SAM refresh also invokes rigid silhouette refinement in the existing shared pipeline; this comparison measures the complete refresh policy.',
        'N10 cloth reuses the completed full control after exact source/config/asset hashes and output checksum verification; the other measured conversions are fresh.',
        '', 'Accuracy ranking (completed rates): '+str(ranking)+'. Selection is pending visual/depth review.']
(out/'README.md').write_text('\n'.join(lines)+'\n')
print('Completed accuracy ranking',[(n,results[n]['macro_common_iou']) for n in ranking])
print('Ready objects',{n:list(r['objects']) for n,r in results.items()})
