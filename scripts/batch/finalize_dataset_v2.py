#!/usr/bin/env python
"""Validate completeness; preserve v1 manifest schema; aggregate QA and hash payload."""
import sys,json,hashlib,subprocess,shutil,datetime
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from realpc.v4 import SCHEMA
OUT=Path('/home/howardji/data_ext/realpc_out/v2');RUN=Path('/home/howardji/data_ext/astra_meshes/dataset_v2')
expected={f'sweep/episode_{i:04d}.npz' for i in range(50)}|{'hang/episode_0000.npz','greycloth/episode_0000.npz'}
review=json.load(open(RUN/'pilot_review.json'))
actual={str(p.relative_to(OUT)) for p in OUT.glob('*/episode_*.npz')};assert actual==expected,(expected-actual,actual-expected)
for f in expected:
 p=OUT/f;q=json.load(open(p.with_name(p.stem+'_qa.json')));assert q['loader_validation']['loaded']
 if not f.startswith('greycloth'):assert (p.with_name(p.stem+'_alignment.json')).exists()
 key=f[:-4]
 if key in review['episodes']:
  q['visual_review']=review['episodes'][key];json.dump(q,open(p.with_name(p.stem+'_qa.json'),'w'),indent=2)
subprocess.run([sys.executable,str(ROOT/'scripts/make_manifest.py'),'--out_dir',str(OUT),'--dataset-name','agentwm_v2','--assets',str(ROOT/'assets/objects_v3')],check=True)
allrows={};checks=[]
for p in sorted(OUT.glob('*/*_alignment.json')):
 d=json.load(open(p));checks.append({'episode':d['episode'],**{k:{'frames':v['frames'],'iou_median':v['median'],'pass_frac':v['pass']} for k,v in d['summary']['tracker'].items()}})
 for source,parts in d['per_frame'].items():
  for key,vals in parts.items():
   values=np.array(vals)[:,1];allrows.setdefault(source,{}).setdefault(key,[]).append((d['episode'],values))
agg={}
for source,parts in allrows.items():
 agg[source]={}
 for key,rows in parts.items():
  vals=np.concatenate([v for _,v in rows]);epass=np.array([(v>=.5).mean() for _,v in rows]);agg[source][key]={'episodes':len(rows),'sampled_frames':len(vals),'pooled_pass_frac':float((vals>=.5).mean()),'median_episode_pass_frac':float(np.median(epass)),'median_iou':float(np.median(vals)),'episode_pass_frac':{e:float((v>=.5).mean()) for e,v in rows}}
summary={'threshold':.5,'sampling':'every16 ticks plus displayed timestamps','sources':{'legacy':'Original v1 causal masks, fixed reference, not manual GT','tracker':'New v2 causal masks used by tracker, self-consistency','independent':'Q28 SAM masks, only episode0 sweep/hang; not manual GT'},'objects':agg}
json.dump(summary,open(OUT/'alignment_summary.json','w'),indent=2)
man=json.load(open(OUT/'manifest.json'));man['alignment_checks']=checks;man['aggregate']['alignment']=agg;man['object_masks']='SAM2.1 hiera-small, forward-only, objects_v3 mesh/FK prompts refined in current RGB; image-refine-seeds; no evaluation masks used for conversion.'
man['tracker_flags']={'REALPC_SIL':1,'REALPC_CAM_EQUAL':1,'REALPC_SIL_METHOD':'contour','REALPC_COLD_COMPONENTS':1};man['tracker_base_commit']='aa7ef8a';man['qa_notes']=summary['sources'];man['cloth']='Same v1 tracks and calibration transform, exact points/validity equality verified; valid only before tick160.'
prov=OUT/'provenance';prov.mkdir(exist_ok=True);shutil.copy2(ROOT/'configs/calib_yam_rig_v3.json',prov/'calib_yam_rig_v3.json');shutil.copytree(ROOT/'assets/objects_v3',prov/'objects_v3',dirs_exist_ok=True);shutil.copy2(RUN/'pilot_review.json',prov/'pilot_review.json')
man['calibration']['sha256']=hashlib.sha256((prov/'calib_yam_rig_v3.json').read_bytes()).hexdigest();assert man['calibration']['sha256']=='04972b6b0176d7a5d818ef83bf88804e7f924364e81f6ebfd3c7444ed1282c2b'
schema=dict(SCHEMA,obj_support='[P,T,2] int32 current masked depth support top/side',obj_mode='[P,T] string observation cameras or fk/carrier/hold');man['schema']=schema
json.dump(man,open(OUT/'manifest.json','w'),indent=2);json.dump(schema,open(OUT/'schema.json','w'),indent=2)
lines=[]
for p in sorted(OUT.rglob('*')):
 if p.is_file() and p.name!='SHA256SUMS':lines.append(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(OUT)))
(OUT/'SHA256SUMS').write_text('\n'.join(lines)+'\n')
subprocess.run(['sha256sum','--check','--quiet','SHA256SUMS'],cwd=OUT,check=True)
print(json.dumps({'episodes':len(expected),'files_hashed':len(lines),'alignment':{s:{k:round(v['pooled_pass_frac'],3) for k,v in a.items()} for s,a in agg.items()}},indent=2))
