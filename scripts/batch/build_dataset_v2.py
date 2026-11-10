#!/usr/bin/env python
"""Resumable local v2 conversion. Pilot must be visually approved before --phase full.
Every subprocess uses the frozen final tracker; never transfers or launches remote work.
"""
import os,sys,json,time,argparse,subprocess,shutil,gc
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from realpc.dataset import load
P=argparse.ArgumentParser();P.add_argument('--phase',choices=['pilot','full'],required=True);P.add_argument('--workers',type=int,default=2);a=P.parse_args()
DATA=Path('/home/howardji/data_ext/agentwm_probe/hf');OUT=Path('/home/howardji/data_ext/realpc_out/v2');RUN=Path('/home/howardji/data_ext/astra_meshes/dataset_v2');SEG=RUN/'seg';PILOT=[0,10,20,30,45]
ENV=dict(os.environ,REALPC_SIL='1',REALPC_CAM_EQUAL='1',REALPC_SIL_METHOD='contour',REALPC_COLD_COMPONENTS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libstdc++.so.6')
for d in [OUT,RUN/'logs',SEG]:d.mkdir(parents=True,exist_ok=True)
if a.phase=='full' and not (RUN/'pilot_review.json').exists():raise SystemExit('Missing human/agent visual pilot review artifact')

def run(cmd,log):
 with log.open('a') as f:
  f.write('\nCOMMAND '+repr(cmd)+'\n');f.flush()
  subprocess.run([sys.executable]+cmd,cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT,check=True)

def episode(ep):
 task=ep.parent.name;name=ep.name;key=task+'/'+name;dest=OUT/task/(name+'.npz');log=RUN/'logs'/(task+'_'+name+'.log');done=RUN/'logs'/(task+'_'+name+'.done.json');t=time.time()
 if done.exists() and dest.exists():return {'episode':key,'cached':True}
 print('START',key,flush=True)
 if task!='greycloth':
  if not all((SEG/task/(name+'_'+c+'.npz')).exists() for c in ['top','side']):
   run(['scripts/segment_objects.py','--episode',str(ep),'--out_dir',str(SEG),'--assets','assets/objects_v3','--image-refine-seeds','--sam_ckpt','/home/howardji/realpc/.model_cache/sam2/sam2.1_hiera_small.pt'],log)
 if not dest.exists():
  run(['scripts/convert_v4.py','--episodes',str(ep),'--out_dir',str(OUT),'--assets','assets/objects_v3','--seg_dir',str(SEG)],log)
 qa=json.load(open(dest.with_name(name+'_qa.json')))
 # Real loader expansion, sequential within each worker. Episode arrays are released afterward.
 d=load(str(dest));assert d['points'].shape==(*d['valid'].shape,3);assert np.isfinite(d['points'][d['valid']]).all();assert np.array_equal(d['ticks'],np.arange(len(np.load(ep/'low_dim.npz')['t_cap'])))
 if task=='greycloth':
  old=np.load('/home/howardji/data_ext/realpc_out/v1/greycloth/episode_0000.npz');assert np.array_equal(d['cloth_valid'],old['cloth_valid']);assert np.array_equal(d['cloth_points'],old['cloth_points'])
 qa['loader_validation']={'loaded':True,'points_shape':list(d['points'].shape),'finite_valid_points':True,'cloth_unchanged_from_v1':task=='greycloth'};del d;gc.collect()
 if task!='greycloth':
  audit=RUN/'qa'/task/name
  if not (audit/'metrics.json').exists():
   n=qa['ticks'];visual=np.linspace(0,n-1,6).astype(int).tolist() if int(name[-4:]) in PILOT and task=='sweep' else [0,n//2,n-1]
   cmd=['scripts/audit_v3.py','--converted',str(dest),'--assets','assets/objects_v3','--every','16','--out',str(audit),'--reference-name','legacy','--tracker-seg-dir',str(SEG),'--visual']+list(map(str,visual))+['--scene-occlusion']
   run(cmd,log)
  al=json.load(open(audit/'metrics.json'));qa['alignment']=al['summary'];qa['alignment_every']=16
  shutil.copy2(audit/'metrics.json',dest.with_name(name+'_alignment.json'))
  summary={k:{'frames':v['frames'],'iou_median':v['median'],'pass_frac':v['pass']} for k,v in al['summary']['tracker'].items()}
  json.dump({'episode':key,'reference':'v2 causal SAM tracker masks (self-consistency)','summary':summary},open(dest.with_name(name+'_selfqa.json'),'w'),indent=2)
 else:qa['alignment']={'status':'not_applicable','reason':'No rigid library objects; cloth unchanged from v1, valid before tick160.'}
 json.dump(qa,open(dest.with_name(name+'_qa.json'),'w'),indent=2)
 result={'episode':key,'seconds':round(time.time()-t,1),'loader_pass':True};json.dump(result,open(done,'w'),indent=2);print('DONE',json.dumps(result),flush=True);return result

eps=[DATA/'sweep'/f'episode_{i:04d}' for i in PILOT] if a.phase=='pilot' else [p for task in ['sweep','hang','greycloth'] for p in sorted((DATA/task).glob('episode_*'))]
fail=[]
with ThreadPoolExecutor(max_workers=a.workers) as pool:
 fut={pool.submit(episode,ep):ep for ep in eps}
 for f in as_completed(fut):
  try:f.result()
  except Exception as e:fail.append({'episode':str(fut[f]),'error':repr(e)});print('FAILED',fail[-1],flush=True)
json.dump({'phase':a.phase,'failures':fail,'episodes':len(eps)},open(RUN/(a.phase+'_status.json'),'w'),indent=2)
if fail:raise SystemExit(1)
