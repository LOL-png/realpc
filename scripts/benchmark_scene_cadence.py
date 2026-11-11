"""Serial, full-episode SAM cadence ablation using the shared live step.

Does not write the dataset_v2 tree or restart collection conversion.
"""
import argparse,copy,hashlib,json,os,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from realpc.scene_pipeline import load_scene_config
p=argparse.ArgumentParser()
p.add_argument('--out',required=True)
p.add_argument('--rates',type=int,nargs='+',default=[10,1,2,3])
p.add_argument('--episodes',nargs='+',default=['hang/episode_0000','sweep/episode_0000','greycloth/episode_0000'])
a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
base=load_scene_config(ROOT/'configs/scene_pipeline_v2.json')
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libstdc++.so.6')
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
plan=dict(rates=a.rates,episodes=a.episodes,base_config=base,commit=commit,
          selection='Accuracy first: equal-object mean common-reference IoU; per-view pass rates and depth checked for regressions. Box has no Q28 mask, so raw-depth agreement is reported separately. Speed has no selection reward.',
          notes=['Only interval changes; failure-recovery thresholds remain fixed.',
                 'Rigid Q28 masks are fixed evaluation-only references.',
                 'Cloth uses identical cached current-frame masks/depth for every rate; this is a proxy, not hidden-material ground truth.',
                 'Full causal episodes from t0, serialized for uncontended fps. Cloth tracker remains approximate and unchanged.',
                 'Separate benchmark output; cancelled bulk dataset/encoder work stays cancelled.'])
(out/'plan.json').write_text(json.dumps(plan,indent=2))

def run(args,log,stage):
    (out/'status.json').write_text(json.dumps(dict(stage=stage,log=str(log),started=time.time()),indent=2))
    print(stage,flush=True)
    with log.open('w') as f:
        subprocess.run([sys.executable,'-u',*args],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)

for rate in a.rates:
    root=out/f'n{rate}';root.mkdir(exist_ok=True)
    config=copy.deepcopy(base);config['segmentation']['interval']=rate
    cfg=root/'config.json';cfg.write_text(json.dumps(config,indent=2))
    expected=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    for episode in a.episodes:
        task,name=episode.split('/');dest=root/'data';qa=dest/task/(name+'_qa.json')
        if qa.exists():
            assert json.loads(qa.read_text())['config_sha256']==expected
        run(['scripts/convert_scene.py','--episodes','/home/howardji/data_ext/agentwm_probe/hf/'+episode,
             '--out',str(dest),'--config',str(cfg),'--save-masks','--skip-existing'],
            root/(task+'_'+name+'_convert.log'),f'N{rate} convert {episode}')
        audit=root/'audit'/task/name;audit.mkdir(parents=True,exist_ok=True)
        if task=='greycloth':
            result=audit/'camera_audit.json'
            args=['scripts/cloth_multiview_audit.py','--fit',f'n{rate}='+str(dest/task/(name+'_cloth_mesh.npz')),
                  '--inputs','/home/howardji/data_ext/astra_cloth/auto_inputs','--out',str(result)]
        else:
            result=audit/'metrics.json'
            # Use a fixed existing label source for extra consistency scores;
            # the primary independent Q28 reference is loaded by the auditor.
            args=['scripts/audit_v3.py','--converted',str(dest/task/(name+'.npz')),
                  '--scene-seg-dir','/home/howardji/data_ext/realpc_out/v2',
                  '--reference-name','fixed_n10_reference','--assets',str(ROOT/'assets/objects_v3'),
                  '--depth-check','--every','1','--out',str(audit),'--visual','0','175','350','450']
        if not result.exists():run(args,root/(task+'_'+name+'_audit.log'),f'N{rate} audit {episode}')
(out/'status.json').write_text(json.dumps(dict(stage='complete',finished=time.time()),indent=2))
print('ALL CADENCE RUNS COMPLETE',flush=True)
