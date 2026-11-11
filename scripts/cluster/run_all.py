#!/usr/bin/env python
"""One sequential GPU allocation: validate, convert 52 episodes, QA, encode.

Completed stages have fingerprinted markers. Conversion additionally checks its
own code/config hashes. A requeued job resumes completed episodes; no subprocess
submits another job, and no cloud polling or transfer runs here.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time

REPO = Path(__file__).resolve().parents[2]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-root', required=True)
    p.add_argument('--dataset', required=True)
    a = p.parse_args()
    root, dataset = Path(a.run_root), Path(a.dataset)
    reports, markers, logs = root/'reports', root/'markers', root/'logs'
    for path in (reports, markers, logs, dataset):
        path.mkdir(parents=True, exist_ok=True)
    recordings = root/'inputs/hf'
    config = REPO/'configs/scene_pipeline_v2.json'
    ckpt = root/'checkpoints/ae_wmstats.pt'
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    fingerprint = hashlib.sha256((commit+sha(config)+sha(ckpt)).encode()).hexdigest()
    os.environ.update(REALPC_EPISODES_ROOT=str(recordings),
                      REALPC_MASKS_ROOT=str(root/'references/masks'),
                      REALPC_MODEL_CACHE=str(root/'models'),
                      PLWM_SE3_SRC=str(root/'plwm_tag'),
                      REALPC_TEST_EPISODE=str(recordings/'sweep/episode_0000'),
                      PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    start = time.time()
    state = dict(job_id=os.environ.get('SLURM_JOB_ID'), node=os.environ.get('SLURMD_NODENAME'),
                 code_commit=commit, dataset=str(dataset), fingerprint=fingerprint,
                 started=datetime.now(timezone.utc).isoformat(), state='running')

    def space_guard():
        used = sum(int(subprocess.check_output(['du','-s','-B1',str(p)],text=True).split()[0])
                   for p in (root,dataset))
        free = shutil.disk_usage(root).free
        atomic_json(root/'space.json', dict(used_bytes=used, free_bytes=free, limit_bytes=50*10**9))
        if used > 47*10**9 or free < 8*10**9:
            state.update(state='space_pause', used_bytes=used, free_bytes=free)
            atomic_json(root/'status.json',state)
            raise RuntimeError('Space guard: archive verified bulky reports before resuming; nothing deleted')

    def run(name, args, outputs=(), executable=None):
        marker = markers/(name+'.json')
        if marker.exists() and all(Path(f).exists() for f in outputs):
            old = json.loads(marker.read_text())
            if old.get('fingerprint') == fingerprint:
                print('REUSE',name,flush=True)
                return
        space_guard()
        state.update(stage=name, updated=datetime.now(timezone.utc).isoformat(),
                     elapsed_seconds=time.time()-start)
        atomic_json(root/'status.json',state)
        command = [executable or sys.executable, *map(str,args)]
        print('START',name,flush=True)
        began = time.time()
        with (logs/(name+'.log')).open('a') as log:
            log.write('\nSTART '+datetime.now(timezone.utc).isoformat()+'\n');log.flush()
            result = subprocess.run(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            state.update(state='failed',returncode=result.returncode,log=str(logs/(name+'.log')))
            atomic_json(root/'status.json',state)
            raise RuntimeError(f'{name} failed ({result.returncode}); see {logs/name}.log')
        atomic_json(marker,dict(fingerprint=fingerprint,seconds=time.time()-began,
                               completed=datetime.now(timezone.utc).isoformat(),outputs=list(map(str,outputs)),
                               output_sha256={str(f):sha(f) for f in outputs if Path(f).is_file()}))
        print('DONE',name,round(time.time()-began,1),'seconds',flush=True)

    run('environment',['scripts/cluster/check_environment.py','--out',reports/'environment.json'],
        [reports/'environment.json'])
    tests=sorted(str(f.relative_to(REPO)) for f in (REPO/'tests').glob('test_*.py'))
    run('tests',['-m','pytest',*tests,'-q'])
    validation=[f'sweep/episode_{n:04d}' for n in (0,10,20,30,40)]+['hang/episode_0000','greycloth/episode_0000']

    def convert(key):
        path=dataset/(key+'.npz')
        run('convert_'+key.replace('/','_'),['scripts/convert_scene.py','--episodes',recordings/key,
            '--out',dataset,'--config',config,'--save-masks','--skip-existing'],
            [path,path.with_name(path.stem+'_qa.json')])
        # Video publication follows each conversion; relay only transfers files.
        dest = reports/'episodes'/key
        run('qa_'+key.replace('/','_'),['scripts/qa_scene_dataset.py','--dataset',dataset,
            '--out',reports/'episodes','--episodes',key,'--every','50','--all-videos',
            '--cloth-reference-inputs',root/'references/cloth_inputs'],[dest/'both_cameras.mp4'])

    for key in validation:
        convert(key)
    for task in ('sweep','hang','greycloth'):
        out=reports/f'parity_{task}.json'
        run('parity_'+task,['tests/check_scene_live_parity.py','--converted',dataset/task/'episode_0000.npz',
                           '--frames','32','--out',out],[out])
    run('validation_qa',['scripts/qa_scene_dataset.py','--dataset',dataset,'--out',reports/'validation',
        '--episodes',*validation,'--every','50','--example-videos',
        '--cloth-reference-inputs',root/'references/cloth_inputs'],
        [reports/'validation/alignment_summary.json'])
    # A hard physical failure blocks bulk conversion. Approximate visible fit is
    # reported separately; silhouette overlap is not material correspondence GT.
    qa=json.loads((reports/'validation/alignment_summary.json').read_text())
    for key, values in qa['cloth_physics'].items():
        if values['endpoint_crossing_frames'] or max(values['max_extension'],values['max_compression'])>.021:
            state.update(state='failed',stage='cloth_physical_gate',detail=values)
            atomic_json(root/'status.json',state)
            raise RuntimeError(f'Cloth physical audit failed: {key}')
    atomic_json(reports/'validation_gate.json',dict(passed=True,physical_gate='zero endpoint crossings; strain <=2.1%',
        note='Visible fit and hidden cloth geometry remain approximate; inspect validation videos.',code_commit=commit))
    episodes=sorted(str(e.relative_to(recordings)) for task in ('sweep','hang','greycloth')
                    for e in (recordings/task).glob('episode_*') if e.is_dir())
    assert len(episodes)==52, f'Expected52 episodes, got {len(episodes)}'
    for key in episodes:
        convert(key)
    run('verify_files',['scripts/verify_scene_dataset.py','--dataset',dataset,'--episodes-root',recordings,
                       '--config',config],[dataset/'manifest.json'])
    run('all_qa',['scripts/qa_scene_dataset.py','--dataset',dataset,'--out',dataset/'qa','--every','50',
        '--example-videos','--cloth-reference-inputs',root/'references/cloth_inputs'],
        [dataset/'qa/alignment_summary.json'])
    run('verify_with_qa',['scripts/verify_scene_dataset.py','--dataset',dataset,'--episodes-root',recordings,
                        '--config',config],[dataset/'SHA256SUMS'])
    enc=reports/'encoder';(enc/'episodes').mkdir(parents=True,exist_ok=True)
    combined={}
    for key in episodes:
        out=enc/'episodes'/(key.replace('/','_')+'.json')
        run('encoder_'+key.replace('/','_'),['scripts/roundtrip_v4.py','--converted',dataset/(key+'.npz'),
            '--fb_ckpt',ckpt,'--fb-only','--out',out],[out])
        combined.update(json.loads(out.read_text()))
    atomic_json(enc/'roundtrip_v2.json',combined)
    v1=root.parent/'agentwm_v1'
    for label,base in [('v1',v1),('v2',dataset)]:
        out=enc/(label+'_cloth_matched.json')
        run('matched_'+label,['scripts/roundtrip_v4.py','--converted',base/'greycloth/episode_0000.npz',
            '--fb_ckpt',ckpt,'--fb-only','--min-frame','1','--max-frame','159','--out',out],[out])
    for task,t0 in [('sweep',240),('hang',240),('greycloth',192)]:
        out=enc/(task+'_roundtrip.mp4')
        run('video_'+task,['scripts/roundtrip_video.py','--converted',dataset/task/'episode_0000.npz',
            '--t0',str(t0),'--ckpt',ckpt,'--out',out,'--title',task+' dataset_v2'],[out])
    run('encoder_summary',['scripts/cluster/summarize_encoder.py','--root',root], [enc/'REPORT.md'])
    state.update(state='complete',stage='complete',episodes=52,elapsed_seconds=time.time()-start,
                 finished=datetime.now(timezone.utc).isoformat())
    atomic_json(root/'status.json',state)
    print('COMPLETE:52 episodes, QA, manifests, encoder audit and three roundtrip videos',flush=True)


if __name__=='__main__':
    main()
