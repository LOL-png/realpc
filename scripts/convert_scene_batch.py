#!/usr/bin/env python
"""Run independent causal episodes with bounded local concurrency and resume."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes-root', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--config', default=str(ROOT / 'configs/scene_pipeline_v2.json'))
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    recordings, output = Path(args.episodes_root), Path(args.out)
    episodes = sorted(p for task in ('sweep', 'hang', 'greycloth')
                      for p in (recordings / task).glob('episode_*') if p.is_dir())
    logs = output / '_logs'
    logs.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault('OMP_NUM_THREADS', '2')
    env.setdefault('MKL_NUM_THREADS', '2')

    def convert(path):
        key = path.parent.name + '/' + path.name
        with (logs / (key.replace('/', '_') + '.log')).open('w') as log:
            subprocess.run([sys.executable, '-u', str(ROOT / 'scripts/convert_scene.py'),
                            '--episodes', str(path), '--out', str(output),
                            '--config', args.config, '--skip-existing', '--save-masks'],
                           cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        return key

    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(convert, p): p for p in episodes}
        for number, future in enumerate(as_completed(futures), 1):
            try:
                print(f'{number}/{len(episodes)} complete {future.result()}', flush=True)
            except Exception as error:
                failures.append(str(futures[future]))
                print(f'FAILED {futures[future]}: {error}', flush=True)
    if failures:
        raise RuntimeError(f'Failed episodes (completed ones retained): {failures}')
    # Individual workers may update the manifest concurrently. Rebuild once,
    # after all have finished, with complete loader/hash checks.
    subprocess.run([sys.executable, '-u', str(ROOT / 'scripts/verify_scene_dataset.py'),
                    '--dataset', str(output), '--episodes-root', str(recordings),
                    '--config', args.config], cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    main()
