#!/usr/bin/env python
"""Verify a complete shared-pipeline dataset and write its manifest/checksums."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from realpc.dataset import load
from realpc.scene_pipeline import load_scene_config


def verify_cloth_quality(data, metrics, config):
    """Verify the same current-frame flags used by live inference."""
    from realpc.cloth_quality import frame_quality
    count = len(data['ticks'])
    assert len(metrics) == count
    assert data['cloth_valid'].shape == (count,)
    assert data['cloth_approximate'].shape == (count,)
    assert data['cloth_approximate'].all(), 'Estimated cloth must remain marked approximate'
    assert np.array_equal(data['cloth_fold_approximate'], data['cloth_approximate'])
    assert not bool(data['cloth_geometry_observed_only'])
    for t, metric in enumerate(metrics):
        assert metric['t'] == t
        quality = frame_quality(data['cloth_points'][:, t], metric, config)
        assert quality == metric['quality'], (t, 'saved live quality differs')
        assert quality['valid'] == bool(data['cloth_valid'][t]), (t, 'saved validity differs')


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--episodes-root', required=True)
    parser.add_argument('--config', default=str(ROOT / 'configs/scene_pipeline_v2.json'))
    args = parser.parse_args()
    root, recordings = Path(args.dataset), Path(args.episodes_root)
    expected = sorted(p.relative_to(recordings) for task in ('sweep', 'hang', 'greycloth')
                      for p in (recordings / task).glob('episode_*') if p.is_dir())
    actual = sorted(p.relative_to(root).with_suffix('') for p in root.glob('*/episode_*.npz')
                    if p.stem.split('_')[-1].isdigit())
    if expected != actual:
        raise ValueError(f'Missing: {set(expected)-set(actual)}; unexpected: {set(actual)-set(expected)}')
    cfg = load_scene_config(args.config)
    config_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    records, implementations, totals = [], {}, {}
    for key in expected:
        path = (root / key).with_suffix('.npz')
        qa = json.loads(path.with_name(path.stem + '_qa.json').read_text())
        data = load(str(path))
        from realpc.io import Episode
        source = Episode(str(recordings / key))
        count = len(source.low['t_cap'])
        assert qa['ticks'] == len(data['points']) == count, str(key)
        assert np.array_equal(data['ticks'], np.arange(count)), str(key)
        assert data['valid'].shape == data['points'].shape[:2], str(key)
        assert np.isfinite(data['points'][data['valid']]).all(), str(key)
        assert bool(data['causal']) and str(data['pipeline']) == 'shared_scene_v2', str(key)
        assert qa['config_sha256'] == str(data['config_sha256']) == config_hash, str(key)
        assert qa['file_sha256'] == sha(path), str(key)
        assert qa['pipeline']['shared_live_conversion_step'], str(key)
        assert qa['pipeline']['segmentation']['one_shared_stream'], str(key)
        implementation = qa['implementation_sha256']
        assert implementation == str(data['implementation_sha256']), str(key)
        previous = implementations.setdefault(key.parts[0], implementation)
        assert previous == implementation, f'Mixed implementations within {key.parts[0]}'
        provenance = json.loads(path.with_name(path.stem + '_run.json').read_text())
        for name, digest in provenance['source_sha256'].items():
            source_path = Path(name) if Path(name).is_absolute() else ROOT / name
            assert sha(source_path) == digest, f'Source changed: {source_path}'
        for camera in ('top', 'side'):
            with np.load(path.with_name(path.stem + '_' + camera + '_masks.npz')) as masks:
                assert len(masks['labels']) == count and bool(masks['causal']), str(key)
        if key.parts[0] == 'greycloth':
            metrics = json.loads(path.with_name(path.stem + '_cloth_metrics.json').read_text())
            verify_cloth_quality(data, metrics, cfg)
        task = totals.setdefault(key.parts[0], dict(episodes=0, frames=0, sam_frames=0, conversion_seconds=0.0))
        task['episodes'] += 1
        task['frames'] += count
        task['sam_frames'] += qa['pipeline']['segmentation']['sam_frames']
        task['conversion_seconds'] += qa['seconds']
        records.append(dict(file=str(path.relative_to(root)), sha256=qa['file_sha256'],
                            bytes=path.stat().st_size, qa=qa))
        print('VERIFIED', key, count, flush=True)
        del data
    for task in totals.values():
        task['sam_frame_rate'] = task['sam_frames'] / task['frames']
        task['mean_job_fps'] = task['frames'] / task['conversion_seconds']
    from realpc.v4 import SCHEMA
    schema = dict(SCHEMA)
    schema.update(cloth_points='[Nc,T,3] float32, causal deformable clean shirt template',
                  cloth_valid='[T] bool; shared current-frame finite/strain/crossing/two-view support gate',
                  cloth_approximate='[T] bool; true for every estimated template frame, including initialization',
                  cloth_fold_approximate='[T] bool; legacy alias of cloth_approximate',
                  cloth_geometry_observed_only='false; hidden material is inferred, not directly measured',
                  cloth_observed='[Nc,T] bool, within 12mm of either current camera depth',
                  config_sha256='SHA256 of canonical shared JSON config',
                  implementation_sha256='SHA256 of source/calibration/library hash mapping')
    (root / 'schema.json').write_text(json.dumps(schema, indent=2))
    library = {}
    for path in sorted((ROOT / 'assets/objects_v3').glob('*/*.ply')):
        meta_path = path.with_suffix('.json')
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        entry = {k: meta[k] for k in ('version', 'n_scans', 'extents_m', 'watertight') if k in meta}
        entry['ply_sha256'] = sha(path)
        library[str(path.relative_to(ROOT / 'assets/objects_v3').with_suffix(''))] = entry
    manifest = dict(dataset=f'agentwm_v2 ({len(records)} episodes)',
                    created=datetime.now(timezone.utc).isoformat(),
                    code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    converter='realpc.scene_pipeline.ScenePipeline.step (same live/conversion causal step)',
                    causal=True, calibration=dict(file='configs/calib_yam_rig_v3.json',
                        sha256=sha(ROOT / 'configs/calib_yam_rig_v3.json'), const_lag_ticks=3),
                    object_masks=f"One shared SAM2 scene stream; interval {cfg['segmentation']['interval']} plus recovery; depth/flow between refreshes",
                    library=library, groups={'0':'robot_left', '1':'robot_right', '2':'static', '3':'library object', '4':'cloth'},
                    points_per_episode=dict(robot=9600, object='1500 per library object', static='first-frame voxel cloud', cloth=4000),
                    schema=schema, loader='realpc.dataset.load(npz) -> points [T,N,3], group, obj_id, is_finger, color, valid',
                    aggregate=dict(episodes=len(records), total_bytes=sum(r['bytes'] for r in records)),
                    alignment_checks=[], pipeline='shared_scene_v2', complete=True, episodes=records, totals=totals,
                    config_sha256=config_hash, implementations=implementations,
                    notes=['Every episode loaded and valid points checked.',
                           'Job timings include conversion and saving; overlapping jobs are not isolated benchmarks.',
                           'All cloth is approximate; validity uses the same current-frame gate as live inference, not a fold timestamp.',
                           'Alignment to propagated masks measures consistency, not independent accuracy.'])
    alignment = root / 'qa/alignment_summary.json'
    if alignment.exists():
        report = json.loads(alignment.read_text())
        manifest['alignment_checks'] = [dict(episode=key, results=value)
                                        for key, value in report['episodes'].items()]
        manifest['alignment_summary'] = report['aggregate']
        manifest['depth_agreement'] = report.get('depth_agreement', {})
        manifest['pose_availability'] = report.get('pose_availability', {})
        manifest['alignment_note'] = report['note']
    temp = root / 'manifest.tmp.json'
    temp.write_text(json.dumps(manifest, indent=2))
    temp.replace(root / 'manifest.json')
    files = sorted(p for p in root.rglob('*') if p.is_file() and p.name != 'SHA256SUMS')
    (root / 'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(root)}\n' for p in files))
    print(json.dumps(totals, indent=2))


if __name__ == '__main__':
    main()
