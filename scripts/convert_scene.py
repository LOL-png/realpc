#!/usr/bin/env python
"""Recording adapter for ScenePipeline.step; the live API is the same step."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from realpc.calib_rig import RigCalib
from realpc.config import load_config, build_arms
from realpc.diffcalib import TorchArm
from realpc.io import Episode, DepthStream
from realpc.v4 import StreamConverter, SCHEMA
from realpc.scene_pipeline import ScenePipeline, load_scene_config


def atomic_npz(path, **values):
    tmp = path.with_name(path.stem + '.tmp.npz')
    np.savez_compressed(tmp, **values)
    tmp.replace(path)


def convert_episode(ep_path, out_root, config, assets, calibration, stop=None, skip=False, save_masks=False,
                    cloth_preview_every=0):
    start = time.perf_counter()
    ep = Episode(str(ep_path))
    dest = out_root / ep.task
    dest.mkdir(parents=True, exist_ok=True)
    output = dest / (ep.name + '.npz')
    qa_path = dest / (ep.name + '_qa.json')
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    count = min(len(ep.low['t_cap']), stop or len(ep.low['t_cap']))
    sources = [Path(__file__), Path(calibration), *sorted((ROOT / 'realpc').glob('*.py')),
               *sorted((ROOT / 'realpc').glob('cloth*.cpp')),
               *sorted((assets / ep.task).glob('*.ply')), *sorted((assets / ep.task).glob('*.json')),
               *sorted((assets / ep.task).glob('*.npz'))]
    if config.get('cloth',{}).get('shirt_asset'):
        sources.append(Path(config['cloth']['shirt_asset']))
    source_hashes = {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    implementation_hash = hashlib.sha256(json.dumps(source_hashes,sort_keys=True).encode()).hexdigest()
    if skip and output.exists() and qa_path.exists():
        old = json.loads(qa_path.read_text())
        if old.get('config_sha256') == config_hash and old.get('implementation_sha256') == implementation_hash and old.get('ticks') == count:
            return old
    cfg = load_config(ROOT / 'configs/yam_bimanual.json')
    r = cfg['robot']
    tarm = TorchArm(str(Path(cfg['_dir']) / r['urdf']), r['arm_joints'], r['finger_joints'])
    cal = RigCalib.load(calibration)
    arms = build_arms(cfg, ep.calib)
    library = assets / ep.task
    if not library.is_dir():
        library = out_root / '_empty_library'
        library.mkdir(exist_ok=True)
    conv = StreamConverter(ep, cal, tarm, arms, str(library), seg_dir=None)
    pipeline = ScenePipeline(conv, config, cloth=ep.task == 'greycloth')
    cameras = [c for c in ('top', 'side', 'left_wrist', 'right_wrist')
               if (Path(ep.path) / (c + '_depth.mkv')).exists()]
    streams = {c: DepthStream(str(Path(ep.path) / (c + '_depth.mkv')), 0) for c in cameras}
    videos = {c: cv2.VideoCapture(ep.video(c)) for c in cameras}
    names = [o.name for o in conv.objs]
    values = dict(robot_T=[], q_left=[], q_right=[])
    obj = {key: [] for key in ('obj_T', 'obj_res', 'obj_held', 'obj_valid', 'obj_support', 'obj_mode')}
    cloth_points, cloth_observed, cloth_metrics, cloth_X = [], [], [], []
    cloth_valid, cloth_approximate = [], []
    preview = dest / (ep.name + '_preview')
    used_masks = {c: [] for c in ('top', 'side')}
    input_seconds = []
    try:
        for t in range(count):
            tic = time.perf_counter()
            frames = {}
            for c in cameras:
                _, depth = streams[c].next()
                ok, image = videos[c].read()
                if not ok or depth is None:
                    raise RuntimeError(f'Missing current {c} frame {t}')
                frames[c] = depth.astype(np.float32) * float(ep.meta['depth_scale'][c]), image
            # Fixed calibrated lag, evaluated only at or before the current tick.
            joint_tick = cal.robot_tick(ep, 'top', t)
            if joint_tick > t:
                raise ValueError('Calibration requests future joints')
            q = {a: cal.q_at(ep, a, joint_tick) for a in ('left', 'right')}
            input_seconds.append(time.perf_counter() - tic)
            row = pipeline.step(t, frames, q)
            values['robot_T'].append(row['robot_T'])
            for a in ('left', 'right'):
                values['q_' + a].append(row['q'][a])
            for key in obj:
                default = (0, 0) if key == 'obj_support' else ''
                obj[key].append([row.get(key, {}).get(n, default) for n in names])
            if save_masks:
                for c in used_masks:
                    used_masks[c].append(row['scene_labels'][c].copy())
            if 'cloth_points' in row:
                cloth_points.append(row['cloth_points'])
                cloth_observed.append(row['cloth_observed'])
                cloth_metrics.append(row['cloth_metrics'])
                cloth_X.append(row['cloth_vertices'])
                cloth_valid.append(row['cloth_valid'])
                cloth_approximate.append(row['cloth_approximate'])
                if cloth_preview_every:
                    # Diagnostic output only: no cached packet is read by step.
                    packets = preview / 'inputs' / 'frames'
                    packets.mkdir(parents=True, exist_ok=True)
                    atomic_npz(packets / f'{t:04d}.npz', **pipeline.last_packet)
                    if t % cloth_preview_every == 0 or t == count - 1:
                        atomic_npz(preview / 'mesh.npz', X=np.array(cloth_X), V0=pipeline.cloth.cm.V0,
                            F=pipeline.cloth.cm.F, sample_faces=pipeline.cloth.cm.sf,
                            sample_bary=pipeline.cloth.cm.sb, ticks=np.arange(t+1), causal=True,
                            partial=t+1<count)
                        (preview / 'metrics.json').write_text(json.dumps(cloth_metrics))
                        (preview / 'template_info.json').write_text(json.dumps(pipeline.cloth.template_info, indent=2))
            if t % 50 == 0:
                progress=dict(tick=t,elapsed_seconds=time.perf_counter()-start,
                    sam_frames=len(pipeline.masks.events),config_sha256=config_hash,
                    health=pipeline.masks.health,latest_events=pipeline.masks.events[-2:])
                if 'cloth_metrics' in row:
                    progress['cloth_metrics'] = row['cloth_metrics']
                    progress['cloth_grasp_events'] = pipeline.cloth.events
                progress_path=dest/(ep.name+'_progress.json')
                temp=progress_path.with_suffix('.tmp.json');temp.write_text(json.dumps(progress,indent=2));temp.replace(progress_path)
                print(ep.task, ep.name, t, 'seconds', round(time.perf_counter() - start, 1),
                      'SAM frames', len(pipeline.masks.events), flush=True)
    finally:
        for stream in streams.values():
            stream.close()
        for video in videos.values():
            video.release()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    data = {k: np.asarray(v, np.float32) for k, v in values.items()}
    for key, value in obj.items():
        dtype = {'obj_mode':'<U16','obj_valid':bool,'obj_held':np.int8,'obj_support':np.int32}.get(key,np.float32)
        array = np.asarray(value, dtype=dtype)
        if not names:
            shape = (0, count, 4, 4) if key == 'obj_T' else ((0, count, 2) if key in ('obj_res', 'obj_support') else (0, count))
            data[key] = np.empty(shape, dtype=dtype)
        else:
            data[key] = np.swapaxes(array, 0, 1)
    data.update(ticks=np.arange(count), t_cap=ep.low['t_cap'][:count], task=ep.task, episode=ep.name,
        robot_local=arms.local.astype(np.float32), robot_body=arms.body.astype(np.int16),
        robot_bodies=np.array([f'{x}/{y}' for x, y in arms.bodies]), obj_names=np.array(names),
        obj_static=np.array([False for _ in names]), static_points=conv.static[0], static_color=conv.static[1],
        calib=Path(calibration).name, const_lag=cal.const_lag, commit=commit, causal=True, start=0,
        pipeline='shared_scene_v2', config_sha256=config_hash, implementation_sha256=implementation_hash)
    if names:
        data['obj_local'] = np.concatenate([o.samples for o in conv.objs]).astype(np.float32)
        data['obj_index'] = np.concatenate([np.full(len(o.samples), i) for i, o in enumerate(conv.objs)]).astype(np.int16)
        from scipy.spatial import cKDTree
        colors = []
        for o in conv.objs:
            color = np.asarray(o.mesh.visual.vertex_colors)[:, :3]
            _, nearest = cKDTree(o.mesh.vertices).query(o.samples)
            colors.append(color[nearest])
        data['obj_color'] = np.concatenate(colors).astype(np.uint8)
    for key, value in ep.low.items():
        if key.startswith(('eef_', 'gripper_', 'action_', 'engaged_', 'joint_pos_')) and len(value) >= count:
            data['low_' + key] = value[:count]
    if cloth_points:
        data.update(cloth_points=np.stack(cloth_points, axis=1), cloth_observed=np.stack(cloth_observed, axis=1),
            cloth_valid=np.asarray(cloth_valid, bool), cloth_approximate=np.asarray(cloth_approximate, bool),
            cloth_fold_approximate=np.asarray(cloth_approximate, bool), cloth_variant='shared_scene_deformable_v2',
            cloth_coordinate_frame='v3_world', cloth_geometry_observed_only=False)
        atomic_npz(dest / (ep.name + '_cloth_mesh.npz'), X=np.array(cloth_X), V0=pipeline.cloth.cm.V0,
                   F=pipeline.cloth.cm.F, sample_faces=pipeline.cloth.cm.sf, sample_bary=pipeline.cloth.cm.sb,
                   ticks=np.arange(count), causal=True)
        (dest / (ep.name + '_cloth_metrics.json')).write_text(json.dumps(cloth_metrics))
        (dest / (ep.name + '_cloth_template.json')).write_text(json.dumps(pipeline.cloth.template_info, indent=2))
    if save_masks:
        for c, labels in used_masks.items():
            atomic_npz(dest / (ep.name + '_' + c + '_masks.npz'), labels=np.array(labels), start=0,
                       names=np.array(names + (['cloth'] if pipeline.cloth else [])), causal=True)
    atomic_npz(output, **data)
    from realpc.dataset import load
    loaded = load(str(output))
    assert len(loaded['points']) == count
    assert np.isfinite(loaded['points'][loaded['valid']]).all(), 'Nonfinite valid points'
    qa = dict(episode=f'{ep.task}/{ep.name}', ticks=count, config_sha256=config_hash, implementation_sha256=implementation_hash,
              seconds=time.perf_counter() - start, input_seconds=sum(input_seconds),
              pipeline=pipeline.summary(), commit=commit, file_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
              load_verified=True, n_robot=len(arms.local), n_static=len(conv.static[0]),
              n_object=sum(len(o.samples) for o in conv.objs), n_cloth=len(cloth_points[0]) if cloth_points else 0)
    (dest / (ep.name + '_run.json')).write_text(json.dumps(dict(commit=commit,config_sha256=config_hash,implementation_sha256=implementation_hash,source_sha256=source_hashes),indent=2))
    (dest / (ep.name + '_config.json')).write_text(json.dumps(config, indent=2))
    qa_tmp=qa_path.with_suffix('.tmp.json')
    qa_tmp.write_text(json.dumps(qa,indent=2))
    qa_tmp.replace(qa_path)
    print('DONE', qa['episode'], round(qa['seconds'], 1), 'seconds', flush=True)
    return qa


def main():
    preset = globals().get('PUBLIC_PRESET')
    parser = argparse.ArgumentParser(prog=f'python -m realpc {preset} convert' if preset else None)
    parser.add_argument('--episodes', nargs='+', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--config', default=str(ROOT / 'configs/pipeline_accurate.json'),
                        help=argparse.SUPPRESS if preset else 'Shared pipeline configuration')
    parser.add_argument('--assets', default=str(ROOT / 'assets/objects_v3'))
    parser.add_argument('--calibration', default=str(ROOT / 'configs/calib_yam_rig_v3.json'))
    parser.add_argument('--stop', type=int)
    parser.add_argument('--skip-existing', action='store_true')
    parser.add_argument('--save-masks', action='store_true')
    parser.add_argument('--cloth-preview-every', type=int, default=0,
                        help='Validation only: save current packets and a cloth prefix every N frames')
    args = parser.parse_args()
    config = load_scene_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for episode in args.episodes:
        convert_episode(Path(episode), out, config, Path(args.assets), args.calibration,
                        args.stop, args.skip_existing, args.save_masks, args.cloth_preview_every)
    (out / 'schema.json').write_text(json.dumps(SCHEMA, indent=2))
    records = [json.loads(p.read_text()) for p in sorted(out.glob('*/*_qa.json'))]
    (out / 'manifest.json').write_text(json.dumps(dict(pipeline='shared_scene_v2', episodes=records), indent=2))


if __name__ == '__main__':
    main()
