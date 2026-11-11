"""Actual live packets must reproduce converted poses, points and masks."""
import argparse
import os
import json
from pathlib import Path
import sys
import shutil
import tempfile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from realpc.io import Episode, DepthStream
from realpc.dataset import load
from realpc.live import SceneSetup, build_live_pipeline

p = argparse.ArgumentParser()
p.add_argument('--converted', required=True)
p.add_argument('--out', required=True)
p.add_argument('--frames', type=int, default=12)
a = p.parse_args()
path = Path(a.converted)
d = load(str(path))
task, episode = str(d['task']), str(d['episode'])
ep = Episode(str(Path(os.environ.get('REALPC_EPISODES_ROOT','/home/howardji/data_ext/agentwm_probe/hf')) / task / episode))
config = json.loads(path.with_name(path.stem + '_config.json').read_text())
library = ROOT / 'assets/objects_v3' / task
# The live side receives only static setup files, not an Episode object.
setup_dir = tempfile.TemporaryDirectory(prefix='realpc-live-setup-')
for name in ('meta.json', 'calib.json', 'table_region.json'):
    shutil.copy2(Path(ep.path) / name, Path(setup_dir.name) / name)
setup = SceneSetup(setup_dir.name, task)
assert not hasattr(setup, 'low') and not hasattr(setup, 'video')
pipe = build_live_pipeline(setup, config, cloth=task == 'greycloth', library=library)
conv, cal = pipe.converter, pipe.converter.cal
cameras = [c for c in ('top', 'side', 'left_wrist', 'right_wrist')
           if (Path(ep.path) / (c + '_depth.mkv')).exists()]
streams = {c: DepthStream(str(Path(ep.path) / (c + '_depth.mkv'))) for c in cameras}
videos = {c: cv2.VideoCapture(ep.video(c)) for c in cameras}

def forbidden(*args, **kwargs):
    raise AssertionError('Live pipeline tried to read recorded joints or video')

read_current_q = cal.q_at
cal.q_at = forbidden
ep.video = forbidden
ep.frame = forbidden
ep.cam_pose = forbidden
maximum = 0.
for t in range(min(a.frames, len(d['ticks']))):
    frames = {}
    for c in cameras:
        _, depth = streams[c].next()
        ok, image = videos[c].read()
        assert ok
        frames[c] = depth.astype(np.float32) * float(ep.meta['depth_scale'][c]), image
    joint_tick = cal.robot_tick(ep, 'top', t)
    assert joint_tick <= t, 'Input adapter requested future joints'
    joints = {side: read_current_q(ep, side, joint_tick) for side in ('left', 'right')}
    recorded_low = ep.low
    ep.low = None  # Direct recorded-array reads are forbidden inside live step too.
    try:
        row = pipe.step(t, frames, joints)
    finally:
        ep.low = recorded_low
    points = pipe.points(row)
    finite = np.isfinite(points) & np.isfinite(d['points'][t])
    error = float(np.max(np.abs(points[finite] - d['points'][t][finite])))
    maximum = max(maximum, error)
    assert np.array_equal(np.isfinite(points), np.isfinite(d['points'][t]))
    assert np.allclose(row['robot_T'],d['robot_T'][t],rtol=0,atol=1e-6)
    if error>=1e-4:
        detail={str(g):float(np.nanmax(np.abs(points[d['group']==g]-d['points'][t,d['group']==g]))) for g in np.unique(d['group'])}
        print('POINT ERROR BY GROUP',detail,flush=True)
        print('POSE DIFFERENCES',{str(n):float(np.nanmax(abs(row['obj_T'][str(n)]-d['obj_T'][i,t]))) for i,n in enumerate(d['obj_names'])},flush=True)
        print('LOCAL SAMPLE DIFFERENCES',{o.name:float(np.max(abs(o.samples-d['obj_local'][d['obj_index']==i]))) for i,o in enumerate(conv.objs)},flush=True)
    assert error < 1e-4, (task, t, error)  # <=0.1mm after rigid rotation and point expansion
    assert row['max_observed_tick'] == t
    if 'cloth_approximate' in d:
        assert row['cloth_valid'] == bool(d['cloth_valid'][t]), (task, t, 'cloth_valid')
        assert row['cloth_approximate'] == bool(d['cloth_approximate'][t]), (task, t, 'cloth_approximate')
    for cam in ('top', 'side'):
        mask_path = path.with_name(path.stem + '_' + cam + '_masks.npz')
        if mask_path.exists():
            labels = np.load(mask_path)['labels'][t]
            assert np.array_equal(labels, row['scene_labels'][cam]), (task, t, cam)
for s in streams.values():
    s.close()
for v in videos.values():
    v.release()
report = dict(task=task, frames=pipe.tick + 1, maximum_point_difference_m=maximum,
              recorded_joint_and_video_access_blocked=True, config_sha256=pipe.config_hash,
              sam_frames=len(pipe.masks.events), exact_saved_masks=True)
report['cloth_quality_flags_checked'] = 'cloth_approximate' in d
report['recorded_low_dim_access_blocked_during_step'] = True
report['recorded_frame_and_camera_pose_access_blocked'] = True
report['live_constructor_uses_setup_only'] = True
report['live_setup_files'] = sorted(p.name for p in Path(setup_dir.name).iterdir())
Path(a.out).write_text(json.dumps(report, indent=2))
setup_dir.cleanup()
print('PASS', report)
