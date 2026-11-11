"""Actual live packets must reproduce converted poses, points and masks."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from realpc.io import Episode, DepthStream
from realpc.calib_rig import RigCalib
from realpc.config import load_config, build_arms
from realpc.diffcalib import TorchArm
from realpc.v4 import StreamConverter
from realpc.scene_pipeline import ScenePipeline
from realpc.dataset import load

p = argparse.ArgumentParser()
p.add_argument('--converted', required=True)
p.add_argument('--out', required=True)
p.add_argument('--frames', type=int, default=12)
a = p.parse_args()
path = Path(a.converted)
d = load(str(path))
task, episode = str(d['task']), str(d['episode'])
ep = Episode(f'/home/howardji/data_ext/agentwm_probe/hf/{task}/{episode}')
config = json.loads(path.with_name(path.stem + '_config.json').read_text())
cfg = load_config(ROOT / 'configs/yam_bimanual.json')
r = cfg['robot']
tarm = TorchArm(str(Path(cfg['_dir']) / r['urdf']), r['arm_joints'], r['finger_joints'])
cal = RigCalib.load(ROOT / 'configs/calib_yam_rig_v3.json')
library = ROOT / 'assets/objects_v3' / task
conv = StreamConverter(ep, cal, tarm, build_arms(cfg, ep.calib), str(library), seg_dir=None)
pipe = ScenePipeline(conv, config, cloth=task == 'greycloth')
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
Path(a.out).write_text(json.dumps(report, indent=2))
print('PASS', report)
