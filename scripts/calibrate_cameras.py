#!/usr/bin/env python
"""Refine each camera's pose from the robot (FK mesh silhouette vs image edges, see realpc/cam_calib.py), using frames
from several episodes of the same rig; held-out frames for the report. Writes a versioned rig calibration json.

  python scripts/calibrate_cameras.py --episodes <ep1> <ep2> ... --out configs/calib_<rig>_<date>.json
"""
import argparse
import datetime
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from realpc.cam_calib import calibrate_camera  # noqa: E402
from realpc.config import build_arms, load_config  # noqa: E402
from realpc.io import Episode  # noqa: E402
from realpc.robot_overlay import MeshProjector, edge_distance  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", nargs="+", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--config", default=os.path.join(ROOT, "configs", "yam_bimanual.json"))
ap.add_argument("--per_episode", type=int, default=14)
a = ap.parse_args()
cfg = load_config(a.config)
eps = [Episode(p) for p in a.episodes]
arms = build_arms(cfg, eps[0].calib)
proj = MeshProjector(arms.robot, max_faces=3000)
out = {"rig": "bimanual YAM, top + side RealSense", "created": datetime.datetime.now().isoformat(timespec="seconds"),
       "method": "FK mesh silhouette -> photo Canny edges, robust LM over the camera pose (realpc/cam_calib.py)",
       "world": "left arm base", "T_lb_rb": eps[0].calib["T_lb_rb"], "cameras": {}}
for cam in ("top", "side"):
    fit, held = [], []
    used = []
    for ep in eps:
        cap = cv2.VideoCapture(ep.video(cam))
        N = len(ep.low["t_wall"])
        ticks = np.linspace(int(N * 0.05), N - 5, a.per_episode).astype(int)
        for j, t in enumerate(ticks):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t)); ok, img = cap.read()
            qm = {arm: arms.qmap(ep.low[f"joint_pos_{arm}"][t]) for arm in arms.T_base}
            (held if j % 3 == 2 else fit).append((img, qm, ep, int(t)))
            used.append(f"{ep.task}/{ep.name}:{int(t)}{' (held-out)' if j % 3 == 2 else ''}")
    K = eps[0].intrinsics(cam)
    T0 = np.asarray(eps[0].low[f"cam_pose_{cam}"][0], float)
    # check the recorder pose is constant across episodes
    drift = max(np.abs(np.asarray(ep.low[f"cam_pose_{cam}"][0]) - T0).max() for ep in eps)
    T1, log = calibrate_camera([(i, q) for i, q, _, _ in fit], proj, arms, K, T0)

    def evaluate(T):
        r = {"left": [], "right": []}
        for img, qm, ep, t in held:
            H, W = img.shape[:2]
            rd = proj.render(qm, arms.T_base, T, K, H, W)
            for arm, (m, d) in rd.items():
                if m.sum() > 300:
                    r[arm].append(edge_distance(img, m)[0])
        return {arm: {"median_px": float(np.nanmedian(v)) if v else None, "p90_px": float(np.nanpercentile(v, 90)) if v else None,
                      "n_frames": len(v)} for arm, v in r.items()}
    before, after = evaluate(T0), evaluate(T1)
    D = np.linalg.inv(T0) @ T1
    out["cameras"][cam] = {"T_world_cam": T1.tolist(), "T_world_cam_recorder": T0.tolist(), "recorder_pose_max_diff_between_episodes": float(drift),
                           "correction_rot_deg": float(np.degrees(np.arccos(np.clip((np.trace(D[:3, :3]) - 1) / 2, -1, 1)))),
                           "correction_trans_mm": float(np.linalg.norm(D[:3, 3]) * 1000),
                           "heldout_outline_edge_px_before": before, "heldout_outline_edge_px_after": after,
                           "fit_log": log, "frames": used}
    print(cam, "correction", round(out["cameras"][cam]["correction_rot_deg"], 2), "deg", round(out["cameras"][cam]["correction_trans_mm"], 1),
          "mm | held-out edge px before", before, "after", after, flush=True)
json.dump(out, open(a.out, "w"), indent=1)
print("saved", a.out)
