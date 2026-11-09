#!/usr/bin/env python
"""Estimate each camera's content lag relative to the robot joint stream: the shift d (ticks) for which the FK mesh
silhouette from q(t + d) best matches the image at tick t (median outline->edge distance over frames with arm motion).

  python scripts/estimate_camera_lag.py --episode <dir> [--cams top,side] [--max_shift 15]
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from realpc.config import build_arms, load_config  # noqa: E402
from realpc.io import Episode  # noqa: E402
from realpc.robot_overlay import MeshProjector, edge_distance  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--episode", required=True); ap.add_argument("--cams", default="top,side")
ap.add_argument("--max_shift", type=int, default=15); ap.add_argument("--n", type=int, default=24)
ap.add_argument("--config", default=os.path.join(ROOT, "configs", "yam_bimanual.json")); ap.add_argument("--out", default=None)
a = ap.parse_args()
ep = Episode(a.episode); arms = build_arms(load_config(a.config), ep.calib); proj = MeshProjector(arms.robot, max_faces=2500)
L = ep.low; N = len(L["t_wall"])
speed = sum(np.linalg.norm(np.diff(L[f"joint_pos_{arm}"][:, :6], axis=0), axis=1) for arm in arms.T_base)
cand = np.arange(a.max_shift + 5, N - a.max_shift - 5)
ticks = cand[np.argsort(-speed[cand])][: a.n * 4][:: 4]           # frames with arm motion
res = {}
for cam in a.cams.split(","):
    cap = cv2.VideoCapture(ep.video(cam)); K = ep.intrinsics(cam)
    imgs = {}
    for t in ticks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t)); ok, im = cap.read(); imgs[int(t)] = im
    scores = {}
    for d in range(-a.max_shift, a.max_shift + 1):
        vals = []
        for t in ticks:
            im = imgs[int(t)]; H, W = im.shape[:2]
            rd = proj.render({arm: arms.qmap(L[f"joint_pos_{arm}"][t + d]) for arm in arms.T_base}, arms.T_base,
                             L[f"cam_pose_{cam}"][t], K, H, W)
            for arm, (m, _) in rd.items():
                if m.sum() > 300:
                    vals.append(edge_distance(im, m)[0])
        scores[d] = float(np.nanmedian(vals)) if vals else np.nan
    best = min(scores, key=lambda k: scores[k] if np.isfinite(scores[k]) else 1e9)
    res[cam] = {"best_shift_ticks": int(best), "edge_px_at_best": scores[best], "edge_px_at_0": scores[0], "scores": scores}
    print(cam, "best shift", best, "ticks; edge px at best %.2f vs at 0 %.2f" % (scores[best], scores[0]),
          {k: round(v, 2) for k, v in scores.items() if abs(k) <= 6}, flush=True)
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
