#!/usr/bin/env python
"""FK mesh overlay stills (top + side camera, N frames) with outline->edge distance and depth residual per arm.

  python scripts/robot_overlay.py --episode <dir> --out grid.png --json metrics.json [--corr handeye.json] [--n 6]
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
from realpc.io import DepthSource, Episode  # noqa: E402
from realpc.robot_overlay import MeshProjector, depth_residual, draw, edge_distance  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--episode", required=True); ap.add_argument("--out", required=True); ap.add_argument("--json", required=True)
ap.add_argument("--config", default=os.path.join(ROOT, "configs", "yam_bimanual.json"))
ap.add_argument("--corr", default=None, help="hand-eye json {cam: {arm: 4x4}}")
ap.add_argument("--n", type=int, default=6); ap.add_argument("--depth_cache", default=os.path.join(ROOT, ".cache"))
ap.add_argument("--label", default="recorder calibration")
a = ap.parse_args()
ep = Episode(a.episode)
arms = build_arms(load_config(a.config), ep.calib)
proj = MeshProjector(arms.robot)
corr = {c: {arm: np.array(T) for arm, T in d.items()} for c, d in json.load(open(a.corr)).items()} if a.corr else {}
ticks = np.linspace(int(len(ep.low["t_wall"]) * 0.08), len(ep.low["t_wall"]) - 5, a.n).astype(int)
rows, metrics = [], {"label": a.label, "frames": [int(t) for t in ticks], "per_cam": {}}
for cam in ("top", "side"):
    cap = cv2.VideoCapture(ep.video(cam)); K = ep.intrinsics(cam)
    src = DepthSource(ep, cam, a.depth_cache)
    acc = {arm: {"edge_px": [], "depth_med_m": [], "depth_within2cm": []} for arm in arms.T_base}
    tiles = []
    for t in ticks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t)); ok, img = cap.read()
        H, W = img.shape[:2]
        q = {arm: ep.low[f"joint_pos_{arm}"][t] for arm in arms.T_base}
        rd = proj.render({arm: arms.qmap(q[arm]) for arm in q}, arms.T_base, ep.low[f"cam_pose_{cam}"][t], K, H, W, corr.get(cam))
        D = src.depth(int(t))
        txt = []
        for arm, (mask, dep) in rd.items():
            if mask.sum() < 200:
                continue
            e, n = edge_distance(img, mask)
            dm, dw, nd = depth_residual(D, dep, mask)
            acc[arm]["edge_px"].append(e); acc[arm]["depth_med_m"].append(dm); acc[arm]["depth_within2cm"].append(dw)
            txt.append(f"{arm[0].upper()}: edge {e:.1f}px  |dz| {dm * 1000 if np.isfinite(dm) else float('nan'):.0f}mm")
        tile = draw(img, rd)
        for j, s in enumerate(txt):
            cv2.putText(tile, s, (10, 28 + 26 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.7 if cam == "top" else 0.55, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(tile, s, (10, 28 + 26 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.7 if cam == "top" else 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(tile, f"{cam} tick {t}", (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(cv2.resize(tile, (960, 540)))
    h = len(tiles) // 2
    rows.append(np.concatenate(tiles[:h], 1)); rows.append(np.concatenate(tiles[h:2 * h], 1))
    metrics["per_cam"][cam] = {arm: {k: (float(np.nanmedian(v)) if len(v) and np.isfinite(v).any() else None) for k, v in d.items()}
                               for arm, d in acc.items()}
grid = np.concatenate(rows, 0)
bar = np.zeros((40, grid.shape[1], 3), np.uint8)
cv2.putText(bar, f"{ep.task}/{ep.name}  FK mesh overlay ({a.label})  red = left arm, magenta = right arm", (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
cv2.imwrite(a.out, np.concatenate([bar, grid], 0))
json.dump(metrics, open(a.json, "w"), indent=1)
print(json.dumps(metrics["per_cam"], indent=1))
