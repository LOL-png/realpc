#!/usr/bin/env python
"""Per-frame side-camera lag from the robot: for moving frames t, find the q-time shift d (ticks) for which the FK
mesh best matches the side image at t (score = share of arm pixels whose measured depth is within 2 cm of the rendered
depth, minus 0.02 x outline->edge px). Compares with the DTW side<->top map (realpc.sync).

  python scripts/side_lag_fk.py --episode <dir> --out lag.json [--every 12] [--max_shift 75]
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
from realpc.robot_overlay import MeshProjector, depth_residual, edge_distance  # noqa: E402
from realpc.sync import side_frame_map  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--episode", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--config", default=os.path.join(ROOT, "configs", "yam_bimanual.json"))
ap.add_argument("--depth_cache", default=os.path.join(ROOT, ".cache"))
ap.add_argument("--every", type=int, default=12); ap.add_argument("--max_shift", type=int, default=75); ap.add_argument("--step", type=int, default=3)
a = ap.parse_args()
ep = Episode(a.episode); arms = build_arms(load_config(a.config), ep.calib); proj = MeshProjector(arms.robot, max_faces=2000)
L = ep.low; N = len(L["t_wall"])
q = np.concatenate([L["joint_pos_left"][:, :6], L["joint_pos_right"][:, :6]], 1)
spd = np.r_[0, np.abs(np.diff(q, axis=0)).max(1)]
src = DepthSource(ep, "side", a.depth_cache); K = ep.intrinsics("side"); cap = cv2.VideoCapture(ep.video("side"))
dtw = side_frame_map(ep, a.depth_cache)                                 # top tick -> side frame
res = []
for t in range(a.max_shift, N - a.max_shift, a.every):
    if np.convolve(spd, np.ones(9) / 9, mode="same")[t] < 0.01:      # moving frames only
        continue
    cap.set(cv2.CAP_PROP_POS_FRAMES, t); ok, img = cap.read()
    D = src.depth(t); H, W = D.shape
    sc = {}
    for d in range(-a.max_shift, a.max_shift + 1, a.step):
        rd = proj.render({arm: arms.qmap(L[f"joint_pos_{arm}"][t + d]) for arm in arms.T_base}, arms.T_base, L["cam_pose_side"][t], K, H, W)
        v = []
        for m, dep in rd.values():
            if m.sum() > 400:
                dm, dw, nd = depth_residual(D, dep, m)
                e, _ = edge_distance(img, m)
                if np.isfinite(dw):
                    v.append(dw - 0.02 * e)
        sc[d] = float(np.mean(v)) if v else np.nan
    good = {k: s for k, s in sc.items() if np.isfinite(s)}
    if not good:
        continue
    best = max(good, key=good.get)
    srt = sorted(good.values())
    margin = srt[-1] - (srt[-2] if len(srt) > 1 else srt[-1])
    # DTW says: side frame t shows top tick u with dtw[u] = t  -> robot time u; lag_dtw = u - t
    u = int(np.argmin(np.abs(dtw - t)))
    res.append({"side_frame": int(t), "fk_best_shift": int(best), "score_best": good[best], "score_at_0": good.get(0, np.nan),
                "dtw_shift": int(u - t)})
    print(res[-1], flush=True)
fk = np.array([r["fk_best_shift"] for r in res]); dt = np.array([r["dtw_shift"] for r in res])
summary = {"n_frames": len(res), "fk_shift_median": float(np.median(fk)) if len(fk) else None,
           "fk_shift_p10_p90": [float(np.percentile(fk, 10)), float(np.percentile(fk, 90))] if len(fk) else None,
           "fk_vs_dtw_agree_within_6_ticks": float(np.mean(np.abs(fk - dt) <= 6)) if len(fk) else None,
           "fk_score_gain_over_0": float(np.nanmean([r["score_best"] - r["score_at_0"] for r in res])) if res else None}
json.dump({"summary": summary, "frames": res}, open(a.out, "w"), indent=1)
print("SUMMARY", summary)
