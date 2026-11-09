#!/usr/bin/env python
"""Per-camera content lag vs the robot joint stream: cross-correlate image motion energy (mean |frame difference|,
grey, 1/8 resolution, smoothed) with the joint speed. lag d > 0: the camera frame at tick t shows the robot at tick t - d.

  python scripts/sync_cameras.py --episode <dir> [--out lags.json] [--max_lag 90]
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from realpc.io import Episode  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--episode", required=True); ap.add_argument("--out", default=None); ap.add_argument("--max_lag", type=int, default=90)
a = ap.parse_args()
ep = Episode(a.episode); L = ep.low
spd = sum(np.r_[0, np.linalg.norm(np.diff(L[f"joint_pos_{arm}"][:, :6], axis=0), axis=1)] for arm in ("left", "right"))
sm = lambda x: np.convolve(x, np.ones(5) / 5, mode="same")
spd = sm(spd)
out = {}
for cam in ("top", "side"):
    cap = cv2.VideoCapture(ep.video(cam)); prev = None; m = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        g = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), None, fx=0.125, fy=0.125).astype(np.float32)
        m.append(0.0 if prev is None else float(np.abs(g - prev).mean())); prev = g
    m = np.array(m)
    rep = m < 0.05                                      # repeated frames (no new capture): interpolate
    if rep.any() and (~rep).sum() > 2:
        m[rep] = np.interp(np.flatnonzero(rep), np.flatnonzero(~rep), m[~rep])
    m = sm(m)
    n = min(len(m), len(spd))
    x = (m[:n] - m[:n].mean()) / (m[:n].std() + 1e-9)
    y = (spd[:n] - spd[:n].mean()) / (spd[:n].std() + 1e-9)
    cc = {}
    for d in range(-a.max_lag, a.max_lag + 1):         # x(t) ~ y(t - d)
        if d >= 0:
            cc[d] = float(np.mean(x[d:n] * y[: n - d]))
        else:
            cc[d] = float(np.mean(x[: n + d] * y[-d:n]))
    best = max(cc, key=cc.get)
    out[cam] = {"lag_ticks": int(best), "corr": cc[best], "corr_at_0": cc[0], "repeated_frame_frac": float(rep.mean())}
    print(ep.task, cam, out[cam], flush=True)
if a.out:
    json.dump(out, open(a.out, "w"), indent=1)
