#!/usr/bin/env python
"""Image-seeded 2D point tracks for robot<->camera calibration (no FK involved in seeding).

CoTracker3 online, a regular grid of queries seeded every `--every` ticks and tracked for `--life` ticks. Only tracks
that move (> `--min_move` px) are kept; static background tracks carry no hand-eye information. Moving objects are
kept too: the fit drops tracks that no robot link explains.

  python scripts/arm_tracks.py --episode <ep dir> --cam side --out <npz>
Output npz: seg_start [S], tracks [S] object array of (life,N,2) float32 native pixels, vis (life,N) bool,
            depth0 (N,) metres at the seed pixel (0 = none).
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from realpc.io import Episode, read_depth_frames  # noqa: E402
from realpc.v3 import load_cotracker, run_instance  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--episode", required=True); ap.add_argument("--cam", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--every", type=int, default=40); ap.add_argument("--life", type=int, default=80)
ap.add_argument("--grid", type=int, default=12, help="grid step in pixels of the 640-wide working image")
ap.add_argument("--min_move", type=float, default=8.0); ap.add_argument("--depth_cache", default=os.path.expanduser("~/data_ext/realpc_cache"))
a = ap.parse_args()

ep = Episode(a.episode)
cap = cv2.VideoCapture(ep.video(a.cam))
frames = []
while True:
    ok, im = cap.read()
    if not ok:
        break
    H0, W0 = im.shape[:2]
    s = 640.0 / W0
    frames.append(cv2.cvtColor(cv2.resize(im, (640, int(round(H0 * s)))), cv2.COLOR_BGR2RGB))
T = len(frames)
h, w = frames[0].shape[:2]
D = read_depth_frames(os.path.join(ep.path, f"{a.cam}_depth.mkv"), range(0, T, a.every))
dscale = float(ep.meta.get("depth_scale", {}).get(a.cam, 0.001))
model = load_cotracker(os.path.join(ROOT, ".model_cache"))
g = a.grid
ys, xs = np.mgrid[g // 2: h: g, g // 2: w: g]
Q = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float32)
starts, TR, VI, D0 = [], [], [], []
for s0 in range(0, max(1, T - 16), a.every):
    seg = frames[s0: s0 + a.life]
    if len(seg) < 16:
        break
    with torch.no_grad():
        tr, vis = run_instance(model, seg, Q)
    tr = tr / s                                                      # native pixels
    move = np.linalg.norm(tr - tr[:1], axis=-1)
    keep = (np.nanmax(np.where(vis, move, 0), 0) > a.min_move / s) & (vis.mean(0) > 0.4)
    q0 = (Q[keep] / s).round().astype(int)
    d0 = D[s0][np.clip(q0[:, 1], 0, H0 - 1), np.clip(q0[:, 0], 0, W0 - 1)].astype(np.float32) * dscale
    starts.append(s0); TR.append(tr[:, keep].astype(np.float32)); VI.append(vis[:, keep] > 0.5); D0.append(d0)
    print(f"{ep.task}/{ep.name} {a.cam} seg {s0} kept {keep.sum()}/{len(Q)}", flush=True)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
np.savez_compressed(a.out, seg_start=np.array(starts), tracks=np.array(TR, dtype=object), vis=np.array(VI, dtype=object),
                    depth0=np.array(D0, dtype=object), T=T, H=H0, W=W0)
print("saved", a.out)
