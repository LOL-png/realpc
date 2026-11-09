#!/usr/bin/env python
"""Track-based hand-eye refinement (see realpc/handeye.py). Writes {cam: {arm: 4x4}} json + a log.

  python scripts/handeye_tracks.py --episode <dir> --out handeye.json [--cams top,side]
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
from realpc.handeye import solve_correction  # noqa: E402
from realpc.io import DepthSource, Episode  # noqa: E402
from realpc.robot_overlay import MeshProjector  # noqa: E402
from realpc.v3 import lift, load_cotracker, run_instance  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--episode", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--config", default=os.path.join(ROOT, "configs", "yam_bimanual.json"))
ap.add_argument("--cams", default="top,side"); ap.add_argument("--depth_cache", default=os.path.join(ROOT, ".cache"))
ap.add_argument("--every", type=int, default=60); ap.add_argument("--life", type=int, default=90)
ap.add_argument("--per_arm", type=int, default=500)
a = ap.parse_args()
ep = Episode(a.episode)
arms = build_arms(load_config(a.config), ep.calib)
proj = MeshProjector(arms.robot)
model = load_cotracker(os.path.join(ROOT, ".model_cache"))
Ld, Bd = arms.dense_local()
N = len(ep.low["t_wall"])
res, logs = {}, {}
rng = np.random.default_rng(0)
for cam in a.cams.split(","):
    src = DepthSource(ep, cam, a.depth_cache); K = ep.intrinsics(cam)
    cap = cv2.VideoCapture(ep.video(cam))
    obs = {arm: [] for arm in arms.T_base}
    tid = 0
    for t0 in range(10, N - a.life, a.every):
        cap.set(cv2.CAP_PROP_POS_FRAMES, t0)
        frames = []
        for _ in range(a.life):
            ok, im = cap.read(); frames.append(im[..., ::-1].copy() if ok else frames[-1])
        H, W = frames[0].shape[:2]
        q0 = {arm: arms.qmap(ep.low[f"joint_pos_{arm}"][t0]) for arm in arms.T_base}
        rd = proj.render(q0, arms.T_base, ep.low[f"cam_pose_{cam}"][t0], K, H, W)
        D0 = src.depth(t0)
        queries, qarm = [], []
        for arm, (mask, dep) in rd.items():
            m = mask & (D0 > 0.25) & np.isfinite(dep) & (np.abs(D0 - dep) < 0.08)
            m = cv2.erode(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0          # away from silhouette edges
            v, u = np.nonzero(m)
            if len(u) < 30:
                continue
            i = rng.choice(len(u), min(a.per_arm, len(u)), replace=False)
            queries.append(np.stack([u[i], v[i]], 1).astype(np.float64)); qarm += [arm] * len(i)
        if not queries:
            continue
        Q = np.concatenate(queries); qarm = np.array(qarm)
        tracks, vis = run_instance(model, frames, Q)
        for r in range(0, a.life, 3):
            t = t0 + r
            P, ok = lift(tracks[r], src.depth(t), K, ep.low[f"cam_pose_{cam}"][t], edge_tol=0.006)
            ok &= vis[r] > 0.5
            LP = arms.link_poses({arm: ep.low[f"joint_pos_{arm}"][t] for arm in arms.T_base})
            for arm in arms.T_base:
                ai = list(arms.T_base).index(arm)
                sel = np.flatnonzero(ok & (qarm == arm))
                if not len(sel):
                    continue
                # link of each track = nearest FK surface point of this arm at the track's FIRST observation
                bodies = [b for b in range(len(arms.bodies)) if arms.bodies[b][0] == arm and arms.bodies[b][1] != "base"]
                R = np.einsum("nij,nj->ni", LP[Bd, :3, :3], Ld) + LP[Bd, :3, 3]
                mm = np.isin(Bd, bodies)
                from scipy.spatial import cKDTree
                d, j = cKDTree(R[mm]).query(P[sel])
                for s_, dd, jj in zip(sel, d, j):
                    if dd < 0.06:
                        b = int(Bd[mm][jj])
                        obs[arm].append((tid + s_, b, P[s_], LP[b]))
        tid += len(Q)
        print(cam, "instance at tick", t0, "queries", len(Q), "obs", {k: len(v) for k, v in obs.items()}, flush=True)
    res[cam], logs[cam] = {}, {}
    for arm, o in obs.items():
        # a track keeps the link of its first observation
        first = {}
        o2 = []
        for x in o:
            first.setdefault(x[0], x[1])
            if x[1] == first[x[0]]:
                o2.append(x)
        # keep tracks seen in >= 5 frames
        ids, cnt = np.unique([x[0] for x in o2], return_counts=True)
        good = set(ids[cnt >= 5].tolist())
        o2 = [x for x in o2 if x[0] in good]
        if len(o2) < 100:
            print(cam, arm, "too few observations", len(o2)); continue
        C, log = solve_correction(o2, arms, arm)
        res[cam][arm] = C.tolist(); logs[cam][arm] = log
        print(cam, arm, "obs", len(o2), "tracks", len(good), "->", log[-1] if log else None, flush=True)
json.dump(res, open(a.out, "w"), indent=1)
json.dump(logs, open(a.out.replace(".json", "_log.json"), "w"), indent=1)
print("saved", a.out)
