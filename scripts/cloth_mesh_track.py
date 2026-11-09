#!/usr/bin/env python
"""Cloth mesh tracking (realpc/cloth_mesh.py) for a cloth episode -> Tracks npz (robot FK + static table + cloth samples)
and per-step metrics json.

  python scripts/cloth_mesh_track.py --episode <dir> --out tracks.npz [--start S --stop E]
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
from scipy.spatial import cKDTree

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from realpc import depth_refine  # noqa: E402
from realpc.cloth_mesh import ClothMesh, backproject, decimate, tsdf_surface  # noqa: E402
from realpc.config import build_arms, load_config  # noqa: E402
from realpc.io import DepthSource, Episode  # noqa: E402
from realpc.pipeline import pack  # noqa: E402
from realpc.sync import side_frame_map  # noqa: E402
from realpc.tracker import StepOutput, rgb_to_lab, voxel_down  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--episode", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--config", default=os.path.join(ROOT, "configs", "yam_bimanual.json"))
ap.add_argument("--start", type=int, default=0); ap.add_argument("--stop", type=int, default=None)
ap.add_argument("--depth_cache", default=os.path.join(ROOT, ".cache"))
ap.add_argument("--template_cell", type=float, default=0.012)
ap.add_argument("--cloth_dE", type=float, default=20.0)
ap.add_argument("--no_sync", action="store_true", help="use the same tick index for the side camera")
ap.add_argument("--pin_dist", type=float, default=0.03)
a = ap.parse_args()
ep = Episode(a.episode); arms = build_arms(load_config(a.config), ep.calib)
tn, td = ep.table_plane()
srcs = {c: DepthSource(ep, c, a.depth_cache) for c in ("top", "side")}
caps = {c: cv2.VideoCapture(ep.video(c)) for c in ("top", "side")}
Ld, Bd = arms.dense_local()
names = np.array([l for _, l in arms.bodies]); wrist = np.isin(names[Bd], ["link5", "gripper"])
tips = {arm: np.isin(arms.body, [i for i, b in enumerate(arms.bodies) if b[0] == arm and b[1] in ("tip_left", "tip_right")]) for arm in arms.T_base}
grip = {arm: [i for i, b in enumerate(arms.bodies) if b == (arm, "gripper")][0] for arm in arms.T_base}


def read(cam, tick):
    caps[cam].set(cv2.CAP_PROP_POS_FRAMES, int(tick)); ok, im = caps[cam].read()
    return im[..., ::-1].copy()


TR = json.load(open(os.path.join(a.episode, "table_region.json")))
T_tab_inv = np.linalg.inv(np.asarray(TR["T_lb_table"], float)); HS = np.asarray(TR["half_size"], float)
CLOTH = {"lab": None}


def views_at(tick, q, table_lab):
    LP = arms.link_poses(q)
    R = np.einsum("nij,nj->ni", LP[Bd, :3, :3], Ld) + LP[Bd, :3, 3]
    tr_r, tr_w = cKDTree(R[~wrist]), cKDTree(R[wrist])
    out = []
    for cam, w in (("top", 1.0), ("side", 0.5)):
        ft = tick if (cam == "top" or SYNC is None) else int(SYNC[min(tick, len(SYNC) - 1)])   # side frame showing this time
        rgb = read(cam, ft)
        raw = np.asarray(srcs[cam].D[ft])
        D = (depth_refine.refine_frame(rgb, raw, srcs[cam].scale).float().cpu().numpy() if cam == "top"
             else raw.astype(np.float32) * srcs[cam].scale)
        K, T = ep.intrinsics(cam), ep.low[f"cam_pose_{cam}"][ft]
        P, u, v = backproject(D, K, T, stride=1 if cam == "side" else 2)
        lab = rgb_to_lab(rgb[v, u])
        h = P @ tn + td
        rob = np.isfinite(tr_r.query(P, distance_upper_bound=0.025)[0]) | np.isfinite(tr_w.query(P, distance_upper_bound=0.04)[0])
        tab = (h < 0.006) & (np.linalg.norm(lab - table_lab, axis=1) < 15) | (h < -0.01)
        Pt = P @ T_tab_inv[:3, :3].T + T_tab_inv[:3, 3]
        outside = (np.abs(Pt[:, 0]) > HS[0]) | (np.abs(Pt[:, 1]) > HS[1]) | (h > 0.4)       # wall / beyond the table
        drop = rob | tab | outside
        if CLOTH["lab"] is not None:                                                        # cloth colour only
            drop |= np.linalg.norm(lab - CLOTH["lab"], axis=1) > a.cloth_dE
        D2 = D.copy()
        st = 1 if cam == "side" else 2
        for dy in range(st):
            for dx in range(st):
                D2[np.minimum(v[drop] + dy, D.shape[0] - 1), np.minimum(u[drop] + dx, D.shape[1] - 1)] = 0
        out.append((D2, K, T, w, P[~drop]))
    return out, LP


SYNC = None if a.no_sync else side_frame_map(ep, a.depth_cache)
T0 = time.time()
stop = len(ep) if a.stop is None else min(a.stop, len(ep))
idx = list(range(a.start, stop))
f0 = ep.frame(idx[0])
M0, vi = voxel_down(f0.points.astype(np.float64), 0.006, return_index=True)
L0 = rgb_to_lab(f0.colors[vi]); h0 = M0 @ tn + td
table_lab = np.median(L0[h0 < 0.008], 0)
table_pts = voxel_down(M0[(h0 < 0.008) & (np.linalg.norm(L0 - table_lab, axis=1) < 15)], 0.012)
tick0 = ep.ticks[idx[0]]
views, LP = views_at(tick0, f0.q, table_lab)
# initial cloth region: largest cluster of the non-robot, non-table measured points above the table
allP = np.concatenate([v_[4] for v_ in views])
allP = allP[(allP @ tn + td) < 0.25]
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
Pd = voxel_down(allP, 0.01)
pairs = cKDTree(Pd).query_pairs(0.02, output_type="ndarray")
nc, lab_ = connected_components(coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(Pd), len(Pd))), directed=False)
cl = Pd[lab_ == np.bincount(lab_).argmax()]
# cloth colour model from the largest cluster (median Lab of measured points of the cluster, top camera)
Pt_, ut, vt = backproject(views[0][0], ep.intrinsics("top"), ep.low["cam_pose_top"][tick0], stride=2)
rgb0 = read("top", tick0)
in_cl = cKDTree(cl).query(Pt_, distance_upper_bound=0.01)[0] < 0.01
CLOTH["lab"] = np.median(rgb_to_lab(rgb0[vt[in_cl], ut[in_cl]]), 0)
print("cloth colour (Lab)", CLOTH["lab"].round(1), flush=True)
views, LP = views_at(tick0, f0.q, table_lab)          # again, with the cloth colour filter
allP = np.concatenate([v_[4] for v_ in views])
Pd = voxel_down(allP, 0.01)
pairs = cKDTree(Pd).query_pairs(0.02, output_type="ndarray")
nc, lab_ = connected_components(coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(Pd), len(Pd))), directed=False)
cl = Pd[lab_ == np.bincount(lab_).argmax()]
lo, hi = cl.min(0) - 0.03, cl.max(0) + 0.03
Vs, Fs, Ns = tsdf_surface(views, lo, hi)
V, F = decimate(Vs, Fs, a.template_cell)
cm = ClothMesh(V, F)
print("template", len(V), "verts", len(F), "faces; surface verts", len(Vs), f"({time.time() - T0:.0f}s)", flush=True)
nr = len(arms.local)
n_cloth, n_tab = len(cm.sf), len(table_pts)
group = np.concatenate([np.where(np.isin(names[arms.body], ["gripper", "tip_left", "tip_right"]), 0, 1), np.full(n_tab, 2), np.full(n_cloth, 4)]).astype(np.int8)
nb = len(arms.bodies)
body = np.concatenate([arms.body, np.full(n_tab, nb + 1), np.full(n_cloth, nb + 0)]).astype(np.int32)
outs, metrics = [], []
prevG, prev_pins = None, None
for k, i in enumerate(idx):
    f = ep.frame(i); tick = ep.ticks[i]
    if k > 0:
        views, LP = views_at(tick, f.q, table_lab)
        X = cm.X
        lo, hi = X.min(0) - 0.06, X.max(0) + 0.06
        lo[2] = min(lo[2], -0.02)
        Ys, Fs, Ns = tsdf_surface(views, lo, hi)
        pin_idx, pin_tgt = [], []
        G = {arm: LP[grip[arm]] for arm in arms.T_base}
        rp = arms.points(f.q, poses=LP)
        for arm in arms.T_base:
            if float(f.q[arm][-1]) < 0.8 and prevG is not None:
                tp = prev_rp[tips[arm]]
                d = cKDTree(tp).query(X)[0]
                pi = np.flatnonzero(d < a.pin_dist)
                if len(pi):
                    Dg = G[arm] @ np.linalg.inv(prevG[arm])
                    pin_idx.append(pi); pin_tgt.append(X[pi] @ Dg[:3, :3].T + Dg[:3, 3])
        pins = (np.concatenate(pin_idx), np.concatenate(pin_tgt)) if pin_idx else None
        cm.step(Ys, Ns, pins)
    else:
        Ys = Vs
        LP = arms.link_poses(f.q); G = {arm: LP[grip[arm]] for arm in arms.T_base}; rp = arms.points(f.q, poses=LP); pins = None
    prevG, prev_rp = G, rp
    S = cm.samples()
    # metrics
    m = {"step": k}
    if len(Ys):
        d1 = cKDTree(Ys).query(S)[0]
        d2 = cKDTree(S).query(Ys)[0]; d2 = d2[d2 < 0.05]   # fused surface near the cloth (other surfaces excluded)
        m.update(mesh_to_surface_med_mm=float(np.median(d1) * 1000), mesh_to_surface_p90_mm=float(np.percentile(d1, 90) * 1000),
                 mesh_supported_lt1cm=float((d1 < 0.01).mean()), surface_to_mesh_med_mm=float(np.median(d2) * 1000),
                 surface_to_mesh_p90_mm=float(np.percentile(d2, 90) * 1000))
    er = cm.edge_ratio()
    m.update(edge_ratio_median=float(np.median(er)), edge_ratio_p5=float(np.percentile(er, 5)), edge_ratio_p95=float(np.percentile(er, 95)),
             pinned=int(len(pins[0]) if pins is not None else 0))
    metrics.append(m)
    pos = np.concatenate([rp, table_pts, S]).astype(np.float32)
    res = {0: (m.get("mesh_to_surface_med_mm", np.nan) / 1000, int(len(Ys)))}
    outs.append(StepOutput(k, f.t, pos, nr, group, body, np.zeros(len(pos), np.float32), np.zeros(len(pos), bool), {}, res))
    if k % 25 == 0:
        print(f"step {k}/{len(idx)}", {kk: round(vv, 3) for kk, vv in m.items() if kk != 'step'}, f"({time.time() - T0:.0f}s)", flush=True)
tr = pack(outs, ep, idx); tr.variant = "cloth_mesh_v1"
tr.save(a.out)
np.savez_compressed(a.out.replace(".npz", "_mesh.npz"), V0=cm.V0, F=cm.F)
json.dump(metrics, open(a.out.replace(".npz", "_metrics.json"), "w"))
print("saved", a.out)
