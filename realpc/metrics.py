"""Tracking-quality metrics against the measurements.

Per step and per group:
  model->measured   distance from each tracked point to the nearest measured point. Occluded points have no measured
                    neighbour, so we report the median over all alive points AND the "supported" fraction (< 1 cm).
  measured->model   distance from each measured point (fused cloud) to the nearest tracked point; > 2 cm = unexplained.
                    Each measured point is attributed to the group of its nearest tracked point.
  robot vs depth    the fused cloud has the arms cut out by the recorder, so robot points are checked against the raw
                    top + side depth: measured points within 3 cm of the FK model that the scene cloud does not explain
                    -> distance to the model surface (gated, so it is a precision measure, not coverage).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

GROUPS = {0: "gripper", 1: "robot", 2: "table", 3: "static objects", 5: "moving objects", 4: "cloth"}


def point_groups(tracks, k0=0, k1=None, moved_thresh=0.01):
    """group per point for a step range: parts split into static / moving by displacement within [k0, k1)."""
    k1 = tracks.pos.shape[1] if k1 is None else k1
    P = tracks.pos[:, k0:k1]
    ref = P[:, 0]
    disp = np.nanmax(np.linalg.norm(P - ref[:, None], axis=-1), axis=1)
    g = tracks.group.astype(int).copy()
    g[(g == 3) & (np.nan_to_num(disp) > moved_thresh)] = 5
    return g


def model_to_measured(tracks, g, k0=0, k1=None):
    k1 = tracks.pos.shape[1] if k1 is None else k1
    out = {}
    md = tracks.mdist[:, k0:k1].astype(np.float32)
    for gi, nm in GROUPS.items():
        m = g == gi
        if not m.any():
            continue
        x = md[m]
        x = x[np.isfinite(x)]
        if len(x) == 0:
            continue
        out[nm] = {"n_points": int(m.sum()), "median_mm": float(np.median(x) * 1000), "supported_lt1cm": float((x < 0.01).mean()),
                   "supported_median_mm": float(np.median(x[x < 0.01]) * 1000) if (x < 0.01).any() else float("nan")}
    return out


def measured_to_model(ep, arms, tracks, g, steps):
    """measured (fused, robot removed) -> nearest tracked scene point; > 2 cm = unexplained."""
    res = {nm: [] for nm in GROUPS.values()}
    unexpl = []
    for k in steps:
        M = measured_cloud(ep, arms, tracks, k)
        P = tracks.pos[:, k]
        ok = np.isfinite(P).all(1) & (g >= 2)
        d, j = cKDTree(P[ok]).query(M)
        gj = g[ok][j]
        unexpl.append(float((d > 0.02).mean()))
        for gi, nm in GROUPS.items():
            if gi >= 2:
                res[nm].append(d[(gj == gi) & (d <= 0.02)])
    out = {"unexplained_gt2cm_frac": float(np.mean(unexpl)), "unexplained_per_step": unexpl}
    for nm, v in res.items():
        if v and sum(len(x) for x in v):
            x = np.concatenate(v)
            out[nm] = {"median_mm": float(np.median(x) * 1000), "p90_mm": float(np.percentile(x, 90) * 1000)}
    return out


def robot_vs_depth(ep, arms, tracks, steps, depth_sources, gate=0.03, scene_tol=0.006):
    from .io import backproject
    res = {"left": [], "right": []}
    model_all = []
    for k in steps:
        f = ep.frame(int(tracks.frame_index[k]))
        D = np.concatenate([backproject(s.depth(f.tick), s.K, s.pose(f.tick), stride=2)[0] for s in depth_sources])
        model = tracks.pos[: tracks.n_robot, k]
        dm, jm = cKDTree(model).query(D, distance_upper_bound=gate)
        cand, jc = D[np.isfinite(dm)], jm[np.isfinite(dm)]
        if len(cand) == 0:
            continue
        ds = cKDTree(f.points).query(cand, distance_upper_bound=scene_tol)[0]
        keep = ~np.isfinite(ds)
        d = dm[np.isfinite(dm)][keep]
        arm = arms.arm_of[jc[keep]]
        for ai, nm in enumerate(("left", "right")):
            res[nm].append(d[arm == ai])
    out = {}
    for nm, v in res.items():
        x = np.concatenate(v) if v else np.zeros(0)
        if len(x):
            out[nm] = {"n": int(len(x)), "median_mm": float(np.median(x) * 1000), "p90_mm": float(np.percentile(x, 90) * 1000)}
    return out


def recompute_mdist(ep, arms, tracks, robot_margin=0.025, voxel=0.006, steps=None):
    """distance of every alive tracked point to the measured cloud (fused cloud, robot-FK points removed), per step.
    Used identically for every variant (causal / offline / cold start)."""
    from .tracker import voxel_down
    T = tracks.pos.shape[1]
    steps = range(T) if steps is None else steps
    md = np.full(tracks.mdist.shape, np.nan, np.float16)
    for k in steps:
        M = measured_cloud(ep, arms, tracks, k, robot_margin, voxel)
        P = tracks.pos[:, k]
        ok = np.isfinite(P).all(1)
        md[ok, k] = cKDTree(M).query(P[ok])[0]
    return md


def measured_cloud(ep, arms, tracks, k, robot_margin=0.025, voxel=0.006, wrist_margin=0.04):
    from .tracker import voxel_down
    f = ep.frame(int(tracks.frame_index[k]))
    M = voxel_down(f.points.astype(np.float64), voxel)
    dense, dbody = arms.dense_local()
    LP = arms.link_poses(f.q)
    R = np.einsum("nij,nj->ni", LP[dbody, :3, :3], dense) + LP[dbody, :3, 3]
    names = np.array([l for _, l in arms.bodies])
    wrist = np.isin(names[dbody], ["link5", "gripper"])
    dr = cKDTree(R[~wrist]).query(M, distance_upper_bound=robot_margin)[0]
    dw = cKDTree(R[wrist]).query(M, distance_upper_bound=wrist_margin)[0]
    return M[~np.isfinite(dr) & ~np.isfinite(dw)]
