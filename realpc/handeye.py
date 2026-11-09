"""Hand-eye (base -> camera) refinement from 2D tracks on the arm (one fixed correction per arm per camera).

The arm itself is always placed by FK; tracks on it are used ONLY here.
  1. CoTracker3 tracks seeded inside the FK-rendered arm silhouette (pixels with a depth reading), lifted with the raw
     depth -> world points B_it (recorder calibration).
  2. Each track belongs to one link (nearest FK surface point at its first frame). A tracked point is fixed on its link,
     so its link-frame position L_i is constant: alternate
       L_i <- robust mean_t  F_t^-1 C B_it      (F_t = link pose by FK), then snapped onto the link's mesh surface
       C   <- RANSAC + Huber Kabsch of {B_it} -> {F_t L_i}     (realpc.se3, same machinery as the SE(3) encoder)
  C maps points measured by that camera into the robot's world, i.e. the corrected camera pose is C @ T_wc.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .se3 import apply, ransac_kabsch, rot_angle


def solve_correction(obs, arms, arm, iters=8, snap=0.02, thr=0.01):
    """obs: list of (track_id, link_body, B (3,), link_pose (4x4)) for one arm and camera. -> C, log."""
    tid = np.array([o[0] for o in obs]); body = np.array([o[1] for o in obs])
    B = np.array([o[2] for o in obs]); F = np.array([o[3] for o in obs])
    Fi = np.linalg.inv(F)
    dense, dbody = arms.dense_local()
    trees = {b: cKDTree(dense[dbody == b]) for b in np.unique(body)}
    pts = {b: dense[dbody == b] for b in np.unique(body)}
    C = np.eye(4)
    log = []
    keep = np.ones(len(obs), bool)
    for it in range(iters):
        CB = apply(C, B)
        Lloc = np.einsum("nij,nj->ni", Fi[:, :3, :3], CB) + Fi[:, :3, 3]
        L = np.zeros_like(Lloc)
        for t in np.unique(tid):
            m = tid == t
            med = np.median(Lloc[m], 0)
            d, j = trees[body[m][0]].query(med)
            if d > snap:
                keep[m] = False
                continue
            keep[m] = True
            L[m] = pts[body[m][0]][j]
        tgt = np.einsum("nij,nj->ni", F[:, :3, :3], L) + F[:, :3, 3]
        k = keep
        if k.sum() < 20:
            break
        C, inl, r = ransac_kabsch(B[k], tgt[k], thr=thr, huber=thr)
        log.append({"iter": it, "n": int(k.sum()), "inliers": float(inl.mean()), "median_mm": float(np.median(r) * 1000),
                    "trans_mm": float(np.linalg.norm(C[:3, 3]) * 1000), "rot_deg": float(np.degrees(rot_angle(C[:3, :3])))})
    return C, log
