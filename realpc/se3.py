"""Shared rigid-fit machinery (Kabsch / weighted Kabsch / RANSAC + Huber refit). Same math as the SE(3)-patch encoder's
per-token Kabsch fit (plwm/ae/se3patch.py)."""
from __future__ import annotations

import numpy as np


def kabsch(A, B, w=None):
    """weighted least-squares rigid T (4x4) with T A ~= B."""
    w = np.ones(len(A)) if w is None else np.asarray(w, float)
    ca, cb = (w[:, None] * A).sum(0) / w.sum(), (w[:, None] * B).sum(0) / w.sum()
    U, _, Vt = np.linalg.svd(((A - ca) * w[:, None]).T @ (B - cb))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = cb - R @ ca
    return T


def apply(T, P):
    return P @ T[:3, :3].T + T[:3, 3]


def rot_angle(R):
    return float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def ransac_kabsch(A, B, thr=0.005, iters=100, huber=0.005, refit_iters=3, rng=None):
    """RANSAC on 3-point minimal sets (inlier < thr), then Huber-weighted Kabsch refits on the inliers.
    -> T, inlier mask, residuals (m)."""
    rng = rng or np.random.default_rng(0)
    n = len(A)
    best_T, best_inl = np.eye(4), np.zeros(n, bool)
    for _ in range(iters):
        i = rng.choice(n, 3, replace=False)
        if np.linalg.matrix_rank(A[i] - A[i].mean(0), tol=1e-4) < 2:
            continue
        T = kabsch(A[i], B[i])
        inl = np.linalg.norm(apply(T, A) - B, axis=1) < thr
        if inl.sum() > best_inl.sum():
            best_T, best_inl = T, inl
    T, inl = best_T, best_inl
    if inl.sum() >= 3:
        for _ in range(refit_iters):
            r = np.linalg.norm(apply(T, A[inl]) - B[inl], axis=1)
            w = np.minimum(1.0, huber / np.maximum(r, 1e-9))
            T = kabsch(A[inl], B[inl], w)
        r = np.linalg.norm(apply(T, A) - B, axis=1)
        inl = r < thr
    else:
        r = np.linalg.norm(apply(T, A) - B, axis=1)
    return T, inl, r
