"""Hand-eye refinement of the arm bases against measured depth (rig calibration step, run once per rig / session).

The recorder's base->camera calibration has a few cm of error (its own fit reports 14-22 mm rms). We refine one rigid
correction per arm, C_arm, so that FK surface points C_arm @ T_world_base @ FK(q) lie on the measured depth surface.
Measured robot points = raw depth points of the top + side cameras that are not explained by the scene cloud (the
recorder's fused cloud has the arms cut out) and lie within `gate` of the current model. Robust point-to-point ICP over
many frames jointly; optionally a robot-clock delay (joint states interpolated at t + delay).

Adapted from the idea of the user's harness `s2_handeye.py` (O(t) = X E(t) Y with X refit + delay search), here with the
robot's own surface as the tracked object instead of a held object.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot


def se3_exp(x):
    T = np.eye(4)
    T[:3, :3] = Rot.from_rotvec(x[:3]).as_matrix()
    T[:3, 3] = x[3:]
    return T


def interp_q(low, arm, t, delay=0.0):
    """joint vector of `arm` at time t + delay from low_dim (linear; gripper included)."""
    tw = low["t_wall"]
    Q = low[f"joint_pos_{arm}"]
    tt = np.clip(t + delay, tw[0], tw[-1])
    return np.array([np.interp(tt, tw, Q[:, k]) for k in range(Q.shape[1])])


def robot_measurements(arms, frame, depth_pts, corr, gate=0.05, scene_tol=0.006):
    """measured points near the robot model (within gate) that the scene cloud does not explain."""
    model = arms.points(frame.q, poses=corrected_poses(arms, frame.q, corr))
    dm, _ = cKDTree(model).query(depth_pts, distance_upper_bound=gate)
    cand = depth_pts[np.isfinite(dm)]
    if len(cand) == 0:
        return cand
    ds, _ = cKDTree(frame.points).query(cand, distance_upper_bound=scene_tol)
    return cand[~np.isfinite(ds)]


def corrected_poses(arms, q, corr):
    P = arms.link_poses(q)
    out = P.copy()
    for ai, arm in enumerate(arms.T_base):
        m = np.array([a == arm for a, _ in arms.bodies])
        out[m] = corr[arm] @ P[m]
    return out


def fit_handeye(arms, frames, depth_pts_list, iters=8, gate0=0.04, gate1=0.015, f_scale=0.005, prior_m=0.03, prior_rad=0.05,
                n_sub=6000, seed=0):
    """frames: list of Frame; depth_pts_list: matching measured world points (raw depth, any cameras).
    Returns {arm: C (4x4, world-frame left correction)}, log. C = B D B^-1 with B the arm base pose and D a small
    SE(3) about the base (soft prior: prior_m metres / prior_rad radians per unit residual); robust ICP (soft-L1)."""
    from scipy.optimize import least_squares
    rng = np.random.default_rng(seed)
    corr = {a: np.eye(4) for a in arms.T_base}
    x = {a: np.zeros(6) for a in arms.T_base}
    log = []
    for it in range(iters):
        gate = gate0 + (gate1 - gate0) * it / max(1, iters - 1)
        for ai, arm in enumerate(arms.T_base):
            B0 = arms.T_base[arm]; B0i = np.linalg.inv(B0)
            A, Bm = [], []
            for f, D in zip(frames, depth_pts_list):
                P = arms.link_poses(f.q)
                Pm = arms.points(f.q, poses=P)
                meas = robot_measurements(arms, f, D, corr, gate=gate)
                if len(meas) < 30:
                    continue
                allc = arms.points(f.q, poses=corrected_poses(arms, f.q, corr))
                d, j = cKDTree(allc).query(meas)
                mine = arms.arm_of[j] == ai
                A.append(Pm[j[mine]]); Bm.append(meas[mine])
            if not A:
                continue
            A, Bm = np.concatenate(A), np.concatenate(Bm)
            if len(A) > n_sub:
                k = rng.choice(len(A), n_sub, replace=False); A, Bm = A[k], Bm[k]
            Al = A @ B0i[:3, :3].T + B0i[:3, 3]            # model points in the base frame
            Bl = Bm @ B0i[:3, :3].T + B0i[:3, 3]

            def res(v):
                D_ = se3_exp(v)
                r = (Al @ D_[:3, :3].T + D_[:3, 3] - Bl).ravel()
                return np.concatenate([r, np.sqrt(len(Al)) * f_scale * np.r_[v[:3] / prior_rad, v[3:] / prior_m]])
            sol = least_squares(res, x[arm], loss="soft_l1", f_scale=f_scale, max_nfev=100)
            x[arm] = sol.x
            corr[arm] = B0 @ se3_exp(sol.x) @ B0i
            r = np.linalg.norm((Al @ se3_exp(sol.x)[:3, :3].T + se3_exp(sol.x)[:3, 3]) - Bl, axis=1)
            log.append({"iter": it, "arm": arm, "gate": round(gate, 3), "n": int(len(Bl)), "median_mm": float(np.median(r) * 1000),
                        "trans_mm": float(np.linalg.norm(sol.x[3:]) * 1000), "rot_deg": float(np.degrees(np.linalg.norm(sol.x[:3])))})
    return corr, log
