"""Rig calibration object shared by conversion, inference, overlays and evaluation.

World = left arm base. Per camera c and arm a the calibration stores T_cam<-base_a (both arms separately, so an arm-base
offset seen differently by the two cameras stays visible). A per-(episode, camera) piecewise-linear lag says which robot
tick a camera frame shows: frame t of camera c shows the robot at q(t - lag_c(t)).
"""
from __future__ import annotations

import json

import numpy as np
import torch


def _interp_rows(Q, tq):
    T = len(Q)
    tq = float(np.clip(tq, 0, T - 1))
    i0 = int(np.floor(tq)); i1 = min(i0 + 1, T - 1); w = tq - i0
    return Q[i0] * (1 - w) + Q[i1] * w


def mean_pose(Ts):
    """chordal mean of rigid transforms."""
    R = sum(T[:3, :3] for T in Ts)
    U, _, Vt = np.linalg.svd(R)
    Rm = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    out = np.eye(4); out[:3, :3] = Rm; out[:3, 3] = np.mean([T[:3, 3] for T in Ts], 0)
    return out


class RigCalib:
    def __init__(self, T_cam_base: dict, lags: dict | None = None, joint_offsets: dict | None = None, name=""):
        self.Tcb = {k: np.asarray(v, float) for k, v in T_cam_base.items()}          # {(cam, arm): 4x4}
        self.lags = lags or {}                                                       # {"task/ep:cam": (ticks, lag)}
        self.joff = joint_offsets or {}
        self.name = name

    # ------------------------------------------------------------------ constructors
    @classmethod
    def load(cls, path):
        d = json.load(open(path))
        Tcb = {tuple(k.split("/")): np.asarray(v) for k, v in d["T_cam_base"].items()}
        lags = {k: (np.asarray(v["knot_ticks"], float), np.asarray(v["lag_ticks"], float)) for k, v in d.get("lags", {}).items()}
        joff = {k: np.asarray(v, float) for k, v in d.get("joint_offsets_rad", {}).items()}
        return cls(Tcb, lags, joff, name=path)

    @classmethod
    def recorder(cls, ep, cams=("top", "side")):
        Tcb = {}
        for c in cams:
            Ti = np.linalg.inv(np.asarray(ep.low[f"cam_pose_{c}"][0], float))
            Tcb[(c, "left")] = Ti
            Tcb[(c, "right")] = Ti @ np.asarray(ep.calib["T_lb_rb"], float)
        return cls(Tcb, name="recorder")

    # ------------------------------------------------------------------ queries
    def T_cam_base(self, cam, arm):
        return self.Tcb[(cam, arm)]

    def T_world_cam(self, cam):
        """camera pose in the world (= left base) frame."""
        return np.linalg.inv(self.Tcb[(cam, "left")])

    def T_world_rb(self, cam=None):
        """right base in the world frame, from one camera or the chordal mean over cameras."""
        cams = [cam] if cam else sorted({c for c, _ in self.Tcb})
        return mean_pose([np.linalg.inv(self.Tcb[(c, "left")]) @ self.Tcb[(c, "right")] for c in cams])

    def lag(self, ep, cam, t):
        k = f"{ep.task}/{ep.name}:{cam}"
        if k not in self.lags:
            return 0.0
        kt, kv = self.lags[k]
        return float(np.interp(t, kt, kv))

    def robot_tick(self, ep, cam, t):
        return t - self.lag(ep, cam, t)

    def q_at(self, ep, arm, tq):
        return _interp_rows(np.asarray(ep.low[f"joint_pos_{arm}"], float), tq)

    def joint_offsets(self, arm):
        o = self.joff.get(arm)
        if o is None or not np.any(o):
            return None
        return torch.tensor(o, dtype=torch.float32, device="cuda")
