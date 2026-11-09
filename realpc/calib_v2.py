"""Rig calibration v2: joint robust least squares over the top / side camera poses and the right-arm base, on
STATIONARY-robot frames only (both arms still for >= `still` ticks before and after, so a camera time offset cannot
leak into the geometry).

Parameters: x = [x_top(6), x_side(6), x_rb(6)], T_wc' = T_wc0 @ se3(x_cam), T_lb_rb' = T_lb_rb0 @ se3(x_rb).
Residual terms (re-associated every outer round):
  edge   FK mesh silhouette outline points (in the arm's base frame) -> projected -> distance transform of Canny edges
  depth  measured arm points (camera depth, robot-near, outlier-gated) -> nearest FK surface point (base frame), 3D
  cross  side static points -> top static surface (point-to-plane), robot / arm regions removed
  plane  table points of each camera -> the table plane (n.x + d = 0)
  prior  weak prior towards the recorder calibration (sigma 2 deg / 2 cm)
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from .cam_calib import bilinear, edge_dt, se3


def stationary_frames(low, still=10, thr=0.004, min_sep=0.25):
    """ticks t with max joint speed (both arms, joints 1-6) < thr rad/tick for [t-still, t+still], spread in q."""
    q = np.concatenate([low["joint_pos_left"][:, :6], low["joint_pos_right"][:, :6]], 1)
    v = np.r_[0, np.abs(np.diff(q, axis=0)).max(1)]
    ok = np.array([v[max(0, t - still): t + still + 1].max() < thr for t in range(len(v))])
    ok[: still] = False; ok[-still:] = False
    cand = np.flatnonzero(ok)
    picked = []
    for t in cand:
        if all(np.linalg.norm(q[t] - q[p]) > min_sep for p in picked):
            picked.append(int(t))
    return picked, cand


def contour_base_points(mask, depth, K, T_wc, T_wb, border=4, step=3):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return np.zeros((0, 3))
    pts = np.concatenate([c[:, 0] for c in cs])[::step]
    H, W = mask.shape
    pts = pts[(pts[:, 0] > border) & (pts[:, 0] < W - border) & (pts[:, 1] > border) & (pts[:, 1] < H - border)]
    z = depth[pts[:, 1], pts[:, 0]]
    ok = np.isfinite(z) & (z > 0.05)
    pts, z = pts[ok], z[ok]
    Pc = np.stack([(pts[:, 0] - K[0, 2]) / K[0, 0] * z, (pts[:, 1] - K[1, 2]) / K[1, 1] * z, z], 1)
    Pw = Pc @ T_wc[:3, :3].T + T_wc[:3, 3]
    Ti = np.linalg.inv(T_wb)
    return Pw @ Ti[:3, :3].T + Ti[:3, 3]


def backproject(D, K, stride=2, zmin=0.28, zmax=1.8):
    Ds = D[::stride, ::stride]
    v, u = np.nonzero((Ds > zmin) & (Ds < zmax))
    z = Ds[v, u]; u = u * stride; v = v * stride
    return np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], 1)


class RigCalibrator:
    def __init__(self, arms, proj, cams, T_wc0, T_rb0, table_n, table_d, w=None):
        self.arms, self.proj, self.cams = arms, proj, cams
        self.T_wc0 = {c: np.asarray(T_wc0[c], float) for c in cams}
        self.T_rb0 = np.asarray(T_rb0, float)
        self.n, self.d = np.asarray(table_n, float), float(table_d)
        self.w = dict(edge=1.0, depth=1.0 / 0.005, cross=1.0 / 0.005, plane=1.0 / 0.005, prior_rad=np.radians(2.0), prior_m=0.02)
        if w:
            self.w.update(w)
        Ld, Bd = arms.dense_local()
        self.Ld, self.Bd = Ld, Bd

    def poses(self, x):
        Twc = {c: self.T_wc0[c] @ se3(x[6 * i: 6 * i + 6]) for i, c in enumerate(self.cams)}
        Trb = self.T_rb0 @ se3(x[12:18])
        return Twc, {"left": np.eye(4), "right": Trb}

    def associate(self, x, frames, statics):
        """frames: list of dict(cam, img, depth, K, q); statics: list of dict(top=(D,K), side=(D,K))."""
        Twc, Tb = self.poses(x)
        A = {"edge": [], "depth": [], "cross": [], "plane": []}
        for f in frames:
            c = f["cam"]; K = f["K"]; H, W = f["img"].shape[:2]
            if "dt" not in f:
                f["dt"] = edge_dt(f["img"])
            r = self.proj.render({arm: self.arms.qmap(f["q"][arm]) for arm in Tb}, Tb, Twc[c], K, H, W)
            LP = {arm: self.arms.robot.fk(self.arms.qmap(f["q"][arm])) for arm in Tb}
            for arm, (m, dep) in r.items():
                if m.sum() < 300:
                    continue
                Xb = contour_base_points(m, dep, K, Twc[c], Tb[arm])
                if len(Xb):
                    A["edge"].append((c, arm, Xb, f["dt"], K))
            # depth: measured arm points near the FK surface -> nearest surface point (base frame of its arm)
            P = backproject(f["depth"], K)
            Pw = P @ Twc[c][:3, :3].T + Twc[c][:3, 3]
            for ai, arm in enumerate(Tb):
                sel = np.array([b for b, (a_, l_) in enumerate(self.arms.bodies) if a_ == arm])
                mb = np.isin(self.Bd, sel)
                LPa = np.stack([LP[arm][self.arms.bodies[b][1]] for b in self.Bd[mb]])
                Sb = np.einsum("nij,nj->ni", LPa[:, :3, :3], self.Ld[mb]) + LPa[:, :3, 3]       # base frame
                Sw = Sb @ Tb[arm][:3, :3].T + Tb[arm][:3, 3]
                d, j = cKDTree(Sw).query(Pw, distance_upper_bound=0.03)
                ok = np.isfinite(d)
                if ok.sum() < 50:
                    continue
                idx = np.flatnonzero(ok)
                if len(idx) > 1500:
                    idx = np.random.default_rng(0).choice(idx, 1500, replace=False)
                A["depth"].append((c, arm, P[idx], Sb[j[idx]]))
        for s in statics:
            Pt = backproject(*s["top"]); Ps = backproject(*s["side"])
            Ptw = Pt @ Twc["top"][:3, :3].T + Twc["top"][:3, 3]
            Psw = Ps @ Twc["side"][:3, :3].T + Twc["side"][:3, 3]
            R = s["robot_w"]
            Ptw_k = ~np.isfinite(cKDTree(R).query(Ptw, distance_upper_bound=0.06)[0])
            Psw_k = ~np.isfinite(cKDTree(R).query(Psw, distance_upper_bound=0.06)[0])
            Pt, Ptw = Pt[Ptw_k], Ptw[Ptw_k]; Ps, Psw = Ps[Psw_k], Psw[Psw_k]
            # top surface normals (PCA of 12 neighbours)
            tr = cKDTree(Ptw)
            d, j = tr.query(Psw, distance_upper_bound=0.02)
            ok = np.isfinite(d)
            idx = np.flatnonzero(ok)
            if len(idx) > 4000:
                idx = np.random.default_rng(1).choice(idx, 4000, replace=False)
            _, nb = tr.query(Ptw[j[idx]], k=12)
            Q = Ptw[nb] - Ptw[nb].mean(1, keepdims=True)
            _, _, Vt = np.linalg.svd(Q)
            nrm = Vt[:, -1]
            A["cross"].append((Ps[idx], Pt[j[idx]], nrm))
            for cam, Pc, Pw in (("top", Pt, Ptw), ("side", Ps, Psw)):
                h = Pw @ self.n + self.d
                m = np.abs(h) < 0.01
                if m.sum() > 50:
                    k = np.flatnonzero(m)
                    k = np.random.default_rng(2).choice(k, min(3000, len(k)), replace=False)
                    A["plane"].append((cam, Pc[k]))
        return A

    def residuals(self, x, A):
        Twc, Tb = self.poses(x)
        out = []
        for c, arm, Xb, dt, K in A["edge"]:
            M = np.linalg.inv(Twc[c]) @ Tb[arm]
            pc = Xb @ M[:3, :3].T + M[:3, 3]
            uv = (pc @ K.T)[:, :2] / np.maximum(pc[:, 2:], 1e-3)
            out.append(self.w["edge"] * bilinear(dt, uv[:, 0], uv[:, 1]) / np.sqrt(max(1, len(Xb) / 200)))
        for c, arm, Pc, Sb in A["depth"]:
            Pw = Pc @ Twc[c][:3, :3].T + Twc[c][:3, 3]
            Sw = Sb @ Tb[arm][:3, :3].T + Tb[arm][:3, 3]
            out.append(self.w["depth"] * (Pw - Sw).ravel() / np.sqrt(max(1, len(Pc) / 200)))
        for Ps, Pt, nrm in A["cross"]:
            Psw = Ps @ Twc["side"][:3, :3].T + Twc["side"][:3, 3]
            Ptw = Pt @ Twc["top"][:3, :3].T + Twc["top"][:3, 3]
            out.append(self.w["cross"] * ((Psw - Ptw) * nrm).sum(1) / np.sqrt(max(1, len(Ps) / 400)))
        for cam, Pc in A["plane"]:
            Pw = Pc @ Twc[cam][:3, :3].T + Twc[cam][:3, 3]
            out.append(self.w["plane"] * (Pw @ self.n + self.d) / np.sqrt(max(1, len(Pc) / 400)))
        pr = np.concatenate([np.r_[x[i: i + 3] / self.w["prior_rad"], x[i + 3: i + 6] / self.w["prior_m"]] for i in (0, 6, 12)])
        out.append(pr * 3.0)
        return np.concatenate(out)

    def solve(self, frames, statics, rounds=4, x0=None):
        x = np.zeros(18) if x0 is None else x0
        log = []
        for rd in range(rounds):
            A = self.associate(x, frames, statics)
            r0 = self.residuals(x, A)
            sol = least_squares(self.residuals, x, args=(A,), loss="soft_l1", f_scale=2.0, max_nfev=80)
            x = sol.x
            log.append({"round": rd, "cost_before": float(np.sum(np.abs(r0))), "cost_after": float(np.sum(np.abs(self.residuals(x, A)))),
                        "n_edge": len(A["edge"]), "n_depth": len(A["depth"]), "n_cross": len(A["cross"]), "n_plane": len(A["plane"])})
        return x, log
