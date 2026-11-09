"""Track-based robot<->camera calibration: bundle adjustment of image-seeded 2D tracks through the arm FK.

Each moving 2D track i is explained as a point X_i fixed on one link l_i of one arm a_i:
    u_i(t) = pi_c( T_cam<-base[c, a_i] . T_base<-link(q_a(t - L_c(t))) . X_i )
Unknowns: per camera and arm the pose T_cam<-base (left and right separately), a per-(episode, camera) piecewise-linear
lag L_c(t) (ticks), optional per-joint zero offsets, and X_i / (a_i, l_i) per track.
  1. association: for every (arm, link) hypothesis, X_i is triangulated linearly (DLT) from all frames of the track;
     the hypothesis with the smallest reprojection RMS wins (tracks on objects / background match nothing -> dropped)
  2. lag scan: per track segment, the lag that explains the most tracks (robust consensus), smoothed
  3. joint robust refinement (Adam, Huber) of all calibration parameters + X_i + lag knots, then re-association.
No masks and no FK-based seeding are needed; the arm kinematics fix the metric scale.
"""
from __future__ import annotations

import numpy as np
import torch

from .diffcalib import TorchArm, se3_exp, interp_q

DEV = "cuda"


class Seq:
    """One (episode, camera) track file + the episode's joint sequences."""

    def __init__(self, ep, cam, trk_npz, K):
        d = np.load(trk_npz, allow_pickle=True)
        self.ep, self.cam, self.key = ep, cam, f"{ep.task}/{ep.name}"
        self.starts = [int(s) for s in d["seg_start"]]
        self.tracks = [np.asarray(t, np.float64) for t in d["tracks"]]
        self.vis = [np.asarray(v, bool) for v in d["vis"]]
        self.depth0 = [np.asarray(x, np.float64) for x in d["depth0"]]
        self.T = int(d["T"])
        self.K = np.asarray(K, float)
        self.q = {a: torch.tensor(ep.low[f"joint_pos_{a}"], dtype=torch.float64, device=DEV) for a in ("left", "right")}


class HandEyeBA:
    def __init__(self, arm: TorchArm, T0: dict, cams=("top", "side"), rig=False):
        """T0[(cam, arm)] = initial 4x4 T_cam<-base (numpy)."""
        self.arm = arm
        self.cams = cams
        self.arms = ("left", "right")
        self.links = arm.links
        self.T0 = {k: torch.tensor(v, dtype=torch.float64, device=DEV) for k, v in T0.items()}
        self.xi = {k: torch.zeros(6, dtype=torch.float64, device=DEV, requires_grad=True) for k in T0}
        self.joff = {a: torch.zeros(6, dtype=torch.float64, device=DEV, requires_grad=True) for a in self.arms}
        # rig mode: the cameras form one rigid rig (recorder top<->side relative pose kept: their static scenes agree to
        # ~5 mm); each arm base gets ONE correction, seen identically by both cameras
        self.rig = rig
        if rig:
            self.T_rel = {c: self.T0[(c, "left")] @ torch.linalg.inv(self.T0[(cams[0], "left")]) for c in cams}
            for c in cams[1:]:
                for a in self.arms:
                    self.xi[(c, a)].requires_grad_(False)

    # ---------------------------------------------------------------- kinematics
    def T_cb(self, cam, a):
        if self.rig:
            c0 = self.cams[0]
            return self.T_rel[cam] @ (se3_exp(self.xi[(c0, a)]) @ self.T0[(c0, a)])
        return se3_exp(self.xi[(cam, a)]) @ self.T0[(cam, a)]

    def link_T(self, seq, a, ticks, use_off=True):
        """fractional ticks [n] -> [n, nlinks, 4, 4] base<-link (float64)."""
        q = interp_q(seq.q[a], ticks)
        T = self.arm.link_tf(q, self.joff[a] if use_off else None)
        return torch.stack([T[l] for l in self.links], 1)

    def proj_mats(self, seq, ticks, use_off=True):
        """-> P [2 arms, nlinks, n, 3, 4] (K . T_cam<-link)."""
        K = torch.tensor(seq.K, dtype=torch.float64, device=DEV)
        out = []
        for a in self.arms:
            Tl = self.link_T(seq, a, ticks, use_off)                  # [n,L,4,4]
            M = self.T_cb(seq.cam, a)[None, None] @ Tl                # [n,L,4,4]
            out.append((K @ M[..., :3, :]).permute(1, 0, 2, 3))       # [L,n,3,4]
        return torch.stack(out)

    # ---------------------------------------------------------------- association
    @torch.no_grad()
    def associate(self, seq, j, lag, max_rms=6.0):
        """segment j at constant lag -> dict(best_h, X (link frame), rms, second) for every track."""
        tr = torch.tensor(seq.tracks[j], dtype=torch.float64, device=DEV)        # [Lf,N,2]
        vi = torch.tensor(seq.vis[j], dtype=torch.float64, device=DEV)           # [Lf,N]
        Lf, N = vi.shape
        ticks = torch.arange(Lf, dtype=torch.float64, device=DEV) + seq.starts[j] - lag
        P = self.proj_mats(seq, ticks).reshape(-1, Lf, 3, 4)                     # [H,Lf,3,4]
        H = P.shape[0]
        u, v = tr[..., 0], tr[..., 1]                                            # [Lf,N]
        # DLT rows, normalised by 1/f so pixel and metric scales are balanced
        r1 = u[None, :, :, None] * P[:, :, None, 2, :] - P[:, :, None, 0, :]       # [H,Lf,N,4]
        r2 = v[None, :, :, None] * P[:, :, None, 2, :] - P[:, :, None, 1, :]
        w = vi[None, :, :, None]
        A = torch.einsum("hkni,hknj->hnij", r1 * w, r1) + torch.einsum("hkni,hknj->hnij", r2 * w, r2)
        evals, evecs = torch.linalg.eigh(A)
        X = evecs[..., 0]                                                        # [H,N,4]
        X = evecs[..., 0] / evecs[..., 0][..., 3:4]
        p = torch.einsum("hkij,hnj->hkni", P, X)                                 # [H,Lf,N,3]
        z = p[..., 2]
        uv = p[..., :2] / z.clamp(min=1e-6)[..., None]
        e = (uv - tr[None]).norm(dim=-1)
        e = torch.where(z > 0.05, e, torch.full_like(e, 1e3))
        rms = torch.sqrt(((e.clamp(max=100) ** 2) * vi[None]).sum(1) / vi.sum(0).clamp(min=1)[None])   # [H,N]
        # link-frame X must lie near the link (within 25 cm of the link origin)
        rms = torch.where(X[..., :3].norm(dim=-1) < 0.25, rms, torch.full_like(rms, 1e3))
        srt, idx = rms.sort(0)
        best = idx[0]
        return dict(h=best.cpu().numpy(), X=X[best, torch.arange(N)][:, :3].cpu().numpy(), rms=srt[0].cpu().numpy(),
                    second=srt[1].cpu().numpy(), all_rms=rms.cpu().numpy())

    def lag_scan(self, seq, j, lags, cap=8.0):
        sc = []
        for L in lags:
            r = self.associate(seq, j, float(L))
            sc.append(float(np.minimum(r["rms"], cap).mean()))
        return np.array(sc)

    # ---------------------------------------------------------------- joint refinement
    def build_obs(self, seqs, assoc, lags, segs_use, max_rms=5.0):
        """flatten inlier tracks into observation tensors. assoc[(key,cam,j)] from associate; lags[(key,cam)] knot array."""
        obs = []
        for s in seqs:
            for j in range(len(s.starts)):
                if (s.key, s.cam, j) not in assoc or not segs_use(s, j):
                    continue
                r = assoc[(s.key, s.cam, j)]
                ok = (r["rms"] < max_rms)
                if ok.sum() < 3:
                    continue
                Lf = s.tracks[j].shape[0]
                for i in np.flatnonzero(ok):
                    vk = np.flatnonzero(s.vis[j][:, i])
                    obs.append(dict(seq=s, j=j, h=int(r["h"][i]), X=r["X"][i], k=vk, uv=s.tracks[j][vk, i]))
        return obs

    def refine(self, seqs, obs, lag_knots, iters=1500, lr=1e-3, fit_offsets=False, off_sigma_deg=2.0,
               fit_lag=True, huber=3.0, log_every=250, verbose=True, depth_w=0.0, depth_sigma=0.01, extra_loss=None):
        """Adam over xi, X_i, lag knots (+ joint offsets). lag_knots[(key,cam)] = (knot_ticks np, values np)."""
        nl = len(self.links)
        X = torch.tensor(np.stack([o["X"] for o in obs]), dtype=torch.float64, device=DEV, requires_grad=True)
        knots = {k: (torch.tensor(t, dtype=torch.float64, device=DEV), torch.tensor(v, dtype=torch.float64, device=DEV, requires_grad=fit_lag))
                 for k, (t, v) in lag_knots.items()}
        params = [{"params": [v for v in self.xi.values() if v.requires_grad], "lr": lr}, {"params": [X], "lr": lr * 2}]
        if fit_lag:
            params.append({"params": [v for _, v in knots.values()], "lr": 0.05})
        if fit_offsets:
            params.append({"params": list(self.joff.values()), "lr": lr})
        opt = torch.optim.Adam(params)
        # group observations by sequence for batched FK
        groups = {}
        for oi, o in enumerate(obs):
            groups.setdefault((o["seq"].key, o["seq"].cam), []).append(oi)
        seq_of = {(s.key, s.cam): s for s in seqs}
        pre = {}
        for g, ois in groups.items():
            s = seq_of[g]
            t_abs, uv, ho, xo, dz = [], [], [], [], []
            for oi in ois:
                o = obs[oi]
                t_abs.append(s.starts[o["j"]] + o["k"]); uv.append(o["uv"])
                ho.append(np.full(len(o["k"]), o["h"])); xo.append(np.full(len(o["k"]), oi))
                zt = np.full(len(o["k"]), np.nan)
                d0 = o.get("d0", 0.0)
                if 0.15 < d0 < 2.5 and len(o["k"]) and o["k"][0] == 0:
                    zt[0] = d0
                dz.append(zt)
            pre[g] = dict(t=torch.tensor(np.concatenate(t_abs), dtype=torch.float64, device=DEV),
                          uv=torch.tensor(np.concatenate(uv), dtype=torch.float64, device=DEV),
                          h=torch.tensor(np.concatenate(ho), device=DEV), xo=torch.tensor(np.concatenate(xo), device=DEV),
                          zt=torch.tensor(np.concatenate(dz), dtype=torch.float64, device=DEV))
            tu, tinv = torch.unique(pre[g]["t"], return_inverse=True)
            pre[g]["tu"], pre[g]["tinv"] = tu, tinv
            pre[g]["arm"] = pre[g]["h"] // nl
            pre[g]["link"] = pre[g]["h"] % nl
        hist = []
        for it in range(iters):
            opt.zero_grad()
            tot, cnt = 0.0, 0
            for g, d in pre.items():
                s = seq_of[g]
                kt, kv = knots[g]
                # FK once per unique observed tick of this sequence, gathered per observation
                lag_u = interp1(d["tu"], kt, kv)
                ticks_u = d["tu"] - lag_u
                K = torch.tensor(s.K, dtype=torch.float64, device=DEV)
                uvp = torch.zeros_like(d["uv"]); zp = torch.zeros_like(d["uv"][:, 0])
                for ai, a in enumerate(self.arms):
                    m = d["arm"] == ai
                    if m.sum() == 0:
                        continue
                    q = interp_q(s.q[a], ticks_u)
                    off = self.joff[a] if fit_offsets else None
                    T = self.arm.link_tf(q, off)
                    Ts = torch.stack([T[l] for l in self.links], 1)                   # [U,L,4,4]
                    M = self.T_cb(s.cam, a)[None, None] @ Ts                          # [U,L,4,4]
                    Mo = M[d["tinv"][m], d["link"][m]]                                # [n,4,4]
                    Xi = X[d["xo"][m]]
                    pc = torch.einsum("nij,nj->ni", Mo[:, :3, :3], Xi) + Mo[:, :3, 3]
                    pp = pc @ K.T
                    uvp[m] = pp[:, :2] / pp[:, 2:].clamp(min=1e-3)
                    zp[m] = pc[:, 2]
                e = (uvp - d["uv"]).norm(dim=-1)
                l = torch.where(e < huber, 0.5 * e ** 2, huber * (e - 0.5 * huber))
                tot = tot + l.sum(); cnt += len(e)
                if depth_w > 0:
                    mz = torch.isfinite(d["zt"])
                    if mz.any():
                        ez = (zp[mz] - d["zt"][mz]).abs() / depth_sigma
                        tot = tot + depth_w * torch.where(ez < huber, 0.5 * ez ** 2, huber * (ez - 0.5 * huber)).sum()
                if fit_lag:
                    tot = tot + 50.0 * ((kv[2:] - 2 * kv[1:-1] + kv[:-2]) ** 2).sum() + 5.0 * ((kv[1:] - kv[:-1]) ** 2).sum()
            loss = tot / max(cnt, 1)
            if extra_loss is not None:
                loss = loss + extra_loss(it)
            if fit_offsets:
                sig = np.radians(off_sigma_deg)
                loss = loss + sum(((o / sig) ** 2).sum() for o in self.joff.values()) / max(cnt, 1) * 100
            loss.backward()
            opt.step()
            if verbose and (it % log_every == 0 or it == iters - 1):
                print(f"  it {it} loss {loss.item():.3f}", flush=True)
            hist.append(loss.item())
        new_knots = {k: (t.cpu().numpy(), v.detach().cpu().numpy()) for k, (t, v) in knots.items()}
        return X.detach().cpu().numpy(), new_knots, hist

    @torch.no_grad()
    def residuals(self, seqs, obs, X, lag_knots):
        """per-observation reprojection error (px) with the current parameters -> list of arrays per obs."""
        nl = len(self.links)
        out = []
        for oi, o in enumerate(obs):
            s = o["seq"]
            kt, kv = lag_knots[(s.key, s.cam)]
            t = torch.tensor(s.starts[o["j"]] + o["k"], dtype=torch.float64, device=DEV)
            lag = interp1(t, torch.tensor(kt, device=DEV, dtype=torch.float64), torch.tensor(kv, device=DEV, dtype=torch.float64))
            a = self.arms[o["h"] // nl]; l = self.links[o["h"] % nl]
            q = interp_q(s.q[a], t - lag)
            T = self.arm.link_tf(q, self.joff[a])[l]
            M = self.T_cb(s.cam, a)[None] @ T
            Xi = torch.tensor(X[oi], dtype=torch.float64, device=DEV)
            pc = M[:, :3, :3] @ Xi + M[:, :3, 3]
            K = torch.tensor(s.K, dtype=torch.float64, device=DEV)
            pp = pc @ K.T
            uv = pp[:, :2] / pp[:, 2:].clamp(min=1e-3)
            out.append((uv - torch.tensor(o["uv"], device=DEV)).norm(dim=-1).cpu().numpy())
        return out


def interp1(x, xp, fp):
    """piecewise-linear interpolation (torch, differentiable in fp), clamped at the ends."""
    if len(xp) == 1:
        return fp[0].expand_as(x)
    i = torch.searchsorted(xp, x.contiguous()).clamp(1, len(xp) - 1)
    x0, x1 = xp[i - 1], xp[i]
    w = ((x - x0) / (x1 - x0)).clamp(0, 1)
    return fp[i - 1] * (1 - w) + fp[i] * w
