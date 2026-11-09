"""v3: causal tracking from 2D point tracks lifted to 3D.

Per frame (current + past data only):
  1. depth: the top camera's raw depth refined with the colour image (harness stage-1 solve, realpc.depth_refine)
  2. 2D correspondence: CoTracker3 ONLINE (sliding window, 8-tick latency) on the top RGB stream. A tracker instance
     is started every `reseed` steps with queries on the pixels of movable objects (dense) and lives `life` steps,
     so newly visible surfaces get tracks and every step is covered by >= 1 instance started <= `life` steps earlier
  3. lift: track pixel -> refined depth at that tick -> world point; tracks that are not visible, have no depth, sit on
     a depth edge, or jump inconsistently are rejected
  4. objects (realpc.tracker, motion="tracks"): segmentation, grasp state, births / deaths as before, but each object's
     ONE SE(3) pose per step = robust (RANSAC + Kabsch) fit of its tracks between the instance start and now, composed
     with the object pose at the instance start (low drift); held objects fall back to G(t) Y when their tracks are
     hidden; free objects move only on contact (robot or a moving neighbour). Cloth (allow_deform): its points move with
     the 3D displacement of the nearest visible lifted tracks (frame to frame).
  5. robot: FK.
Batch conversion runs the instances one after another over the same frames (identical results to running them
concurrently online); a live system keeps one CoTracker predictor per alive instance.
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch
from scipy.spatial import cKDTree

from .pipeline import pack
from .se3 import ransac_kabsch
from .tracker import DEFORM, RIGID, STATIC, Tracker, TrackerConfig


def load_cotracker(cache_dir):
    os.environ.setdefault("TORCH_HOME", cache_dir)
    local = os.path.join(cache_dir, "hub", "facebookresearch_co-tracker_main")
    if os.path.isdir(local):                         # downloaded once; no network afterwards
        m = torch.hub.load(local, "cotracker3_online", source="local")
    else:
        m = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")
    return m.cuda().eval() if torch.cuda.is_available() else m.eval()


@torch.no_grad()
def run_instance(model, frames, queries_xy):
    """frames: list of HxWx3 uint8 (ticks from the instance start); queries at frame 0 (N,2) pixels.
    -> tracks (T,N,2), vis (T,N). Online: frame t's estimate uses frames up to t + model.step."""
    T = len(frames)
    dev = next(model.parameters()).device
    q = torch.cat([torch.zeros(len(queries_xy), 1), torch.as_tensor(queries_xy, dtype=torch.float32)], 1)[None].to(dev)
    vid = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)[None].float()
    step = model.step
    pad = (-(T - step * 2)) % step if T > step * 2 else step * 2 - T
    if pad:
        vid = torch.cat([vid, vid[:, -1:].repeat(1, pad, 1, 1, 1)], 1)
    model(video_chunk=vid[:, : step * 2].to(dev), is_first_step=True, queries=q)
    tr = vis = None
    for ind in range(0, vid.shape[1] - step, step):
        tr, vis = model(video_chunk=vid[:, ind: ind + step * 2].to(dev))
    return tr[0, :T].cpu().numpy(), vis[0, :T].cpu().numpy()


def lift(uv, D, K, T_wc, edge_tol=0.01):
    """pixels (N,2) + depth image (m) -> world points (N,3), valid (N,) (no depth / depth edge -> invalid)."""
    H, W = D.shape
    u = np.clip(np.round(uv[:, 0]).astype(int), 1, W - 2)
    v = np.clip(np.round(uv[:, 1]).astype(int), 1, H - 2)
    inb = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    z = D[v, u]
    nb = np.stack([D[v + dy, u + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)], 1)
    ok = inb & (z > 0.25) & (np.abs(np.median(nb, 1) - z) < edge_tol) & (nb.min(1) > 0)
    x = (uv[:, 0] - K[0, 2]) / K[0, 0] * z
    y = (uv[:, 1] - K[1, 2]) / K[1, 1] * z
    P = np.stack([x, y, z], 1) @ T_wc[:3, :3].T + T_wc[:3, 3]
    return P, ok


def kabsch(A, B):
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = cb - R @ ca
    return T


def ransac_se3(A, B, thr=0.01, iters=64, rng=None):
    """robust rigid A -> B: RANSAC on 3-point samples, refit on inliers. -> T, inlier mask."""
    rng = rng or np.random.default_rng(0)
    n = len(A)
    best = None
    for _ in range(iters):
        i = rng.choice(n, 3, replace=False)
        T = kabsch(A[i], B[i])
        r = np.linalg.norm(A @ T[:3, :3].T + T[:3, 3] - B, axis=1)
        inl = r < thr
        if best is None or inl.sum() > best[1].sum():
            best = (T, inl)
    T, inl = best
    if inl.sum() >= 3:
        T = kabsch(A[inl], B[inl])
        inl = np.linalg.norm(A @ T[:3, :3].T + T[:3, 3] - B, axis=1) < thr
    return T, inl


class V3Converter:
    def __init__(self, ep, arms, cfg: TrackerConfig, model, refine_fn, reseed=32, life=96, n_queries=3000, min_tracks=8,
                 min_inlier_frac=0.6, robot_reject=0.03, inlier_thr=0.005, verbose=True):
        self.ep, self.arms, self.cfg, self.model, self.refine = ep, arms, cfg, model, refine_fn
        self.reseed, self.life, self.nq, self.min_tracks, self.min_frac = reseed, life, n_queries, min_tracks, min_inlier_frac
        self.verbose, self.robot_reject, self.inlier_thr = verbose, robot_reject, inlier_thr
        self.K = ep.intrinsics("top")
        import cv2
        self._cap = cv2.VideoCapture(ep.video("top"))
        self._cache = {}

    def rgb(self, tick):
        import cv2
        if tick not in self._cache:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, tick)
            ok, im = self._cap.read()
            self._cache[tick] = im[..., ::-1].copy()
            if len(self._cache) > 400:
                self._cache.pop(next(iter(self._cache)))
        return self._cache[tick]

    def rgb_range(self, t0, t1):
        import cv2
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, t0)
        out = []
        for _ in range(t0, t1):
            ok, im = self._cap.read()
            out.append(im[..., ::-1].copy() if ok else out[-1])
        return out

    def depth(self, tick):
        if tick not in self._dcache:
            self._dcache[tick] = self.refine(self.rgb(tick), tick)
            if len(self._dcache) > 200:
                self._dcache.pop(next(iter(self._dcache)))
        return self._dcache[tick]

    def _queries(self, trk, tick):
        """pixels on movable objects at this tick: (N,2) pixels, (N,) object id."""
        D = self.depth(tick)
        T_wc = self.ep.low["cam_pose_top"][tick]
        H, W = D.shape
        vv, uu = np.mgrid[2:H - 2:3, 2:W - 2:3]
        uv = np.stack([uu.ravel(), vv.ravel()], 1).astype(np.float64)
        P, ok = lift(uv, D, self.K, T_wc)
        objs = [o for o in trk.objs if o.kind in (RIGID, DEFORM)]
        if not objs or not ok.any():
            return np.zeros((0, 2)), np.zeros(0, int)
        allP = np.concatenate([o.positions() for o in objs])
        own = np.concatenate([np.full(len(o.gid), o.oid) for o in objs])
        d, j = cKDTree(allP).query(P[ok], distance_upper_bound=0.012)
        m = np.isfinite(d)
        uvs, oids = uv[ok][m], own[j[m]]
        if len(uvs) > self.nq:                       # balanced: every object keeps a share
            rng = np.random.default_rng(tick)
            keep = []
            for o in np.unique(oids):
                ii = np.flatnonzero(oids == o)
                keep.append(rng.choice(ii, min(len(ii), max(40, int(self.nq * len(ii) / len(uvs)))), replace=False))
            keep = np.concatenate(keep)
            uvs, oids = uvs[keep], oids[keep]
        return uvs, oids

    def convert(self, start=0, stop=None):
        ep, cfg = self.ep, self.cfg
        self._dcache = {}
        stop = len(ep) if stop is None else min(stop, len(ep))
        idx = list(range(start, stop))
        ticks = [ep.ticks[i] for i in idx]
        trk = Tracker(cfg, self.arms, *ep.table_plane())
        insts = []                                   # dict(k0, tick0, oids, X0, tracks, vis, k1)
        outs, Thist, t0 = [], [], time.time()
        prev_lift = None
        for k, i in enumerate(idx):
            f = ep.frame(i)
            if k > 0:
                # start / extend instances (batch: each instance is run once over its whole life when it starts)
                ext_T, flow_A, flow_B = {}, [], []
                Dk = self.depth(ticks[k]); Twc = ep.low["cam_pose_top"][ticks[k]]
                LP = self.arms.link_poses(f.q)
                Ld, Bd = self.arms.dense_local()
                rtree = cKDTree(np.einsum("nij,nj->ni", LP[Bd, :3, :3], Ld) + LP[Bd, :3, 3])
                best = {}
                cur_lift = {}
                for a, ins in enumerate(insts):
                    if not (ins["k0"] < k <= ins["k1"]):
                        continue
                    r = ticks[k] - ins["tick0"]
                    uv, vis = ins["tracks"][r], ins["vis"][r] > 0.5
                    P, ok = lift(uv, Dk, self.K, Twc)
                    ok &= vis & ins["ok0"]
                    ok &= ~np.isfinite(rtree.query(P, distance_upper_bound=self.robot_reject)[0])   # slid onto the arm
                    cur_lift[a] = (P, ok)
                    for o in np.unique(ins["oids"]):
                        m = ok & (ins["oids"] == o)
                        n_vis = int(((ins["vis"][r] > 0.5) & (ins["oids"] == o)).sum())
                        if m.sum() < self.min_tracks or m.sum() < 0.5 * n_vis:   # most visible tracks must lift cleanly
                            continue
                        Tab, inl, _ = ransac_kabsch(ins["X0"][m], P[m], thr=self.inlier_thr, huber=self.inlier_thr)
                        if inl.sum() >= max(self.min_tracks, self.min_frac * m.sum()) and o in Thist[ins["k0"]]:
                            score = int(inl.sum())
                            if o not in best or score > best[o][0]:
                                best[o] = (score, Tab @ Thist[ins["k0"]][o])
                    if prev_lift is not None and a in prev_lift:          # cloth flow, frame to frame
                        Pp, okp = prev_lift[a]
                        mm = ok & okp & np.isin(ins["oids"], [o.oid for o in trk.objs if o.kind == DEFORM])
                        if mm.any():
                            jump = np.linalg.norm(P[mm] - Pp[mm], axis=1) < 0.05
                            flow_A.append(Pp[mm][jump]); flow_B.append(P[mm][jump])
                prev_lift = cur_lift
                trk.ext_T = {o: T for o, (_, T) in best.items()}
                trk.ext_flow = (np.concatenate(flow_A), np.concatenate(flow_B)) if flow_A else None
            out = trk.step(f)
            if k == 0 and cfg.allow_deform:
                for o in trk.objs:
                    if o.kind == RIGID and len(o.local) >= cfg.deform_min_pts:
                        trk._to_deform(o)
            outs.append(out)
            Thist.append({o.oid: (o.T.copy() if o.kind != DEFORM else np.eye(4)) for o in trk.objs})
            if k % self.reseed == 0 and k < len(idx) - 2:
                k1 = min(len(idx) - 1, k + self.life)
                uvs, oids = self._queries(trk, ticks[k])
                if len(uvs) >= self.min_tracks:
                    frames = self.rgb_range(ticks[k], ticks[k1] + 1)
                    tracks, vis = run_instance(self.model, frames, uvs)
                    X0, ok0 = lift(uvs, self.depth(ticks[k]), self.K, ep.low["cam_pose_top"][ticks[k]])
                    insts.append(dict(k0=k, k1=k1, tick0=ticks[k], oids=oids, X0=X0, ok0=ok0, tracks=tracks, vis=vis))
            if self.verbose and (k % 50 == 0 or k == len(idx) - 1):
                print(f"  v3 step {k}/{len(idx)} points {len(out.pos)} objects {len(trk.objs)} instances {len(insts)} "
                      f"({time.time() - t0:.0f} s)", flush=True)
        tr = pack(outs, ep, idx)
        tr.variant = "v3_causal"
        return tr
