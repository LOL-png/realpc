"""Batch conversion = the streaming tracker run over a frame range (identical code path to inference).

convert(ep, arms, start, stop) -> Tracks: positions [N, T, 3] (NaN before a point's birth), group / body per point,
birth step, per-step distance to the measured cloud, and the frame times / ticks.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .io import attach_views
from .tracker import Tracker, TrackerConfig


@dataclass
class Tracks:
    pos: np.ndarray            # [N, T, 3] float32, NaN where not yet born
    group: np.ndarray          # [N] int8: 0 gripper, 1 robot link, 2 table, 3 rigid, 4 deformable (final kind)
    body: np.ndarray           # [N] int32
    birth: np.ndarray          # [N] int32 step of birth (0 = from the first step)
    mdist: np.ndarray          # [N, T] float16 distance to the measured cloud (m), NaN before birth
    times: np.ndarray          # [T]
    ticks: np.ndarray          # [T]
    n_robot: int
    frame_index: np.ndarray    # [T] index into the episode's fused frames
    held: np.ndarray           # [N, T] bool (point on a part held by a gripper)
    part_T: np.ndarray         # [P, T, 4, 4] float32 rigid object poses (NaN where the object does not exist / is not rigid)
    obj_res: np.ndarray        # [P, T, 2] float32 per object: median measured->object distance (m), n measured points
    variant: str = "causal"

    def save(self, path):
        np.savez_compressed(path, **{k: getattr(self, k) for k in self.__dataclass_fields__})

    @staticmethod
    def load(path):
        d = np.load(path)
        return Tracks(**{k: (d[k] if d[k].shape else d[k].item()) for k in d.files})


def offline_refine(tr: Tracks, smooth: int = 3) -> Tracks:
    """OFFLINE mode (needs the whole sequence; not usable at inference). Ideas from the user's harness:
    - back-fill: a point born at step b on part p existed earlier on that part (it was hidden or not yet seen):
      x(t) = T_p(t) T_p(b)^-1 x(b) for every earlier t where the part exists (shape from the whole episode,
      like the harness's object models / hidden-surface completion). Points of the table part: static, back-filled.
    - pose smoothing: each part's pose is averaged over +-`smooth` steps (like the harness's +-3 frame temporal fusion),
      except across grasp / release changes.
    Robot points are unchanged (FK)."""
    import copy
    from scipy.spatial.transform import Rotation as Rot
    out = copy.deepcopy(tr)
    out.variant = "offline"
    N, T, _ = tr.pos.shape
    body = tr.body.astype(int)
    nb_robot_bodies = int(body[: tr.n_robot].max()) + 1
    PT = tr.part_T.astype(np.float64)
    # smoothed poses
    S = PT.copy()
    if smooth > 0:
        for sid in range(len(PT)):
            ok = np.isfinite(PT[sid, :, 0, 0])
            if ok.sum() < 3:
                continue
            ks = np.flatnonzero(ok)
            R = Rot.from_matrix(PT[sid, ks, :3, :3]); t = PT[sid, ks, :3, 3]
            for ii, k in enumerate(ks):
                lo, hi = max(0, ii - smooth), min(len(ks), ii + smooth + 1)
                S[sid, k, :3, :3] = R[lo:hi].mean().as_matrix()
                S[sid, k, :3, 3] = t[lo:hi].mean(0)
    for i in range(tr.n_robot, N):
        sid = body[i] - nb_robot_bodies
        b = tr.birth[i]
        if b < 0 or sid < 0 or sid >= len(PT) or not np.isfinite(PT[sid, b, 0, 0]):
            continue
        xb = tr.pos[i, b].astype(np.float64)
        loc = np.linalg.inv(PT[sid, b]) @ np.r_[xb, 1.0]
        alive_part = np.isfinite(S[sid, :, 0, 0])
        dead_after = np.isfinite(tr.pos[i]).all(1)
        last = np.flatnonzero(dead_after).max()
        ks = np.flatnonzero(alive_part)
        ks = ks[ks <= last]
        out.pos[i, ks] = (S[sid, ks] @ loc)[:, :3]
    out.birth = np.array([np.flatnonzero(np.isfinite(out.pos[i, :, 0]))[0] if np.isfinite(out.pos[i, :, 0]).any() else -1
                          for i in range(N)], np.int32)
    return out


def convert(ep, arms, start=0, stop=None, cfg: TrackerConfig | None = None, frame_fn=None, depth_sources=None, verbose=True) -> Tracks:
    cfg = cfg or TrackerConfig()
    n, d = ep.table_plane()
    trk = Tracker(cfg, arms, n, d)
    stop = len(ep) if stop is None else min(stop, len(ep))
    outs, t0 = [], time.time()
    idx = list(range(start, stop))
    for k, i in enumerate(idx):
        f = ep.frame(i) if frame_fn is None else frame_fn(i)
        if depth_sources:
            attach_views(ep, f, depth_sources)
        outs.append(trk.step(f))
        if verbose and (k % 100 == 0 or k == len(idx) - 1):
            print(f"  step {k}/{len(idx)} points {len(outs[-1].pos)} objects {len(trk.objs)} "
                  f"({time.time() - t0:.0f} s)", flush=True)
    return pack(outs, ep, idx)


def pack(outs, ep, idx) -> Tracks:
    """StepOutput list -> Tracks (points padded with NaN before birth / after death)."""
    N, T = len(outs[-1].pos), len(outs)
    pos = np.full((N, T, 3), np.nan, np.float32)
    md = np.full((N, T), np.nan, np.float16)
    birth = np.full(N, -1, np.int32)
    held = np.zeros((N, T), bool)
    n_parts = max([max(list(o.obj_T) + list(o.obj_res) + [-1]) + 1 for o in outs])
    part_T = np.full((n_parts, T, 4, 4), np.nan, np.float32)
    obj_res = np.full((n_parts, T, 2), np.nan, np.float32)
    for k, o in enumerate(outs):
        n_ = len(o.pos)
        pos[:n_, k] = o.pos
        md[:n_, k] = o.measured_dist
        held[:n_, k] = o.held
        for sid, Tp in o.obj_T.items():
            part_T[sid, k] = Tp
        for sid, r in o.obj_res.items():
            obj_res[sid, k] = r
        newb = (birth[:n_] < 0)
        birth[:n_][newb] = k
    last = outs[-1]
    return Tracks(pos, last.group.copy(), last.body.copy(), birth, md, np.array([o.t for o in outs]),
                  np.array([ep.ticks[i] for i in idx]), last.n_robot, np.array(idx), held, part_T, obj_res)

