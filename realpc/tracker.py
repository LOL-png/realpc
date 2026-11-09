"""Causal 4D point tracker (object level): one call per frame, past and current data only.

    trk = Tracker(cfg, arms, table_n, table_d)  # arms = realpc.robot.BimanualArms
    for frame in frames:                        # realpc.io.Frame: fused cloud + colours + joint vectors
        out = trk.step(frame)                   # StepOutput: positions of every tracked point at this step

The same object serves batch conversion of recorded episodes (realpc.pipeline.convert) and streaming inference. It can
start at ANY frame: the first `step` segments the scene from that frame alone; surfaces seen later are added ("births").

Groups (a point keeps its id for its whole life):
  robot / gripper  fixed surface points on the URDF links, posed by FK (exact)
  static           the table (plane + table colour) and small clutter: exactly static, never moved
  rigid object     a colour-homogeneous geometric segment (region growing on Lab colour, no per-scene masks). ONE SE(3)
                   pose per object per step; all its points move with it:
                   - held (between the closed fingers of a gripper): T(t) = G(t) Y with Y = G(t_grasp)^-1 T(t_grasp)
                     (harness X E(t) Y, made causal); ICP may correct small slip;
                   - free: exactly static unless the robot is within `contact_radius` or an adjacent object moved this
                     step, AND its measured support dropped; then candidate poses (own ICP / the adjacent object's
                     motion + ICP) are scored by colour-consistent support. An object that moves with an adjacent
                     object is merged into it (over-segmented pieces, e.g. broom handle + bristles, become one body).
  deformable       an object that stops fitting rigidly while moving (cloth) switches to particles: nearest-neighbour
                   attraction both ways + graph smoothing + edge-length preservation + pinning to closed fingers.
                   Cloth has no true correspondences (README limits).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.spatial import cKDTree

STATIC, RIGID, DEFORM, ATTACHED = 0, 1, 2, 3     # ATTACHED = unmodelled robot hardware (camera mounts, cables)


@dataclass
class TrackerConfig:
    voxel: float = 0.006
    table_h: float = 0.008               # table = within this of the plane AND table colour
    table_dE: float = 15.0
    seg_eps: float = 0.012               # region growing radius
    seg_dE: float = 18.0                 # Lab distance to the region's mean colour
    seg_min: int = 40                    # smaller segments -> static clutter
    seg_min_nb: int = 4                  # radius-outlier filter (neighbours within seg_eps)
    robot_margin: float = 0.025
    wrist_margin: float = 0.04
    grip_close: float = 0.8
    grip_open: float = 0.9
    grasp_dist: float = 0.025
    icp_iters: int = 12
    icp_gate: float = 0.02
    fit_tol: float = 0.008
    color_tol: float = 25.0              # Lab distance for colour-consistent support / correspondences
    lost_frac: float = 0.6
    accept_gain: float = 0.1
    contact_radius: float = 0.05
    adjacent: float = 0.02               # objects closer than this are adjacent
    merge_trans: float = 0.01            # co-motion merge: pose within this of "moved with the neighbour"
    merge_rot: float = 0.09              # rad
    slip_trans: float = 0.02             # held: ICP may correct at most this much per step
    held_gain: float = 0.03              # ... when it raises the support by this much
    region: float = 0.03                 # measured points this close to an object belong to its region
    deform_fit: float = 0.35             # own-region explained fraction below this while moving ...
    deform_patience: int = 6             # ... this many consecutive steps -> deformable
    new_support_frac: float = 0.4        # a free object's new pose must take this share of its support from
                                         # measured points that nothing explained before (evidence it moved there)
    allow_deform: bool = False           # task metadata: the scene contains cloth (set per task; rigid otherwise)
    deform_min_pts: int = 300
    deform_near: float = 0.06            # unexplained measured points this close count against rigidity
    deform_unexpl: float = 0.5           # ... when more than this x the object's supported points
    nr_iters: int = 8
    nr_k: int = 8
    nr_smooth: int = 3
    nr_step: float = 0.7
    nr_stiff: int = 2
    nr_radius: float = 0.03
    pin_dist: float = 0.015
    birth_dist: float = 0.012
    birth_voxel: float = 0.012
    birth_persist: int = 3
    birth_min: int = 15
    birth_attach: float = 0.02
    death_steps: int = 40
    max_points: int = 150000
    motion: str = "icp"                  # "icp": poses from ICP on the fused cloud; "tracks": poses supplied per step by
                                         # 2D-track lifting (realpc.v3) through Tracker.ext_T / ext_flow
    move_min_trans: float = 0.005        # tracks mode: a free object moves only if the proposal differs this much ...
    move_min_rot: float = 0.035          # ... (or this many rad) and the robot / a moved neighbour is in contact
    held_max_dev: float = 0.05           # tracks mode: a held object's track pose may deviate this much from G Y
    move_persist: int = 3                # tracks mode: a free object moves after this many consecutive deviating steps
    min_mover_pts: int = 250             # tracks mode: smaller objects move only while held (no free motion)
    merge_persist: int = 10**9           # tracks mode: co-moving adjacent objects merge after this many agreeing steps (off: merged held objects with their support)
    flow_k: int = 8
    flow_radius: float = 0.04
    flow_max_step: float = 0.04          # cloth: per-step displacement larger than this is rejected
    flow_snap: float = 0.015             # cloth: pull toward measured points within this
    attach_dist: float = 0.06            # a segment mostly (attach_frac) within this of the FK robot surface is robot
    attach_frac: float = 0.8             # hardware missing from the URDF: it is fixed to its nearest link

@dataclass
class Obj:
    oid: int
    kind: int
    local: np.ndarray                    # rigid: points in the object frame (= world at the object's birth)
    lab: np.ndarray                      # per point Lab colour at birth
    gid: np.ndarray
    T: np.ndarray = field(default_factory=lambda: np.eye(4))
    T_prev: np.ndarray = field(default_factory=lambda: np.eye(4))
    X: np.ndarray | None = None          # deformable: current positions
    X_prev: np.ndarray | None = None
    edges: np.ndarray | None = None
    rest: np.ndarray | None = None
    held_by: str | None = None
    Y: np.ndarray | None = None
    support: float = 0.0
    unsup: np.ndarray | None = None
    bad: int = 0
    moved: bool = False
    link: int = -1                       # ATTACHED: robot body index
    pending: int = 0                     # tracks mode: consecutive steps the track pose disagreed with the current pose

    def positions(self, T=None):
        if self.kind == DEFORM:
            return self.X
        T = self.T if T is None else T
        return self.local @ T[:3, :3].T + T[:3, 3]


@dataclass
class StepOutput:
    step: int
    t: float
    pos: np.ndarray                      # [N, 3] float32 (ids 0..N-1, robot first; deleted points NaN)
    n_robot: int
    group: np.ndarray                    # [N] 0 gripper, 1 robot link, 2 static, 3 rigid object, 4 deformable
    body: np.ndarray                     # [N] robot body id, or n_bodies + object id
    measured_dist: np.ndarray            # [N] distance to the nearest measured point (m)
    held: np.ndarray                     # [N] bool
    obj_T: dict = field(default_factory=dict)       # {object id: 4x4}
    obj_res: dict = field(default_factory=dict)     # {object id: (median measured->object distance m, n measured)}


def voxel_down(P, v, return_index=False):
    k = np.floor(P / v).astype(np.int64)
    _, idx = np.unique(k, axis=0, return_index=True)
    idx = np.sort(idx)
    return (P[idx], idx) if return_index else P[idx]


def rgb_to_lab(C):
    L = cv2.cvtColor(np.asarray(C, np.uint8)[None], cv2.COLOR_RGB2LAB)[0].astype(np.float64)
    return L * np.array([100 / 255, 1, 1]) - np.array([0, 128, 128])


def region_grow(P, L, eps, dE, min_nb):
    """segments of points connected within eps whose Lab colour is within dE of the growing region's mean."""
    nb = cKDTree(P).query_ball_point(P, eps)
    deg = np.array([len(x) for x in nb])
    good = deg >= min_nb
    lab = np.full(len(P), -1)
    cur = 0
    for s in np.argsort(-deg):
        if lab[s] >= 0 or not good[s]:
            continue
        lab[s] = cur
        stack, sm, n = [s], L[s].copy(), 1
        while stack:
            i = stack.pop()
            for j in nb[i]:
                if lab[j] < 0 and good[j] and np.linalg.norm(L[j] - sm / n) < dE:
                    lab[j] = cur; stack.append(j); sm += L[j]; n += 1
        cur += 1
    return lab, cur


def kabsch(A, B):
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = cb - R @ ca
    return T


def apply_T(T, p):
    return T[:3, :3] @ p + T[:3, 3]


def rot_angle(R):
    return float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


class Tracker:
    def __init__(self, cfg: TrackerConfig, arms, table_n, table_d):
        self.cfg, self.arms = cfg, arms
        self.n, self.d = np.asarray(table_n, float), float(table_d)
        self.objs: list[Obj] = []
        self.n_robot = len(arms.local)
        self.next_gid = self.n_robot
        self.next_oid = 0
        self.k = 0
        self._un_cnt = {}
        self.dead: list[int] = []
        self.table_lab = None
        names = [l for _, l in arms.bodies]
        self.tip_bodies = {arm: [i for i, b in enumerate(arms.bodies) if b[0] == arm and b[1] in ("tip_left", "tip_right")]
                           for arm in arms.T_base}
        self.grip_body = {arm: [i for i, b in enumerate(arms.bodies) if b == (arm, "gripper")][0] for arm in arms.T_base}
        L, B = arms.dense_local()
        self._dense = (L, B, np.isin(np.array(names)[B], ["link5", "gripper"]))
        self.lab_group = np.zeros(0, np.int8)
        self.lab_body = np.zeros(0, np.int32)
        self.ext_T = {}                  # {object id: proposed 4x4 pose this step} (tracks mode)
        self.ext_flow = None             # (points at t-1 [n,3], points at t [n,3]) of visible lifted tracks

    # ------------------------------------------------------------------ bookkeeping
    def height(self, P):
        return P @ self.n + self.d

    def _label(self, o):
        need = int(o.gid.max()) + 1 if len(o.gid) else 0
        if need > len(self.lab_group):
            pad = need - len(self.lab_group) + 4096
            self.lab_group = np.concatenate([self.lab_group, np.zeros(pad, np.int8)])
            self.lab_body = np.concatenate([self.lab_body, np.zeros(pad, np.int32)])
        if o.kind == ATTACHED:
            nm = self.arms.bodies[o.link][1]
            self.lab_group[o.gid] = 0 if nm in ("gripper", "tip_left", "tip_right", "link5") else 1
            self.lab_body[o.gid] = o.link
        else:
            self.lab_group[o.gid] = {STATIC: 2, RIGID: 3, DEFORM: 4}[o.kind]
            self.lab_body[o.gid] = len(self.arms.bodies) + o.oid

    def _new_segment_obj(self, P, L):
        """object segment -> ATTACHED (robot hardware) if it hugs the robot, else RIGID."""
        cfg = self.cfg
        d, j = self._tree_rd.query(P, distance_upper_bound=cfg.attach_dist)
        if np.isfinite(d).mean() > cfg.attach_frac:
            link = int(np.bincount(self._dense_body[j[np.isfinite(d)]]).argmax())
            Tl = self._link_P[link]
            Ti = np.linalg.inv(Tl)
            o = self._new_obj(P @ Ti[:3, :3].T + Ti[:3, 3], L, ATTACHED, link=link)
            o.T = Tl.copy()
            return o
        o = self._new_obj(P, L, RIGID)
        o.support = self._support(o)
        return o

    def _new_obj(self, P, L, kind, link=-1):
        gid = np.arange(self.next_gid, self.next_gid + len(P)); self.next_gid += len(P)
        o = Obj(self.next_oid, kind, P.astype(np.float64).copy(), np.asarray(L, float).copy(), gid, link=link)
        self.next_oid += 1
        o.unsup = np.zeros(len(P), np.int16)
        self.objs.append(o); self._label(o)
        return o

    def _grow(self, o, P, L):
        gid = np.arange(self.next_gid, self.next_gid + len(P)); self.next_gid += len(P)
        if o.kind == DEFORM:
            o.X = np.concatenate([o.X, P]); o.local = np.concatenate([o.local, P])
        else:
            Ti = np.linalg.inv(o.T)
            o.local = np.concatenate([o.local, P @ Ti[:3, :3].T + Ti[:3, 3]])
        o.lab = np.concatenate([o.lab, L]); o.gid = np.concatenate([o.gid, gid])
        o.unsup = np.concatenate([o.unsup, np.zeros(len(P), np.int16)])
        if o.kind == DEFORM:
            self._graph(o)
        self._label(o)

    def _segment(self, P, L, allow_table=True):
        """table / clutter / object segments of a point set; returns list of (points, lab, kind)."""
        cfg = self.cfg
        out = []
        h = self.height(P)
        if allow_table and self.table_lab is not None:
            tab = (h < cfg.table_h) & (np.linalg.norm(L - self.table_lab, axis=1) < cfg.table_dE)
        else:
            tab = np.zeros(len(P), bool)
        if tab.any():
            out.append((P[tab], L[tab], "table"))
        R, LR = P[~tab], L[~tab]
        if len(R):
            lab, n = region_grow(R, LR, cfg.seg_eps, cfg.seg_dE, cfg.seg_min_nb)
            cnt = np.bincount(lab[lab >= 0], minlength=n) if n else np.zeros(0, int)
            small = np.ones(len(R), bool)
            for c in np.flatnonzero(cnt >= cfg.seg_min):
                m = lab == c
                out.append((R[m], LR[m], "object")); small &= ~m
            if small.any():
                out.append((R[small], LR[small], "clutter"))
        return out

    # ------------------------------------------------------------------ support / ICP
    def _supported(self, P, Lp):
        d, j = self._tree_m.query(P, distance_upper_bound=self.cfg.fit_tol)
        ok = np.isfinite(d)
        ok[ok] = np.linalg.norm(self._L[j[ok]] - Lp[ok], axis=1) < self.cfg.color_tol
        return ok

    def _support(self, o, T=None):
        P = o.positions(T)
        return float(self._supported(P, o.lab).mean()) if len(P) else 0.0

    def _icp(self, o, T0):
        cfg = self.cfg
        T = T0.copy()
        for _ in range(cfg.icp_iters):
            P = o.positions(T)
            d, j = self._tree_m.query(P, distance_upper_bound=cfg.icp_gate)
            ok = np.isfinite(d)
            ok[ok] = np.linalg.norm(self._L[j[ok]] - o.lab[ok], axis=1) < cfg.color_tol * 1.5
            if ok.sum() < max(10, 0.15 * len(P)):
                break
            T = kabsch(o.local[ok], self._M[j[ok]])
        return T

    # ------------------------------------------------------------------ deformable
    def _graph(self, o):
        k = min(self.cfg.nr_k, len(o.X) - 1)
        d, j = cKDTree(o.X).query(o.X, k + 1)
        e = np.unique(np.sort(np.stack([np.repeat(np.arange(len(o.X)), k), j[:, 1:].ravel()], 1), 1), axis=0)
        o.edges, o.rest = e, np.linalg.norm(o.X[e[:, 0]] - o.X[e[:, 1]], axis=1)

    def _to_deform(self, o):
        o.X = o.positions().copy(); o.kind = DEFORM; o.local = o.X.copy(); o.held_by = None
        self._graph(o); self._label(o)

    def _nonrigid(self, o, meas, pin_idx, pin_disp):
        cfg = self.cfg
        X = o.X.copy()
        if len(pin_idx):
            X[pin_idx] += pin_disp
        if len(meas):
            tm = cKDTree(meas)
            nb = o.edges
            for _ in range(cfg.nr_iters):
                disp = np.zeros_like(X); w = np.zeros(len(X))
                d, j = tm.query(X, distance_upper_bound=cfg.nr_radius)
                ok = np.isfinite(d)
                disp[ok] += meas[j[ok]] - X[ok]; w[ok] += 1
                d2, j2 = cKDTree(X).query(meas, distance_upper_bound=cfg.nr_radius)
                ok2 = np.isfinite(d2)
                np.add.at(disp, j2[ok2], meas[ok2] - X[j2[ok2]]); np.add.at(w, j2[ok2], 1)
                disp /= np.maximum(w, 1)[:, None]
                w = (w > 0).astype(float)
                for _ in range(cfg.nr_smooth):
                    acc = disp * w[:, None]; wa = w.copy()
                    np.add.at(acc, nb[:, 0], disp[nb[:, 1]] * w[nb[:, 1], None]); np.add.at(wa, nb[:, 0], w[nb[:, 1]])
                    np.add.at(acc, nb[:, 1], disp[nb[:, 0]] * w[nb[:, 0], None]); np.add.at(wa, nb[:, 1], w[nb[:, 0]])
                    disp = acc / np.maximum(wa, 1e-9)[:, None]; w = (wa > 0).astype(float)
                if len(pin_idx):
                    disp[pin_idx] = 0
                X = X + cfg.nr_step * disp
                for _ in range(cfg.nr_stiff):
                    a, b = nb[:, 0], nb[:, 1]
                    v = X[b] - X[a]; Lv = np.linalg.norm(v, axis=1) + 1e-9
                    c = 0.25 * (1 - o.rest / Lv)[:, None] * v
                    dx = np.zeros_like(X); cnt = np.zeros(len(X))
                    np.add.at(dx, a, c); np.add.at(dx, b, -c); np.add.at(cnt, a, 1); np.add.at(cnt, b, 1)
                    dx /= np.maximum(cnt, 1)[:, None]
                    if len(pin_idx):
                        dx[pin_idx] = 0
                    X = X + dx
        o.X = X

    # ------------------------------------------------------------------ step
    def step(self, frame) -> StepOutput:
        cfg = self.cfg
        M, vi = voxel_down(frame.points.astype(np.float64), cfg.voxel, return_index=True)
        L = rgb_to_lab(frame.colors[vi])
        link_P = self.arms.link_poses(frame.q)
        robot_pts = self.arms.points(frame.q, poses=link_P)
        Ld, Bd, wrist = self._dense
        R = np.einsum("nij,nj->ni", link_P[Bd, :3, :3], Ld) + link_P[Bd, :3, 3]
        keep = ~np.isfinite(cKDTree(R[~wrist]).query(M, distance_upper_bound=cfg.robot_margin)[0])
        keep &= ~np.isfinite(cKDTree(R[wrist]).query(M, distance_upper_bound=cfg.wrist_margin)[0])
        self._M, self._L = M[keep], L[keep]
        self._link_P = link_P
        self._tree_rd = cKDTree(R); self._dense_body = Bd
        for o in self.objs:
            if o.kind == ATTACHED:
                o.T = link_P[o.link].copy()
        self._tree_m = cKDTree(self._M)
        self._tree_r = cKDTree(robot_pts)
        G = {arm: link_P[self.grip_body[arm]] for arm in self.arms.T_base}
        tips = {arm: [robot_pts[self.arms.body == b] for b in self.tip_bodies[arm]] for arm in self.arms.T_base}
        if self.k == 0:
            near = self.height(self._M) < cfg.table_h
            self.table_lab = np.median(self._L[near], 0) if near.any() else None
            for P, Lp, kind in self._segment(self._M, self._L):
                if kind == "object":
                    self._new_segment_obj(P, Lp)
                else:
                    self._new_obj(P, Lp, STATIC)
            for o in self.objs:
                o.support = self._support(o)
        else:
            self._track(frame, G, tips)
        self._prev_G = G
        out = self._output(frame, robot_pts)
        self.k += 1
        return out

    def _track(self, frame, G, tips):
        cfg = self.cfg
        movable = [o for o in self.objs if o.kind in (RIGID, DEFORM)]
        for o in movable:
            o.T_prev = o.T.copy()
            o.X_prev = o.X.copy() if o.kind == DEFORM else None
        # grasp state
        for arm in self.arms.T_base:
            g = float(frame.q[arm][-1])
            for o in movable:
                if o.held_by == arm and g > cfg.grip_open:
                    o.held_by, o.Y = None, None
            if g < cfg.grip_close and not any(q.held_by == arm for q in movable):
                best = None
                for o in movable:
                    if o.kind != RIGID or o.held_by is not None:
                        continue
                    tr = cKDTree(o.positions())
                    dl, dr = tr.query(tips[arm][0])[0].min(), tr.query(tips[arm][1])[0].min()
                    if dl < cfg.grasp_dist and dr < cfg.grasp_dist and (best is None or dl + dr < best[0]):
                        best = (dl + dr, o)
                if best is not None:
                    o = best[1]
                    o.held_by, o.Y = arm, np.linalg.inv(self._prev_G[arm]) @ o.T
        moved = set()
        if cfg.motion == "tracks":
            self._track_ext(frame, G, tips, movable, moved)
            self._deaths()
            self._births()
            return
        # held rigid objects follow their gripper
        for o in movable:
            if o.kind == RIGID and o.held_by is not None:
                T0 = G[o.held_by] @ o.Y
                T1 = self._icp(o, T0)
                if np.linalg.norm(T1[:3, 3] - T0[:3, 3]) < cfg.slip_trans and self._support(o, T1) > self._support(o, T0) + cfg.held_gain:
                    T0 = T1
                o.T = T0; o.Y = np.linalg.inv(G[o.held_by]) @ o.T
                o.moved = True; moved.add(o.oid)
        # deformable objects
        for o in movable:
            if o.kind == DEFORM:
                pin_idx, pin_disp = [], []
                for arm in self.arms.T_base:
                    if float(frame.q[arm][-1]) < cfg.grip_close:
                        dd = cKDTree(np.concatenate(tips[arm])).query(o.X)[0]
                        pi = np.flatnonzero(dd < cfg.pin_dist)
                        if len(pi):
                            D = G[arm] @ np.linalg.inv(self._prev_G[arm])
                            pin_idx.append(pi); pin_disp.append(o.X[pi] @ D[:3, :3].T + D[:3, 3] - o.X[pi])
                pin_idx = np.concatenate(pin_idx) if pin_idx else np.zeros(0, int)
                pin_disp = np.concatenate(pin_disp) if pin_disp else np.zeros((0, 3))
                d = cKDTree(o.X).query(self._M, distance_upper_bound=cfg.region)[0]
                self._nonrigid(o, self._M[np.isfinite(d)], pin_idx, pin_disp)
                if np.abs(o.X - o.X_prev).max() > 0.005:
                    o.moved = True; moved.add(o.oid)
        # measured points explained by nothing (objects at their current / held poses)
        allP = np.concatenate([o.positions() for o in self.objs])
        own = np.concatenate([np.full(len(o.gid), o.oid) for o in self.objs])
        dmin, jmin = cKDTree(allP).query(self._M)
        self._unexpl = dmin > cfg.birth_dist
        self._owner_m = np.where(dmin < cfg.region, own[jmin], -1)
        # free rigid objects: move only on evidence + contact
        for _ in range(3):
            changed = False
            for o in list(movable):
                if o not in self.objs or o.kind != RIGID or o.held_by is not None or o.oid in moved:
                    continue
                P0 = o.positions()
                s0 = self._support(o)
                if s0 >= cfg.lost_frac * o.support and s0 > 0.05:
                    continue
                nbrs = [q for q in self.objs if q.oid in moved and q is not o and q.kind in (RIGID, DEFORM) and
                        np.isfinite(cKDTree(q.positions(q.T_prev) if q.kind == RIGID else q.X_prev).query(
                            P0, distance_upper_bound=cfg.adjacent)[0]).any()]
                near_robot = bool(np.isfinite(self._tree_r.query(P0, distance_upper_bound=cfg.contact_radius)[0]).any())
                if not (near_robot or nbrs):
                    continue
                cands = [(self._icp(o, o.T), None)]
                for q in nbrs:
                    if q.kind == RIGID:
                        D = q.T @ np.linalg.inv(q.T_prev)
                        cands.append((self._icp(o, D @ o.T), (q, D @ o.T)))
                sc = [self._support(o, T) for T, _ in cands]
                b = int(np.argmax(sc))
                d, j = self._tree_m.query(o.positions(cands[b][0]), distance_upper_bound=cfg.fit_tol)
                okj = j[np.isfinite(d)]
                new_frac = float(self._unexpl[okj].mean()) if len(okj) else 0.0
                if sc[b] > s0 + cfg.accept_gain and new_frac >= cfg.new_support_frac:
                    T, src = cands[b]
                    o.T = T; o.moved = True; moved.add(o.oid); changed = True
                    if src is not None:
                        q, Tq = src
                        if np.linalg.norm(T[:3, 3] - Tq[:3, 3]) < cfg.merge_trans and rot_angle(T[:3, :3] @ Tq[:3, :3].T) < cfg.merge_rot:
                            self._merge(o, q)
            if not changed:
                break
        # rigidity test -> deformable: a moving object whose own points lose support while many measured points that
        # nothing explains (of similar colour) appear right next to it is not rigid
        for o in [o for o in self.objs if cfg.allow_deform and o.kind == RIGID and o.oid in moved and len(o.local) >= cfg.deform_min_pts]:
            P = o.positions()
            s = self._support(o)
            near = cKDTree(P).query(self._M, distance_upper_bound=cfg.deform_near)[0]
            un = np.isfinite(near) & self._unexpl_after(o)
            col_ok = np.linalg.norm(self._L - o.lab.mean(0), axis=1) < 2 * cfg.seg_dE
            n_un = int((un & col_ok).sum())
            n_sup = int(self._supported(P, o.lab).sum())
            o.bad = o.bad + 1 if (s < cfg.deform_fit and n_un > cfg.deform_unexpl * max(n_sup, 1)) else 0
            if o.bad >= cfg.deform_patience:
                self._to_deform(o)
        for o in self.objs:
            if o.kind in (RIGID, DEFORM):
                o.support = max(self._support(o), 0.98 * o.support)
        self._deaths()
        self._births()

    def _track_ext(self, frame, G, tips, movable, moved):
        """poses from lifted 2D tracks (ext_T), gripper kinematics for held objects, contact prior for free ones."""
        cfg = self.cfg
        self._G_now = G
        for o in movable:
            if o.kind == RIGID and o.held_by is not None:
                T0 = G[o.held_by] @ o.Y
                Te = self.ext_T.get(o.oid)
                o.T = Te if (Te is not None and np.linalg.norm(Te[:3, 3] - T0[:3, 3]) < cfg.held_max_dev) else T0
                o.Y = np.linalg.inv(G[o.held_by]) @ o.T
                o.moved = True; moved.add(o.oid)
        for o in movable:
            if o.kind == DEFORM and self.ext_flow is not None and len(self.ext_flow[0]) >= 3:
                A, B = self.ext_flow
                d, j = cKDTree(A).query(o.X, k=min(cfg.flow_k, len(A)), distance_upper_bound=cfg.flow_radius)
                d, j = np.atleast_2d(d), np.atleast_2d(j)
                ok = np.isfinite(d)
                w = np.where(ok, np.exp(-(np.where(ok, d, 0) / (cfg.flow_radius / 2)) ** 2), 0.0)
                jj = np.where(ok, j, 0)
                disp = (w[..., None] * (B[jj] - A[jj])).sum(1) / np.maximum(w.sum(1), 1e-9)[:, None]
                has = w.sum(1) > 1e-6
                disp = np.where((np.linalg.norm(disp, axis=1) < cfg.flow_max_step)[:, None], disp, 0.0)
                if has.any():
                    o.X = o.X + np.where(has[:, None], disp, 0.0)
                    # pull onto the measured surface (colour-consistent nearest measured point), half way
                    dm, jm = self._tree_m.query(o.X, distance_upper_bound=cfg.flow_snap)
                    okm = np.isfinite(dm)
                    okm[okm] = np.linalg.norm(self._L[jm[okm]] - o.lab[okm], axis=1) < cfg.color_tol * 1.5
                    o.X[okm] += 0.5 * (self._M[jm[okm]] - o.X[okm])
                    if np.abs(disp[has]).max() > 0.003:
                        o.moved = True; moved.add(o.oid)
        for _ in range(2):
            for o in movable:
                if o.kind == RIGID and o.oid not in self.ext_T:
                    o.pending = 0
                if o.kind != RIGID or o.held_by is not None or o.oid in moved or o.oid not in self.ext_T:
                    continue
                if len(o.gid) < cfg.min_mover_pts:
                    continue
                Te = self.ext_T[o.oid]
                dt = np.linalg.norm(Te[:3, 3] - o.T[:3, 3]); dr = rot_angle(Te[:3, :3] @ o.T[:3, :3].T)
                if dt < cfg.move_min_trans and dr < cfg.move_min_rot:
                    o.pending = 0
                    continue
                o.pending += 1
                if o.pending < cfg.move_persist:
                    continue
                P0 = o.positions()
                near_robot = bool(np.isfinite(self._tree_r.query(P0, distance_upper_bound=cfg.contact_radius)[0]).any())
                nbr = any(q.oid in moved and np.isfinite(cKDTree(q.positions()).query(P0, distance_upper_bound=cfg.adjacent)[0]).any()
                          for q in movable if q is not o)
                on_robot = np.isfinite(self._tree_rd.query(o.positions(Te), distance_upper_bound=0.02)[0]).mean() > 0.3
                if (near_robot or nbr) and not on_robot:
                    o.T = Te; o.moved = True; moved.add(o.oid)
        self._merge_comoving(movable, moved)

    def _merge_comoving(self, movable, moved):
        """tracks mode: adjacent rigid objects whose motion this step agrees become one rigid body."""
        cfg = self.cfg
        ms = [o for o in movable if o.kind == RIGID and o.oid in moved and o in self.objs]
        if not hasattr(self, "_comove"):
            self._comove = {}
        seen = set()
        for i, o in enumerate(ms):
            for q in ms[i + 1:]:
                if o not in self.objs or q not in self.objs:
                    continue
                Do, Dq = o.T @ np.linalg.inv(o.T_prev), q.T @ np.linalg.inv(q.T_prev)
                c = np.concatenate([o.positions(o.T_prev), q.positions(q.T_prev)]).mean(0)
                if np.linalg.norm(apply_T(Do, c) - apply_T(Dq, c)) > 0.01 or rot_angle(Do[:3, :3] @ Dq[:3, :3].T) > 0.09:
                    continue
                if np.linalg.norm(Do[:3, 3]) < 0.005 and rot_angle(Do[:3, :3]) < 0.02:
                    continue                            # both (nearly) still: no evidence they are one body
                if not np.isfinite(cKDTree(q.positions(q.T_prev)).query(o.positions(o.T_prev), distance_upper_bound=cfg.adjacent)[0]).any():
                    continue
                key = (min(o.oid, q.oid), max(o.oid, q.oid))
                self._comove[key] = self._comove.get(key, 0) + 1
                if self._comove[key] < cfg.merge_persist:     # agreement on several consecutive moving steps
                    continue
                big, small = (o, q) if len(o.gid) >= len(q.gid) else (q, o)
                if small.held_by is not None and big.held_by is None:
                    big.held_by, big.Y = small.held_by, None
                self._merge(small, big)
                if big.held_by is not None and big.Y is None:
                    big.Y = np.linalg.inv(self._G_now[big.held_by]) @ big.T

    def _unexpl_after(self, o):
        """measured points not explained by any object at its CURRENT pose."""
        allP = np.concatenate([q.positions() for q in self.objs])
        return cKDTree(allP).query(self._M, distance_upper_bound=self.cfg.birth_dist)[0] > self.cfg.birth_dist

    def _merge(self, o, q):
        """o moved with q: o's points join q's rigid body (one pose from now on)."""
        if o not in self.objs or q not in self.objs:
            return
        if q.kind == DEFORM:
            q.X = np.concatenate([q.X, o.positions()]); q.local = np.concatenate([q.local, o.positions()])
        else:
            A = np.linalg.inv(q.T) @ o.T
            q.local = np.concatenate([q.local, o.local @ A[:3, :3].T + A[:3, 3]])
        q.lab = np.concatenate([q.lab, o.lab]); q.gid = np.concatenate([q.gid, o.gid]); q.unsup = np.concatenate([q.unsup, o.unsup])
        if q.kind == DEFORM:
            self._graph(q)
        self.objs.remove(o); self._label(q)

    def _deaths(self):
        cfg = self.cfg
        for o in list(self.objs):
            if o.kind in (STATIC, ATTACHED) or o.held_by is not None:
                continue
            sup = self._supported(o.positions(), o.lab)
            o.unsup = np.where(sup, 0, o.unsup + 1).astype(np.int16)
            dead = o.unsup >= cfg.death_steps
            if dead.any():
                self.dead.extend(o.gid[dead].tolist())
                for a in ("local", "lab", "gid", "unsup"):
                    setattr(o, a, getattr(o, a)[~dead])
                if o.kind == DEFORM:
                    o.X = o.X[~dead]
                    if len(o.X) > 1:
                        self._graph(o)
                if len(o.gid) == 0:
                    self.objs.remove(o)

    def _births(self):
        cfg = self.cfg
        M, L = self._M, self._L
        allP = np.concatenate([o.positions() for o in self.objs]) if self.objs else np.zeros((0, 3))
        owner = np.concatenate([np.full(len(o.gid), i) for i, o in enumerate(self.objs)]) if self.objs else np.zeros(0, int)
        d, j = cKDTree(allP).query(M) if len(allP) else (np.full(len(M), np.inf), np.zeros(len(M), int))
        un = d > cfg.birth_dist
        key = np.floor(M / cfg.birth_voxel).astype(np.int64)
        kk = (key[:, 0] * 73856093) ^ (key[:, 1] * 19349663) ^ (key[:, 2] * 83492791)
        cnt = {x: self._un_cnt.get(x, 0) + 1 for x in np.unique(kk[un])}
        self._un_cnt = cnt
        if not un.any() or self.next_gid >= cfg.max_points:
            return
        ripe = un & np.isin(kk, [x for x, c in cnt.items() if c >= cfg.birth_persist])
        if ripe.sum() < cfg.birth_min:
            return
        for x in np.unique(kk[ripe]):
            cnt.pop(x, None)
        U, LU, dU, jU = M[ripe], L[ripe], d[ripe], j[ripe]
        attach = dU < cfg.birth_dist + cfg.birth_attach
        objs = list(self.objs)
        for oi in np.unique(owner[jU[attach]]):
            m = attach & (owner[jU] == oi)
            o = objs[oi]
            col_ok = np.linalg.norm(LU[m] - o.lab.mean(0), axis=1) < 2 * cfg.seg_dE
            if o.kind != STATIC and col_ok.any():
                mm = np.flatnonzero(m)[col_ok]
                self._grow(o, U[mm], LU[mm])
                attach[np.flatnonzero(m)[~col_ok]] = False
            else:
                attach[m] = False
        rest = ~attach
        if rest.sum() >= cfg.birth_min:
            for P, Lp, kind in self._segment(U[rest], LU[rest]):
                if kind == "object":
                    self._new_segment_obj(P, Lp)
                elif kind == "table":
                    tab = [o for o in self.objs if o.kind == STATIC]
                    self._grow(tab[0], P, Lp) if tab else self._new_obj(P, Lp, STATIC)

    def _output(self, frame, robot_pts):
        N = self.next_gid
        pos = np.zeros((N, 3), np.float32)
        grp = np.zeros(N, np.int8); body = np.zeros(N, np.int32); held = np.zeros(N, bool)
        pos[: self.n_robot] = robot_pts
        lname = np.array([l for _, l in self.arms.bodies])
        grp[: self.n_robot] = np.where(np.isin(lname[self.arms.body], ["gripper", "tip_left", "tip_right"]), 0, 1)
        body[: self.n_robot] = self.arms.body
        if N > self.n_robot:
            grp[self.n_robot:] = self.lab_group[self.n_robot:N]; body[self.n_robot:] = self.lab_body[self.n_robot:N]
        res = {}
        for o in self.objs:
            pos[o.gid] = o.positions()
            held[o.gid] = o.held_by is not None
        # per-object pose residual: measured points whose nearest tracked point (any object) is on the object, within
        # `region` -> median distance
        objs = [o for o in self.objs if o.kind in (RIGID, DEFORM) and o.moved]
        if objs and self.objs:
            allP = np.concatenate([o.positions() for o in self.objs])
            own = np.concatenate([np.full(len(o.gid), o.oid) for o in self.objs])
            d, j = cKDTree(allP).query(self._M, distance_upper_bound=self.cfg.region)
            ok = np.isfinite(d)
            for o in objs:
                m = ok & (own[np.minimum(j, len(own) - 1)] == o.oid)
                res[o.oid] = (float(np.median(d[m])) if m.any() else float("nan"), int(m.sum()))
        if self.dead:
            pos[np.array(self.dead)] = np.nan
        md = np.full(N, np.nan, np.float32)
        al = np.isfinite(pos).all(1)
        md[al] = self._tree_m.query(pos[al])[0]
        return StepOutput(self.k, frame.t, pos, self.n_robot, grp, body, md, held,
                          {o.oid: o.T.copy() for o in self.objs if o.kind == RIGID}, res)
