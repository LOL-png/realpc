"""Causal 6-DoF pose tracking of library meshes (training-data conversion and live inference use this same code).

State per library object: pose T_world_obj, status (static | contact | held by an arm with a fixed grasp offset).
Per tick t (only frames <= t are used):
  1. prediction  held: G_arm(t) . T_grip_obj (FK of the gripper link); otherwise the previous pose
  2. tracks      2D tracks of the object's visible pixels (top camera, frame t-1 -> t), lifted to 3D with the depth of
                 both frames, RANSAC + Kabsch -> a candidate pose (moving, non-held objects)
  3. ICP         projective point-to-plane ICP of the VISIBLE mesh samples (robot / other objects occlude, from z-buffer
                 renders) against the top (+ side when its lag is small) depth, Huber weights
  4. silhouette  free-space term: a visible mesh sample whose pixel shows a surface clearly BEHIND it (we look through
                 where the object would be) is pulled to that measured point
  Static objects without robot contact keep their pose (no drift). Points = fixed mesh samples; robot = FK samples.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

from .diffcalib import Renderer


def se3_small(xi):
    w, v = xi[:3], xi[3:]
    th = np.linalg.norm(w)
    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    R = np.eye(3) + (np.sin(th) / th * K + (1 - np.cos(th)) / th ** 2 * K @ K if th > 1e-9 else K)
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = v
    return T


def kabsch(A, B, w=None):
    w = np.ones(len(A)) if w is None else w
    ca = (A * w[:, None]).sum(0) / w.sum(); cb = (B * w[:, None]).sum(0) / w.sum()
    H = ((A - ca) * w[:, None]).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = cb - R @ ca
    return T


@dataclass
class LibObject:
    name: str
    mesh: trimesh.Trimesh
    n_samples: int = 1500
    samples: np.ndarray = None          # [S,3] object frame (the output points)
    normals: np.ndarray = None
    icp_pts: np.ndarray = None          # denser samples for ICP
    icp_nrm: np.ndarray = None
    verts_t: torch.Tensor = None
    faces_t: torch.Tensor = None

    def __post_init__(self):
        p, fi = trimesh.sample.sample_surface_even(self.mesh, self.n_samples * 2, seed=0)
        p, fi = np.asarray(p)[: self.n_samples], np.asarray(fi)[: self.n_samples]
        self.samples = p.astype(np.float64); self.normals = self.mesh.face_normals[fi]
        q, fj = trimesh.sample.sample_surface_even(self.mesh, 6000, seed=1)
        self.icp_pts = np.asarray(q, float); self.icp_nrm = self.mesh.face_normals[np.asarray(fj)]
        m = self.mesh
        if len(m.faces) > 6000:
            import fast_simplification
            v2, f2 = fast_simplification.simplify(np.asarray(m.vertices, np.float32), np.asarray(m.faces), target_reduction=1 - 6000 / len(m.faces))
            m = trimesh.Trimesh(v2, f2, process=False)
        self.verts_t = torch.tensor(np.asarray(m.vertices), dtype=torch.float32, device="cuda")
        self.faces_t = torch.tensor(np.asarray(m.faces), dtype=torch.int64, device="cuda")


@dataclass
class ObjState:
    T: np.ndarray
    held: str | None = None
    T_grip_obj: np.ndarray | None = None
    grip_at_grasp: float = 1.0
    last_res: tuple = (np.nan, 0)
    history: list = field(default_factory=list)


class CamView:
    """one camera at one tick: depth (m), K, T_world_cam, grey image (for flow), low-res renderer."""

    def __init__(self, name, D, K, Twc, grey=None, lowres=(180, 320)):
        self.name, self.D, self.K, self.Twc = name, D, np.asarray(K, float), np.asarray(Twc, float)
        self.Tcw = np.linalg.inv(self.Twc)
        self.grey = grey
        H, W = D.shape
        self.s = lowres[1] / W
        Kl = self.K.copy(); Kl[:2] *= self.s
        self.R = Renderer(Kl, int(round(H * self.s)), lowres[1])

    def zbuf(self, verts_world: torch.Tensor, faces):
        M = torch.tensor(self.Tcw, dtype=torch.float32, device="cuda")
        Vc = verts_world @ M[:3, :3].T + M[:3, 3]
        _, z = self.R.raster(Vc[None], faces)
        return z[0]


class MeshTracker:
    def __init__(self, objects: list, init_poses: dict, arm_model, cfg=None):
        """objects: LibObject list; init_poses {name: T_world_obj}; arm_model: TorchArm (for FK, gripper, occlusion)."""
        self.objs = {o.name: o for o in objects}
        self.state = {o.name: ObjState(np.asarray(init_poses[o.name], float)) for o in objects}
        self.arm = arm_model
        self.cfg = dict(contact=0.03, grasp_tip=0.03, grip_close=0.06, release_open=0.15, icp_iters=6, huber=0.005,
                        max_corr=0.03, free_tol=0.015, min_inl=40, move_gate=0.003, flow=True)
        self.cfg.update(cfg or {})
        self.prev = None                            # previous tick's views (for flow / track Kabsch)
        self.tick = -1

    # -------------------------------------------------------------- robot
    def robot(self, q: dict, T_world_base: dict):
        """-> world vertices per arm (torch), gripper poses, fingertip points."""
        out = {}
        for arm_, Tb in T_world_base.items():
            qt = torch.tensor(np.asarray(q[arm_])[None], dtype=torch.float32, device="cuda")
            V = self.arm.verts(qt)[0]
            Tb_t = torch.tensor(Tb, dtype=torch.float32, device="cuda")
            Vw = V @ Tb_t[:3, :3].T + Tb_t[:3, 3]
            L = self.arm.link_tf(qt)
            G = Tb @ L["gripper"][0].cpu().numpy().astype(float)
            tips = np.concatenate([(Tb @ L[k][0].cpu().numpy().astype(float))[None] for k in ("tip_left", "tip_right")])
            out[arm_] = dict(V=Vw, G=G, tips=tips[:, :3, 3], grip=float(q[arm_][6]))
        return out

    # -------------------------------------------------------------- main step
    def step(self, t, views: list, q: dict, T_world_base: dict):
        """views: list of CamView for this tick (cameras whose content time ~ t). Returns {name: T_world_obj}."""
        self.tick = t
        rob = self.robot(q, T_world_base)
        # robot z-buffers per view (occlusion)
        robot_z = []
        for v in views:
            zs = [v.zbuf(r["V"], self.arm.faces) for r in rob.values()]
            z = torch.stack([torch.where(x > 0, x, torch.full_like(x, 1e9)) for x in zs]).amin(0)
            robot_z.append(z.cpu().numpy())
        # 1. prediction + grasp / release logic
        for name, st in self.state.items():
            o = self.objs[name]
            P = st.T[:3, :3] @ o.samples.T
            P = (P.T + st.T[:3, 3])
            if st.held is not None:
                r = rob[st.held]
                if r["grip"] > st.grip_at_grasp + self.cfg["release_open"]:
                    st.held, st.T_grip_obj = None, None              # released: stays where it is
                else:
                    st.T = r["G"] @ st.T_grip_obj
            else:
                for arm_, r in rob.items():
                    d = cKDTree(P).query(r["tips"])[0]
                    if d.max() < self.cfg["grasp_tip"] and r["grip"] < 1.0 - self.cfg["grip_close"]:
                        st.held = arm_
                        st.grip_at_grasp = r["grip"]
                        st.T_grip_obj = np.linalg.inv(r["G"]) @ st.T
                        break
        # object z-buffers (inter-object occlusion), with predicted poses
        obj_z = {}
        for name, st in self.state.items():
            o = self.objs[name]
            Tt = torch.tensor(st.T, dtype=torch.float32, device="cuda")
            Vw = o.verts_t @ Tt[:3, :3].T + Tt[:3, 3]
            obj_z[name] = [v.zbuf(Vw, o.faces_t).cpu().numpy() for v in views]
        out = {}
        for name, st in self.state.items():
            o = self.objs[name]
            near_robot = min(cKDTree((st.T[:3, :3] @ o.samples.T).T + st.T[:3, 3]).query(
                torch.cat([r["V"] for r in rob.values()]).cpu().numpy()[::7], distance_upper_bound=self.cfg["contact"])[0].min(),
                1.0) < self.cfg["contact"]
            moving = st.held is not None or near_robot
            if moving:
                T0 = st.T.copy()
                if st.held is None and self.cfg["flow"] and self.prev is not None:
                    Tk = self.track_kabsch(o, st.T, views, self.prev, robot_z)
                    if Tk is not None:
                        T0 = Tk @ st.T
                T1, res = self.icp(o, T0, views, robot_z, obj_z, name)
                if res[1] >= self.cfg["min_inl"]:
                    # static-but-touched objects only move on clear evidence
                    dmove = np.linalg.norm(T1[:3, 3] - st.T[:3, 3])
                    if st.held is not None or dmove > self.cfg["move_gate"]:
                        st.T = T1
                    if st.held is not None:                            # slow update of the grasp offset (slip)
                        st.T_grip_obj = np.linalg.inv(rob[st.held]["G"]) @ st.T
                st.last_res = res
            else:
                st.last_res = self.residual(o, st.T, views, robot_z, obj_z, name)
            out[name] = st.T.copy()
            st.history.append(st.T.copy())
        self.prev = views
        return out

    # -------------------------------------------------------------- measurement helpers
    def visible(self, o, T, v, robot_z, obj_z_self, obj_z_all, name, dense=True):
        P = o.icp_pts if dense else o.samples
        N = o.icp_nrm if dense else o.normals
        Pw = P @ T[:3, :3].T + T[:3, 3]
        Nw = N @ T[:3, :3].T
        Pc = Pw @ v.Tcw[:3, :3].T + v.Tcw[:3, 3]
        z = Pc[:, 2]
        H, W = v.D.shape
        u = v.K[0, 0] * Pc[:, 0] / np.maximum(z, 1e-6) + v.K[0, 2]
        w = v.K[1, 1] * Pc[:, 1] / np.maximum(z, 1e-6) + v.K[1, 2]
        ok = (z > 0.1) & (u >= 0) & (u < W - 1) & (w >= 0) & (w < H - 1)
        camc = v.Twc[:3, 3]
        ok &= ((camc - Pw) * Nw).sum(1) > 0                            # front-facing
        ul = np.clip((u * v.s).astype(int), 0, robot_z.shape[1] - 1); wl = np.clip((w * v.s).astype(int), 0, robot_z.shape[0] - 1)
        ok &= z < robot_z[wl, ul] - 0.01                               # robot in front -> occluded
        for other, zl in obj_z_all.items():
            if other == name:
                continue
            zo = zl[wl, ul]
            ok &= ~((zo > 0) & (zo < z - 0.01))
        return Pw[ok], Nw[ok], u[ok], w[ok], z[ok]

    def icp(self, o, T, views, robot_z, obj_z, name):
        T = T.copy()
        res = (np.nan, 0)
        for it in range(self.cfg["icp_iters"]):
            A, b, wts, resid = [], [], [], []
            for vi, v in enumerate(views):
                Pw, Nw, u, w, z = self.visible(o, T, v, robot_z[vi], None, {k: zz[vi] for k, zz in obj_z.items()}, name)
                if len(Pw) < 10:
                    continue
                d = v.D[w.astype(int), u.astype(int)]
                valid = d > 0.1
                # measured point at the sample's pixel (projective association)
                uu, ww = u[valid], w[valid]
                dd = d[valid]
                Pm_c = np.stack([(uu - v.K[0, 2]) / v.K[0, 0] * dd, (ww - v.K[1, 2]) / v.K[1, 1] * dd, dd], 1)
                Pm = Pm_c @ v.Twc[:3, :3].T + v.Twc[:3, 3]
                Ps, Ns = Pw[valid], Nw[valid]
                dz = dd - z[valid]
                close = np.abs(dz) < self.cfg["max_corr"]
                free = dz > self.cfg["free_tol"]                        # seen through: free-space violation
                use = close | free
                if use.sum() < 5:
                    continue
                Ps, Ns, Pm = Ps[use], Ns[use], Pm[use]
                r = ((Pm - Ps) * Ns).sum(1)
                J = np.c_[np.cross(Ps, Ns), Ns]
                hub = self.cfg["huber"]
                wt = np.where(np.abs(r) < hub, 1.0, hub / np.maximum(np.abs(r), 1e-9))
                wt = wt * np.where(free[use] & ~close[use], 0.5, 1.0)
                A.append(J); b.append(r); wts.append(wt); resid.append(np.abs(r[close[use]]))
            if not A:
                break
            J = np.concatenate(A); r = np.concatenate(b); wt = np.concatenate(wts)
            H = (J * wt[:, None]).T @ J + 1e-6 * np.eye(6)
            g = (J * wt[:, None]).T @ r
            xi = np.linalg.solve(H, g)
            dT = se3_small(np.r_[xi[:3], xi[3:]])
            T = dT @ T
            rr = np.concatenate(resid) if resid else np.zeros(0)
            res = (float(np.median(rr)) if len(rr) else np.nan, int(len(rr)))
            if np.linalg.norm(xi) < 1e-5:
                break
        return T, res

    def residual(self, o, T, views, robot_z, obj_z, name):
        rr = []
        for vi, v in enumerate(views):
            Pw, Nw, u, w, z = self.visible(o, T, v, robot_z[vi], None, {k: zz[vi] for k, zz in obj_z.items()}, name)
            if len(Pw) == 0:
                continue
            d = v.D[w.astype(int), u.astype(int)]
            ok = (d > 0.1) & (np.abs(d - z) < self.cfg["max_corr"])
            rr.append(np.abs(d[ok] - z[ok]))
        rr = np.concatenate(rr) if rr else np.zeros(0)
        return (float(np.median(rr)) if len(rr) else np.nan, int(len(rr)))

    def track_kabsch(self, o, T, views, prev, robot_z):
        """2D tracks (DIS optical flow, frame t-1 -> t, top camera) on the object's visible pixels, lifted with both
        depths -> RANSAC Kabsch -> delta pose (world). None if not enough agreeing tracks."""
        v1 = views[0]; v0 = prev[0]
        if v1.grey is None or v0.grey is None:
            return None
        Pw, Nw, u, w, z = self.visible(o, T, v0, robot_z[0], None, {}, o.name, dense=True)
        if len(u) < 30:
            return None
        if not hasattr(self, "_dis"):
            self._dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        flow = self._dis.calc(v0.grey, v1.grey, None)
        ui, wi = u.astype(int), w.astype(int)
        f = flow[wi, ui]
        u2, w2 = u + f[:, 0], w + f[:, 1]
        H, W = v1.D.shape
        ok = (u2 >= 0) & (u2 < W) & (w2 >= 0) & (w2 < H)
        d0 = v0.D[wi, ui]; d1 = np.zeros_like(d0); d1[ok] = v1.D[w2[ok].astype(int), u2[ok].astype(int)]
        ok &= (d0 > 0.1) & (d1 > 0.1) & (np.abs(d0 - z) < 0.02)
        if ok.sum() < 20:
            return None
        def lift(v, uu, ww, dd):
            Pc = np.stack([(uu - v.K[0, 2]) / v.K[0, 0] * dd, (ww - v.K[1, 2]) / v.K[1, 1] * dd, dd], 1)
            return Pc @ v.Twc[:3, :3].T + v.Twc[:3, 3]
        A = lift(v0, u[ok], w[ok], d0[ok]); B = lift(v1, u2[ok], w2[ok], d1[ok])
        rng = np.random.default_rng(self.tick)
        best = None
        for _ in range(64):
            i = rng.choice(len(A), 3, replace=False)
            Tk = kabsch(A[i], B[i])
            inl = np.linalg.norm(A @ Tk[:3, :3].T + Tk[:3, 3] - B, axis=1) < 0.006
            if best is None or inl.sum() > best[1].sum():
                best = (Tk, inl)
        if best[1].sum() < max(15, 0.4 * len(A)):
            return None
        return kabsch(A[best[1]], B[best[1]])
