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

import os
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
        self.tree = cKDTree(self.icp_pts)
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
    valid: bool = True
    held: str | None = None
    T_grip_obj: np.ndarray | None = None
    grip_at_grasp: float = 1.0
    last_res: tuple = (np.nan, 0)
    history: list = field(default_factory=list)
    lost: int = 0
    relocs: int = 0
    carrier: str | None = None
    support: dict = field(default_factory=dict)
    sil_iou: tuple = (np.nan, np.nan)
    mode: str = ""
    T_car_obj: np.ndarray | None = None


class CamView:
    """one camera at one tick: depth (m), K, T_world_cam, grey image (for flow), low-res renderer."""

    def __init__(self, name, D, K, Twc, grey=None, lowres=(180, 320), rgb=None, labels=None):
        self.name, self.D, self.K, self.Twc = name, D, np.asarray(K, float), np.asarray(Twc, float)
        self.Tcw = np.linalg.inv(self.Twc)
        self.grey = grey
        self.rgb = rgb
        self.labels = labels                     # object label image (any resolution): 0 none, i+1 = object i
        H, W = D.shape
        self.s = lowres[1] / W
        Kl = self.K.copy(); Kl[:2] *= self.s
        self.R = Renderer(Kl, int(round(H * self.s)), lowres[1])

    def masked_points(self, lid, max_n=1500, stride=2):
        """world points of the depth pixels whose label is `lid`."""
        if getattr(self,"depth_generated",False):
            P=self.associated_points.get(lid,np.zeros((0,3)))
            return P if len(P)<=max_n else P[np.random.default_rng(0).choice(len(P),max_n,replace=False)]
        if self.labels is None:
            return np.zeros((0, 3))
        H, W = self.D.shape
        lh, lw = self.labels.shape
        vv, uu = np.mgrid[0:H:stride, 0:W:stride]
        lab = self.labels[(vv * lh // H).clip(0, lh - 1), (uu * lw // W).clip(0, lw - 1)]
        d = self.D[vv, uu]
        m = (lab == lid) & (d > 0.1)
        if m.sum() == 0:
            return np.zeros((0, 3))
        u, v, z = uu[m].astype(float), vv[m].astype(float), d[m]
        P = np.stack([(u - self.K[0, 2]) / self.K[0, 0] * z, (v - self.K[1, 2]) / self.K[1, 1] * z, z], 1)
        P = P @ self.Twc[:3, :3].T + self.Twc[:3, 3]
        if len(P) > max_n:
            P = P[np.random.default_rng(0).choice(len(P), max_n, replace=False)]
        return P

    def mask_px(self, lid):
        return 0 if self.labels is None else int((self.labels == lid).sum())

    def zbuf(self, verts_world: torch.Tensor, faces):
        M = torch.tensor(self.Tcw, dtype=torch.float32, device="cuda")
        Vc = verts_world @ M[:3, :3].T + M[:3, 3]
        if (Vc[:,2]>.03).sum()<3:return torch.zeros((self.R.H,self.R.W),device=Vc.device)
        try:_, z = self.R.raster(Vc[None], faces)
        except IndexError:return torch.zeros((self.R.H,self.R.W),device=Vc.device)
        return z[0]


class MeshTracker:
    def __init__(self, objects: list, init_poses: dict, arm_model, cfg=None):
        """objects: LibObject list; init_poses {name: T_world_obj}; arm_model: TorchArm (for FK, gripper, occlusion)."""
        self.objs = {o.name: o for o in objects}
        self.state = {o.name: ObjState(np.asarray(init_poses[o.name], float),valid=getattr(o,"initial_valid",True)) for o in objects}
        self.arm = arm_model
        self.cfg = dict(contact=0.03, grasp_tip=0.02, grip_close=0.15, release_open=0.15, icp_iters=6, huber=0.005, held_max_corr=0.01, held_blend=0.1, held_blend_mask=0.3, mask_min_pts=60, mask_move_gate=0.002, cam_min_pts=40, sil_refine=os.environ.get("REALPC_SIL", "0") == "1", sil_steps=int(os.environ.get("REALPC_SIL_STEPS", "12")), sil_lr=0.07,
                        sil_min_px=40, sil_min_gain=0.01, cam_equal_weight=os.environ.get("REALPC_CAM_EQUAL", "0") == "1",
                        max_corr=0.03, free_tol=0.015, min_inl=40, move_gate=0.003, flow=True)
        self.cfg.update(cfg or {})
        self.cfg["depth_mesh"] = os.environ.get("REALPC_DEPTH_MESH","0")=="1"
        if self.cfg["depth_mesh"]:
            self.cfg.update(sil_refine=False,flow=False)
        self.prev = None                            # previous tick's views (for flow / track Kabsch)
        self.tick = -1
        self.previous_grip = {}
        self.first_tick = None
        self.initial_hypotheses = {}
        self.close_until = {}
        self.previous_G = {}
        self.contact_votes = {}
        self.carrier_votes = {}
        self.acquire_held = False
        self.lib_points, self.region, self.plane, self.bg_colors, self._cm = None, None, None, None, {}

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
            # finger pads = the finger link mesh vertices (the link origins sit inside the gripper body)
            fingers = [Vw[self.arm.vlink == self.arm.links.index(k)].cpu().numpy() for k in ("tip_left", "tip_right")]
            out[arm_] = dict(V=Vw, G=G, fingers=fingers, grip=float(q[arm_][6]))
        return out

    # -------------------------------------------------------------- main step
    def step(self, t, views: list, q: dict, T_world_base: dict, aux_views=None):
        """views: list of CamView for this tick (cameras whose content time ~ t). Returns {name: T_world_obj}."""
        self.tick = t
        if self.cfg["depth_mesh"]:self.cfg["sil_refine"]=any(v.labels is not None for v in views)
        rob = self.robot(q, T_world_base)
        # robot z-buffers per view (occlusion)
        robot_z = []
        for v in views:
            zs = [v.zbuf(r["V"], self.arm.faces) for r in rob.values()]
            z = torch.stack([torch.where(x > 0, x, torch.full_like(x, 1e9)) for x in zs]).amin(0)
            robot_z.append(z.cpu().numpy())
        if self.first_tick is None:
            self.first_tick=t
            self.acquire_held=bool(self.initial_hypotheses)
        for name,hypotheses in self.initial_hypotheses.items():
            if hypotheses:
                lid=list(self.objs).index(name)+1
                Tn,ib,ia=self.sil_refine(self.objs[name],self.state[name].T,views,robot_z,lid,
                                        alternatives=[np.asarray(H) for H in hypotheses])
                if Tn is not None and ia>.3:
                    self.state[name].T=Tn
        self.initial_hypotheses={}
        # Direct grasps require a recent closing event (or tight cold-start contact).
        # One gripper cannot independently rigidly attach the tool AND a nearby block.
        first_tick = not self.previous_grip
        for a,r in rob.items():
            if self.previous_grip.get(a,r["grip"]) - r["grip"] > .015:
                self.close_until[a] = t + 10
        busy = {st.held for st in self.state.values() if st.held is not None}
        # 1. prediction + grasp / release logic
        for name, st in self.state.items():
            o = self.objs[name]
            if not st.valid or getattr(o,"passive_fixture",False):continue
            P = st.T[:3, :3] @ o.samples.T
            P = (P.T + st.T[:3, 3])
            if st.held is not None:
                r = rob[st.held]
                if r["grip"] > st.grip_at_grasp + self.cfg["release_open"]:
                    busy.discard(st.held)
                    st.held, st.T_grip_obj = None, None              # released: stays where it is
                else:
                    st.T = r["G"] @ st.T_grip_obj
            else:
                tree = cKDTree(P)
                for arm_, r in rob.items():
                    if arm_ in busy:
                        continue
                    d = np.array([tree.query(f)[0].min() for f in r["fingers"]])
                    closing = t <= self.close_until.get(arm_, -1)
                    cold_contact = t-self.first_tick < 5 and d.max() < .015
                    if (closing or cold_contact) and d.max() < self.cfg["grasp_tip"] and r["grip"] < 1.0 - self.cfg["grip_close"]:
                        st.held = arm_
                        st.grip_at_grasp = r["grip"]
                        st.T_grip_obj = np.linalg.inv(r["G"]) @ st.T
                        busy.add(arm_)
                        break
        self.previous_grip = {a:r["grip"] for a,r in rob.items()}
        if "broom" in self.objs and os.environ.get("REALPC_BROOM_MATERIAL","0")=="1" and os.environ.get("REALPC_BROOM_CLEAN_MASK","0")=="1":
            from .broom_refine import BroomRefiner
            if not hasattr(self,"broom_refiner"):self.broom_refiner=BroomRefiner(self.objs["broom"])
            observed,_,_,_,_=self.broom_refiner.observe(self.state["broom"].T,views,robot_z)
            greens={v.name:v.labels==1 for v in observed};lid=list(self.objs).index("broom")+1
            for v,rz in zip(views,robot_z):
                if v.labels is None:continue
                shape=v.labels.shape;lab=v.labels.copy();zrobot=cv2.resize(rz,(shape[1],shape[0]),interpolation=0);depth=cv2.resize(v.D,(shape[1],shape[0]),interpolation=0)
                green=cv2.resize(greens.get(v.name,np.zeros(shape,bool)).astype(np.uint8),(shape[1],shape[0]),interpolation=0).astype(bool)
                foreground=(zrobot<1e8)&((depth<.1)|(zrobot<depth+.008))&~green
                lab[(lab==lid)&foreground]=254
                # Material identity + preceding/FK pose gates the added bristles.
                lab[green]=lid;v.labels=lab
        if self.cfg["depth_mesh"]:
            for st in self.state.values():
                if st.carrier is not None and st.T_car_obj is not None:
                    st.T=self.state[st.carrier].T@st.T_car_obj
        # object z-buffers (inter-object occlusion), with predicted poses
        obj_z = {}
        for name, st in self.state.items():
            o = self.objs[name]
            if not st.valid:
                obj_z[name]=[np.zeros((v.R.H,v.R.W),np.float32) for v in views];continue
            Tt = torch.tensor(st.T, dtype=torch.float32, device="cuda")
            Vw = o.verts_t @ Tt[:3, :3].T + Tt[:3, 3]
            obj_z[name] = [v.zbuf(Vw, o.faces_t).cpu().numpy() for v in views]
        if self.cfg["depth_mesh"]:
            from .depth_masks import associate_depth
            associate_depth(self,views,robot_z,obj_z)
        out = {}
        self.depth_contact={}
        Rv = torch.cat([r["V"] for r in rob.values()]).cpu().numpy()[::7]
        prevT = {n: (st.history[-1] if st.history else st.T).copy() for n, st in self.state.items()}
        moved = {}                                                   # objects already updated this tick: delta pose
        order = sorted(self.state, key=lambda n: self.state[n].held is None)      # held objects first
        for name in order:
            st = self.state[name]
            o = self.objs[name]
            if not st.valid:
                st.mode="unobserved";out[name]=np.full((4,4),np.nan);continue
            if getattr(o,"passive_fixture",False):
                st.mode="passive_fixture";st.held=None;st.carrier=None
                if name=="box" and getattr(st,"allow_fixture_update",True) and (t-self.first_tick)%5==0:
                    from .box_fixture import refine_box_silhouette
                    st.T,self.box_silhouette=refine_box_silhouette(o,st.T,views,robot_z)
                st.last_res=self.residual(o,st.T,views,robot_z,obj_z,name)
                out[name]=st.T.copy();st.history.append(st.T.copy());continue
            Pw = (st.T[:3, :3] @ o.samples.T).T + st.T[:3, 3]
            tree = cKDTree(Pw)
            near_robot = tree.query(Rv, distance_upper_bound=self.cfg["contact"])[0].min() < self.cfg["contact"]
            self.depth_contact[name]=near_robot
            # object-object contact: a neighbour that moved this tick may push / carry this object
            carriers = []
            for other, dT in moved.items():
                if np.linalg.norm(dT[:3, 3]) < 0.002:
                    continue
                So = self.state[other].T
                Q = (So[:3, :3] @ self.objs[other].samples[::3].T).T + So[:3, 3]
                dmin = tree.query(Q, distance_upper_bound=0.02)[0].min()
                if dmin < 0.02:
                    carriers.append(dT)
                    # Contact generates a motion hypothesis only. Actual carrier
                    # attachment is confirmed below from observed co-motion.
            if st.carrier is not None:
                car = self.state[st.carrier]
                if car.held is None and np.linalg.norm(moved.get(st.carrier, np.eye(4))[:3, 3]) < 0.002:
                    st.carrier, st.T_car_obj = None, None                       # carrier put down: object rests
                else:
                    st.T = car.T @ st.T_car_obj
            # ---- mask-backed update (object masks from the causal segmenter): the main path
            lid = list(self.objs).index(name) + 1
            # joint multi-camera measurement: each camera with enough visible support gets EQUAL total weight (an occluded
            # camera drops out; the other camera(s) carry the pose); none -> held FK / carrier / previous pose
            Ms = [v.masked_points(lid) for v in views] if any(v.labels is not None for v in views) else []
            sup = {v.name: len(m_) for v, m_ in zip(views, Ms)}
            use_v = [m_ for m_ in Ms if len(m_) >= self.cfg["cam_min_pts"]]
            M = np.concatenate(use_v) if use_v else np.zeros((0, 3))
            Mw = np.concatenate([np.full(len(m_), 1.0 / len(m_)) for m_ in use_v]) * len(M) / max(1, len(use_v)) if use_v else np.zeros(0)
            if not self.cfg["cam_equal_weight"]:
                Mw = np.ones(len(M))
            st.support = sup
            st.mode = ("+".join(v.name for v, m_ in zip(views, Ms) if len(m_) >= self.cfg["cam_min_pts"]) or
                       ("fk" if st.held is not None else ("carrier" if st.carrier is not None else "hold")))
            mask_prediction = st.T.copy()
            depth_observation=self.cfg["depth_mesh"] and all(getattr(v,"depth_generated",False) for v in views)
            depth_distance=o.tree.query((M-st.T[:3,3])@st.T[:3,:3])[0] if depth_observation and len(M) else np.zeros(0)
            stationary_fit=len(depth_distance)>=self.cfg["mask_min_pts"] and np.median(depth_distance)<.004 and np.percentile(depth_distance,80)<.008
            if depth_observation and st.held is None and st.carrier is None and not carriers and (stationary_fit or (not near_robot and len(M)<self.cfg["mask_min_pts"])):
                # A resting object should not chase depth noise or slide along an
                # underconstrained plane. Current ray evidence still detects loss.
                if len(M):
                    dd=depth_distance
                    st.last_res=(float(np.median(dd)),len(dd))
                else:st.last_res=(np.nan,0)
                st.mode="rest_depth";moved[name]=np.eye(4);out[name]=st.T.copy();st.history.append(st.T.copy());continue
            if len(M) >= self.cfg["mask_min_pts"] and st.held is None:
                nearest = cKDTree(o.samples @ st.T[:3,:3].T + st.T[:3,3]).query(M)[0]
                if np.median(nearest) > .035 and (st.carrier is not None or st.lost >= 2):
                    # Mask identity comes from current causal segmentation. A physical
                    # carrier relationship is revoked when both geometry and imagery disagree.
                    from .cold_pose import register_components
                    Pl, Cl = self.lib_points[name] if self.lib_points is not None else (o.icp_pts,np.full_like(o.icp_pts,.5))
                    Tr, fit, rmse, _ = register_components(Pl,Cl,M,np.full_like(M,.5))
                    if fit > .3 and rmse < .009:
                        st.T = Tr; st.carrier = None; st.T_car_obj = None; st.relocs += 1
                        mask_prediction = Tr.copy()
            if len(M) >= self.cfg["mask_min_pts"]:
                if st.held is not None:
                    G = rob[st.held]["G"]
                    T1, res = self.icp_masked(o, st.T, M, w=Mw)
                    dT = T1 @ np.linalg.inv(st.T)
                    dang = np.degrees(np.arccos(np.clip((np.trace(dT[:3, :3]) - 1) / 2, -1, 1)))
                    if res[1] >= self.cfg["mask_min_pts"] and np.linalg.norm(T1[:3, 3] - st.T[:3, 3]) < 0.03 and dang < 10:
                        T_new = np.linalg.inv(G) @ T1
                        a_ = self.cfg["held_blend_mask"]
                        Tb = st.T_grip_obj.copy()
                        Tb[:3, 3] = (1 - a_) * Tb[:3, 3] + a_ * T_new[:3, 3]
                        U, _, Vt = np.linalg.svd((1 - a_) * Tb[:3, :3] + a_ * T_new[:3, :3])
                        Tb[:3, :3] = U @ Vt
                        st.T_grip_obj = Tb
                        st.T = G @ Tb
                    st.last_res = res
                else:
                    cands = [st.T.copy()]
                    if st.carrier is None:
                        cands += [dT @ st.T for dT in carriers]
                    if self.cfg["flow"] and self.prev is not None and st.carrier is None:
                        Tk = self.track_kabsch(o, st.T, views, self.prev, robot_z)
                        if Tk is not None:
                            cands.append(Tk @ st.T)
                    best = None
                    for T0 in cands:
                        T1, res = self.icp_masked(o, T0, M, w=Mw)
                        if res[1] >= self.cfg["mask_min_pts"] and (best is None or res[0] < best[1][0]):
                            best = (T1, res)
                    if best is None:                                  # mask present, pose far off: re-localize
                        st.lost += 1
                        if st.lost >= 3 and (st.lost - 3) % 15 == 0 and len(M) >= 300 and self.lib_points is not None:
                            from . import objlib
                            Pl, Cl = self.lib_points[o.name]
                            Tr, fit, rmse = objlib.register(Pl, Cl, M, np.full((len(M), 3), 0.5), voxel=0.004, seed=self.tick)
                            if fit > 0.4 and rmse < 0.005:
                                st.T = Tr; st.lost = 0; st.relocs += 1; st.carrier = None
                        st.last_res = (np.nan, 0)
                    else:
                        T1, res = best
                        st.lost = 0
                        dT = T1 @ np.linalg.inv(st.T)
                        dmove = np.linalg.norm(T1[:3, 3] - st.T[:3, 3])
                        dang = np.degrees(np.arccos(np.clip((np.trace(dT[:3, :3]) - 1) / 2, -1, 1)))
                        # move on clear evidence only (no jitter of resting objects), never by jumps
                        if (dmove > self.cfg["mask_move_gate"] or dang > 1.5) and dmove < 0.08:
                            st.T = T1
                            if st.carrier is not None:
                                st.T_car_obj = np.linalg.inv(self.state[st.carrier].T) @ st.T
                        st.last_res = res
                # per-tick silhouette refinement against the masks of every camera that sees the object
                if self.cfg["sil_refine"]:
                    Tn, ib, ia = self.sil_refine(o, st.T, views, robot_z, lid, alternatives=[mask_prediction])
                    if Tn is not None and ia > ib + self.cfg["sil_min_gain"]:
                        if st.held is not None:
                            G = rob[st.held]["G"]
                            st.T_grip_obj = np.linalg.inv(G) @ Tn
                        elif st.carrier is not None:
                            st.T_car_obj = np.linalg.inv(self.state[st.carrier].T) @ Tn
                        st.T = Tn
                    st.sil_iou = (ib, ia)
                moved[name] = st.T @ np.linalg.inv(prevT[name])
                out[name] = st.T.copy()
                st.history.append(st.T.copy())
                continue
            if any(v.labels is not None for v in views):
                if self.cfg["sil_refine"]:
                    Tn, ib, ia = self.sil_refine(o, st.T, views, robot_z, lid)
                    if Tn is not None and ia > ib + self.cfg["sil_min_gain"]:
                        st.T = Tn
                        if st.held is not None:
                            st.T_grip_obj = np.linalg.inv(rob[st.held]["G"]) @ Tn
                        elif st.carrier is not None:
                            st.T_car_obj = np.linalg.inv(self.state[st.carrier].T) @ Tn
                    st.sil_iou = (ib, ia)
                # Both masks absent: preserve calibrated FK/carrier/last-pose prediction.
                moved[name] = st.T @ np.linalg.inv(prevT[name])
                out[name] = st.T.copy()
                st.history.append(st.T.copy())
                continue

            moving = (st.held is not None or near_robot or bool(carriers)) and st.carrier is None
            if moving:
                cands = [st.T.copy()]
                if st.held is None:
                    cands += [dT @ st.T for dT in carriers]
                    if self.cfg["flow"] and self.prev is not None:
                        Tk = self.track_kabsch(o, st.T, views, self.prev, robot_z)
                        if Tk is not None:
                            cands.append(Tk @ st.T)
                best = None
                for T0 in cands:
                    T1, res = self.icp(o, T0, views, robot_z, obj_z, name)
                    if res[1] >= self.cfg["min_inl"] and (best is None or res[0] < best[1][0]):
                        best = (T1, res)
                if best is not None:
                    T1, res = best
                    dT = T1 @ np.linalg.inv(st.T)
                    dmove = np.linalg.norm(T1[:3, 3] - st.T[:3, 3])
                    dang = np.degrees(np.arccos(np.clip((np.trace(dT[:3, :3]) - 1) / 2, -1, 1)))
                    if st.held is not None:
                        # held: the gripper FK carries the object; ICP may only correct small slip (<= 1 cm, 5 deg),
                        # blended slowly into the grasp offset (no drift from a wrong ICP lock)
                        if dmove < self.cfg["held_max_corr"] and dang < 5.0:
                            G = rob[st.held]["G"]
                            T_new = np.linalg.inv(G) @ T1
                            a_ = self.cfg["held_blend"]
                            Tb = st.T_grip_obj.copy()
                            Tb[:3, 3] = (1 - a_) * Tb[:3, 3] + a_ * T_new[:3, 3]
                            U, _, Vt = np.linalg.svd((1 - a_) * Tb[:3, :3] + a_ * T_new[:3, :3])
                            Tb[:3, :3] = U @ Vt
                            st.T_grip_obj = Tb
                            st.T = G @ Tb
                    elif dmove > self.cfg["move_gate"] and dmove < 0.05:
                        # free objects move only on clear evidence (contact + ICP), and not by jumps
                        st.T = T1
                    st.last_res = res
                else:
                    st.last_res = self.residual(o, st.T, views, robot_z, obj_z, name)
            else:
                st.last_res = self.residual(o, st.T, views, robot_z, obj_z, name)
            # lost: we look through where a free object should be -> causal re-localization (swept / carried objects)
            if st.held is None:
                vio = self.see_through(o, st.T, views, robot_z, obj_z, name)
                st.lost = st.lost + 1 if vio > 0.5 else 0
                if st.lost >= 3 and (st.lost - 3) % 5 == 0:
                    excl = [(self.state[n].T[:3, :3] @ self.objs[n].samples.T).T + self.state[n].T[:3, 3] for n in self.state if n != name]
                    Tr = self.relocalize(o, views, rob, excl)
                    if Tr is not None:
                        st.T = Tr; st.lost = 0; st.relocs += 1; st.carrier = None; st.T_car_obj = None
            moved[name] = st.T @ np.linalg.inv(prevT[name])
            out[name] = st.T.copy()
            st.history.append(st.T.copy())
        if "broom" in self.objs and os.environ.get("REALPC_BROOM_MATERIAL","0")=="1" and (not self.cfg["depth_mesh"] or ((t-self.first_tick)%4==0 and (self.state["broom"].held is not None or self.depth_contact.get("broom",False) or t==self.first_tick))):
            from .broom_refine import BroomRefiner
            if not hasattr(self,"broom_refiner"):self.broom_refiner=BroomRefiner(self.objs["broom"])
            extras=aux_views or [];extra_z=[]
            for v in extras:
                zz=[v.zbuf(r["V"],self.arm.faces) for r in rob.values()]
                extra_z.append(torch.stack([torch.where(z>0,z,torch.full_like(z,1e9)) for z in zz]).amin(0).cpu().numpy())
            st=self.state["broom"];Tn,detail=self.broom_refiner.refine(self,st.T,views+extras,robot_z+extra_z)
            self.broom_refiner.stats=detail;st.T=Tn
            if st.held is not None:st.T_grip_obj=np.linalg.inv(rob[st.held]["G"])@Tn
            st.history[-1]=Tn.copy();out["broom"]=Tn.copy()
        # A block near a moving tool is not necessarily riding it. Require observed
        # contact and co-motion over three ticks before assigning a rigid carrier.
        for name,st in self.state.items():
            if getattr(self.objs[name],"passive_fixture",False) or st.held is not None or st.carrier is not None or sum(st.support.values())<60:
                continue
            o=self.objs[name];W=o.samples@st.T[:3,:3].T+st.T[:3,3];tree=cKDTree(W)
            for parent,car in self.state.items():
                if parent==name or car.held is None or np.linalg.norm(o.mesh.extents)>=.5*np.linalg.norm(self.objs[parent].mesh.extents):
                    continue
                pred=car.T@np.linalg.inv(prevT[parent])@prevT[name]
                Q=self.objs[parent].samples[::3]@car.T[:3,:3].T+car.T[:3,3]
                near=tree.query(Q)[0].min()<.020
                moving=np.linalg.norm(st.T[:3,3]-prevT[name][:3,3])>.0015
                agrees=np.linalg.norm(st.T[:3,3]-pred[:3,3])<.008
                key=(name,parent);self.carrier_votes[key]=self.carrier_votes.get(key,0)+1 if near and moving and agrees else 0
                if self.carrier_votes[key]>=3:
                    st.carrier=parent;st.T_car_obj=np.linalg.inv(car.T)@st.T;break
        # During cold acquisition, a partly hidden handle may miss the tight initial
        # contact gate. Confirm attachment from three current/past co-motion observations.
        if self.acquire_held and t-self.first_tick<60 and self.previous_G:
            busy={s.held for s in self.state.values() if s.held is not None}
            for name,st in self.state.items():
                if getattr(self.objs[name],"passive_fixture",False) or st.held is not None or st.carrier is not None or sum(st.support.values())<60:
                    continue
                W=self.objs[name].samples@st.T[:3,:3].T+st.T[:3,3];tree=cKDTree(W)
                for arm_,r in rob.items():
                    if arm_ in busy or r["grip"]>=.85:continue
                    oldG=self.previous_G[arm_];G=r["G"]
                    U=np.linalg.inv(G)@st.T;U0=np.linalg.inv(oldG)@prevT[name]
                    near=max(tree.query(f)[0].min() for f in r["fingers"])<.035
                    moving=np.linalg.norm(G[:3,3]-oldG[:3,3])>.0015 and np.linalg.norm(st.T[:3,3]-prevT[name][:3,3])>.001
                    consistent=np.linalg.norm(U[:3,3]-U0[:3,3])<.008 and np.linalg.norm(U[:3,:3]-U0[:3,:3])<.15
                    key=(name,arm_);self.contact_votes[key]=self.contact_votes.get(key,0)+1 if near and moving and consistent else 0
                    if self.contact_votes[key]>=3:
                        st.held=arm_;st.grip_at_grasp=r["grip"];st.T_grip_obj=U;busy.add(arm_);break
        self.previous_G={a:r["G"].copy() for a,r in rob.items()}
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

    def sil_refine(self, o, T, views, robot_z, lid, alternatives=()):
        """a few Adam steps on a 6-DoF correction (object frame) maximising the support-weighted soft-silhouette IoU
        with the object's mask in every camera (robot pixels and other objects ignored), prior to the current pose."""
        if os.environ.get("REALPC_SIL_METHOD", "adam") == "contour":
            from .joint_pose import refine_contours
            return refine_contours(o, T, views, robot_z, lid, alternatives)
        from .diffcalib import se3_exp, soft_iou
        data = []
        for vi, v in enumerate(views):
            if v.labels is None:
                continue
            h, w = v.R.H, v.R.W
            lab = cv2.resize(v.labels, (w, h), interpolation=cv2.INTER_NEAREST)
            m = lab == lid
            if m.sum() < self.cfg["sil_min_px"]:
                continue
            rz = robot_z[vi]
            ign = ((lab > 0) & (lab != lid)) | (np.isfinite(rz) & (rz < 1e8) & ~m)
            Tcw = torch.tensor(v.Tcw, dtype=torch.float32, device="cuda")
            data.append((v, torch.tensor(m, dtype=torch.float32, device="cuda")[None],
                         torch.tensor(~ign, dtype=torch.float32, device="cuda")[None], Tcw, min(1.0, m.sum() / 300.0)))
        if not data:
            return None, 0.0, 0.0
        T0 = torch.tensor(T, dtype=torch.float32, device="cuda")
        # Dimensionless parameters: Adam must not give metres and radians the same step.
        d = torch.zeros(6, device="cuda", requires_grad=True)
        units = torch.tensor([.15, .15, .15, .02, .02, .02], device="cuda")
        opt = torch.optim.Adam([d], lr=self.cfg["sil_lr"])
        V = o.verts_t

        def loss_iou(dd, sigma):
            Tn = T0 @ se3_exp(dd * units)
            Vw = V @ Tn[:3, :3].T + Tn[:3, 3]
            tot, wsum, ious = 0.0, 0.0, []
            for v, m, val, Tcw, wc in data:
                sil, _ = v.R.raster((Vw @ Tcw[:3, :3].T + Tcw[:3, 3])[None], o.faces_t,
                                    sigma=sigma, fpp=8 if sigma else 1)
                iou = soft_iou(sil, m, val)[0]
                tot = tot + wc * (1 - iou); wsum += wc; ious.append(float(iou.detach()))
            return tot / wsum, float(np.average(ious, weights=[x[4] for x in data])), ious
        with torch.no_grad():
            _, i0, per0 = loss_iou(d, None)
        # Reject ICP hypotheses which fit planar depth but lose the two-view silhouette.
        original_i0 = i0
        for alt in alternatives:
            saved = T0
            T0 = torch.tensor(alt, dtype=torch.float32, device="cuda")
            with torch.no_grad():
                _, ai, ap = loss_iou(d, None)
            if ai > i0:
                i0, per0 = ai, ap
            else:
                T0 = saved
        best_i, best_d = i0, d.detach().clone()
        for it in range(self.cfg["sil_steps"]):
            # A large blur with a finite faces-per-pixel budget loses interior faces on
            # dense thin meshes. 1e-6 reproduces their actual hard silhouette.
            l, _, _ = loss_iou(d, 1e-6)
            l = l + 0.003 * (d ** 2).sum()
            opt.zero_grad(); l.backward(); opt.step()
            with torch.no_grad():
                d.clamp_(-1.5, 1.5)
                if (it + 1) % 4 == 0 or it == self.cfg["sil_steps"] - 1:
                    _, ih, per = loss_iou(d, None)
                    # Do not trade a large regression in a well-observed camera for
                    # a gain in the other. Weak / occluded views are omitted above.
                    if ih > best_i and all(q >= p - .08 for q, p in zip(per, per0)):
                        best_i, best_d = ih, d.detach().clone()
        with torch.no_grad():
            Tn = (T0 @ se3_exp(best_d * units)).cpu().numpy().astype(float)
        return Tn, original_i0, best_i

    def icp_masked(self, o, T, M, iters=10, w=None):
        """data -> model point-to-plane ICP of the object's masked depth points M (world) onto the mesh (object frame
        KD-tree), Huber; right-multiplied update in the object frame. -> T, (median |r|, n)"""
        T = T.copy()
        res = (np.nan, 0)
        if self.cfg["depth_mesh"]:iters=min(iters,5)
        for it in range(iters):
            Ti = np.linalg.inv(T)
            Mo = M @ Ti[:3, :3].T + Ti[:3, 3]
            dist, j = o.tree.query(Mo)
            gate = 0.03 if it < 3 else 0.015
            ok = dist < gate
            if ok.sum() < 20:
                return T, (np.nan, int(ok.sum()))
            p, n, x = o.icp_pts[j[ok]], o.icp_nrm[j[ok]], Mo[ok]
            wc = np.ones(ok.sum()) if w is None else w[ok]
            r = ((x - p) * n).sum(1)
            # model point p moves by dT: p' = p + w x p + v  -> residual r - n.(w x p + v)
            J = np.c_[np.cross(p, n), n]
            hub = self.cfg["huber"]
            wt = np.where(np.abs(r) < hub, 1.0, hub / np.maximum(np.abs(r), 1e-9)) * wc
            if self.cfg["depth_mesh"]:
                # Metric scaling makes rotational and translational observability
                # comparable. Preserve the prediction in unconstrained modes
                # (e.g. yaw of a pole, tangential motion of a planar patch).
                radius=max(.02,float(np.sqrt(np.mean(np.sum(o.icp_pts**2,axis=1)))))
                scale=np.r_[np.full(3,1/radius),np.ones(3)]
                A=J*scale
                H=(A*wt[:,None]).T@A
                eig,U=np.linalg.eigh(H);g=(A*wt[:,None]).T@r
                inv=np.where(eig>max(1e-8,.015*eig[-1]),1/np.maximum(eig,1e-8),0)
                xi=scale*(U@(inv*(U.T@g)))
                factor=max(1.,np.linalg.norm(xi[:3])/np.deg2rad(5),np.linalg.norm(xi[3:])/.02)
                xi/=factor
            else:
                H = (J * wt[:, None]).T @ J + 1e-5 * np.eye(6)
                xi = np.linalg.solve(H, (J * wt[:, None]).T @ r)
            T = T @ se3_small(xi)
            res = (float(np.median(np.abs(r))), int(ok.sum()))
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
            nv = (d > 0.1).sum(); nf = ((d > 0.1) & (d > z + 0.02)).sum()
            self._viol = (getattr(self, "_viol", (0, 0))[0] + nf, getattr(self, "_viol", (0, 0))[1] + nv)
        rr = np.concatenate(rr) if rr else np.zeros(0)
        return (float(np.median(rr)) if len(rr) else np.nan, int(len(rr)))

    def see_through(self, o, T, views, robot_z, obj_z, name):
        """share of the object's visible samples (with depth) whose pixel shows a surface >= 2 cm BEHIND them."""
        self._viol = (0, 0)
        self.residual(o, T, views, robot_z, obj_z, name)
        nf, nv = self._viol
        return nf / nv if nv >= 30 else 0.0

    def relocalize(self, o, views, rob, exclude):
        """lost object: colour-gated global registration of the library points into the current cloud (robot and the
        other objects removed). -> T or None."""
        from . import objlib
        if self.lib_points is None or o.name not in self.lib_points:
            return None
        Pl, Cl = self.lib_points[o.name]
        Ps, Cs = [], []
        for v in views:
            if v.rgb is None:
                continue
            P, C, _, _ = objlib.depth_points(v.D, v.rgb, v.K, v.Twc, stride=2)
            Ps.append(P); Cs.append(C)
        if not Ps:
            return None
        P, C = np.concatenate(Ps), np.concatenate(Cs)
        if self.region is not None:
            k = objlib.table_crop(P, self.region, *self.plane, above=0.003)
            P, C = P[k], C[k]
        R = torch.cat([r["V"] for r in rob.values()]).cpu().numpy()[::3]
        keep = objlib.robot_free(P, R, 0.02)
        for X in exclude:
            keep &= objlib.robot_free(P, X, 0.01)
        P, C = P[keep], C[keep]
        if o.name not in self._cm:
            self._cm[o.name] = objlib.colour_model(Cl, np.zeros((0, 3)) if self.bg_colors is None else self.bg_colors)
        pr = self._cm[o.name](C)
        cand = pr > 0.5
        if cand.sum() < 40:
            return None
        T, fit, rmse = objlib.register(Pl, Cl, P[cand], C[cand], voxel=0.004, seed=self.tick)
        if fit > 0.5 and rmse < 0.004:
            return T
        return None

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
