"""v4 converter: robot FK (calibrated, joint offsets) + library objects tracked as 6-DoF poses + static scene.

`StreamConverter` is the one code path for training-data conversion AND live inference: it is fed one tick at a time
(top + side RGB-D and the joint vector), uses nothing from the future, and can start at any tick (cold start: static
scene and object poses come from the first tick it sees). The batch converter (scripts/convert_v4.py) just loops it.

Saved episode format (compact, lossless): per-body poses + fixed local samples; `realpc.dataset.load` expands it into
points [T, N, 3] with fixed identity. See `SCHEMA` below.
"""
from __future__ import annotations

import glob
import json
import os

import cv2
import numpy as np
import trimesh

from . import objlib
from .meshtrack import CamView, LibObject, MeshTracker

SCHEMA = {
    "ticks": "[T] int, video/low_dim tick of each output step",
    "t_cap": "[T] float64, capture time (s)",
    "q_left / q_right": "[T,7] joints (+ calibrated zero offsets) + normalised gripper (1 = open), at the robot tick the "
                        "top image shows (tick - lag)",
    "low_<key>": "recorded low_dim arrays at the output ticks: eef_pos/quat, action_*, gripper, engaged (both arms)",
    "robot_local": "[Nr,3] float32 link-frame surface samples (URDF meshes, fingers included)",
    "robot_body": "[Nr] int, index into robot_bodies", "robot_bodies": "[B] 'arm/link' names",
    "robot_T": "[T,B,4,4] float32 world (left arm base) pose of every robot link",
    "obj_names": "[P]", "obj_local": "[No,3] float32 object-frame mesh samples", "obj_index": "[No] int object of each sample",
    "obj_color": "[No,3] uint8", "obj_T": "[P,T,4,4] float32 world pose", "obj_res": "[P,T,2] median |depth residual| (m), n",
    "obj_held": "[P,T] int8 0 none / 1 left / 2 right", "obj_valid": "[P,T] bool (pose defined)",
    "static_points": "[Ns,3] float32 static scene (table region) at the first output tick", "static_color": "[Ns,3] uint8",
    "cloth_points": "[Nc,T,3] float32 (greycloth only, cloth v1 tracker)", "cloth_valid": "[T] bool (pre-fold windows only)",
    "groups": "0 robot_left, 1 robot_right, 2 static, 3 library object, 4 cloth (expanded by realpc.dataset.load)",
}


def load_library(lib_dir, n_samples=1500):
    objs, metas = [], {}
    for f in sorted(glob.glob(os.path.join(lib_dir, "*.ply"))):
        meta = json.load(open(f[:-4] + ".json"))
        objs.append(LibObject(meta["name"], trimesh.load(f, process=False), n_samples=n_samples))
        metas[meta["name"]] = meta
    return objs, metas


class StreamConverter:
    def __init__(self, ep, cal, tarm, arms, lib_dir, start=0, cold=False, side_max_lag=2.5, static_voxel=0.008, seg_dir=None):
        """ep: Episode (intrinsics, table region, recorded low_dim); cal: RigCalib; tarm: TorchArm; arms: BimanualArms."""
        self.ep, self.cal, self.tarm, self.arms = ep, cal, tarm, arms
        self.T_world_base = {"left": np.eye(4), "right": cal.T_world_rb()}
        self.arms.T_base["right"] = self.T_world_base["right"]
        self.objs, self.metas = load_library(lib_dir)
        self.lib_dir, self.start, self.cold = lib_dir, start, cold
        self.side_max_lag, self.static_voxel = side_max_lag, static_voxel
        self.trk = None
        self.static = None
        self.key = f"{ep.task}/{ep.name}"
        self.init_info = {}
        # object masks of the causal segmenter (scripts/segment_objects.py); offline replay of what runs live
        self.seg = {}
        if seg_dir:
            for c in ("top", "side"):
                p = os.path.join(seg_dir, ep.task, f"{ep.name}_{c}.npz")
                if os.path.exists(p):
                    z = np.load(p)
                    if int(z["start"]) <= start:
                        self.seg[c] = (z["labels"], int(z["start"]))

    # ------------------------------------------------------------------ first tick: object poses + static scene
    def _init(self, t, frames):
        init = {}
        Ps, Cs = [], []
        region = json.load(open(os.path.join(self.ep.path, "table_region.json")))
        for c, (D, im) in frames.items():
            P, C, _, _ = objlib.depth_points(D, im, self.ep.intrinsics(c), self.cal.T_world_cam(c), stride=2)
            k = objlib.table_crop(P, region, *self.ep.table_plane(), above=-0.01)
            Ps.append(P[k]); Cs.append(C[k])
        P, C = np.concatenate(Ps), np.concatenate(Cs)
        for o in self.objs:
            T0 = None
            if not self.cold and t == 0:
                for sc in self.metas[o.name]["scans"]:
                    if sc["episode"] == self.key:
                        T0 = np.asarray(sc["T_world_obj_t0"], float)
            if T0 is None:
                pts = np.load(os.path.join(self.lib_dir, f"{o.name}_points.npz"))
                T0, fit, rmse = objlib.register(pts["P"].astype(float), pts["C"].astype(float), P, C, voxel=0.005)
                self.init_info[o.name] = {"cold_registration_fitness": fit, "rmse_mm": rmse * 1000}
            else:
                self.init_info[o.name] = {"library_scan_pose": True}
            init[o.name] = T0
        self.trk = MeshTracker(self.objs, init, self.tarm)
        self.trk.lib_points = {o.name: tuple(np.load(os.path.join(self.lib_dir, f"{o.name}_points.npz"))[k].astype(float) for k in ("P", "C"))
                               for o in self.objs}
        self.trk.region, self.trk.plane = region, self.ep.table_plane()
        # static scene = table-region points not explained by the robot or a library object
        q = {a: self.cal.q_at(self.ep, a, self.cal.robot_tick(self.ep, "top", t)) for a in ("left", "right")}
        R = self.arms.points(q)
        keep = objlib.robot_free(P, R, 0.03)
        for o in self.objs:
            S = o.icp_pts @ init[o.name][:3, :3].T + init[o.name][:3, 3]
            keep &= objlib.robot_free(P, S, 0.012)
        P, C = P[keep], C[keep]
        kk = np.floor(P / self.static_voxel).astype(np.int64)
        _, idx = np.unique(kk, axis=0, return_index=True)
        self.static = (P[idx].astype(np.float32), (np.clip(C[idx], 0, 1) * 255).astype(np.uint8))
        self.trk.bg_colors = C[idx][::3]                       # static-scene colours: negatives of the colour models

    # ------------------------------------------------------------------ per tick
    def step(self, t, frames):
        """frames: {"top": (depth_m, bgr), "side": (depth_m, bgr)} of tick t. -> dict for this tick."""
        if self.trk is None:
            self._init(t, frames)
        def lab(c):
            if c not in self.seg:
                return None
            L, s0 = self.seg[c]
            return L[t - s0] if 0 <= t - s0 < len(L) else None
        views = [CamView("top", frames["top"][0], self.ep.intrinsics("top"), self.cal.T_world_cam("top"),
                         grey=cv2.cvtColor(frames["top"][1], cv2.COLOR_BGR2GRAY), rgb=frames["top"][1], labels=lab("top"))]
        # side frames are used when their content time matches the top camera's (same lag); both cameras then show the
        # robot at robot_tick(top) = t - lag
        if "side" in frames and abs(self.cal.lag(self.ep, "side", t) - self.cal.lag(self.ep, "top", t)) <= self.side_max_lag:
            views.append(CamView("side", frames["side"][0], self.ep.intrinsics("side"), self.cal.T_world_cam("side"), lowres=(240, 320),
                                 rgb=frames["side"][1], labels=lab("side")))
        tr = self.cal.robot_tick(self.ep, "top", t)
        q = {a: self.cal.q_at(self.ep, a, tr) for a in ("left", "right")}
        poses = self.trk.step(t, views, q, self.T_world_base)
        out = {"q": q, "robot_T": self.arms.link_poses(q).astype(np.float32), "obj_T": {}, "obj_res": {}, "obj_held": {}}
        for o in self.objs:
            st = self.trk.state[o.name]
            out["obj_T"][o.name] = poses[o.name].astype(np.float32)
            out["obj_res"][o.name] = st.last_res
            out["obj_held"][o.name] = 0 if st.held is None else (1 if st.held == "left" else 2)
            out.setdefault("obj_relocs", {})[o.name] = st.relocs
            if st.carrier is not None and st.held is None:
                out["obj_held"][o.name] = 3                       # carried by another (held) object
        return out

    def points(self, out):
        """points of one tick (robot | objects | static), fixed order -> [N,3] (what a live consumer uses)."""
        R = self.arms.points(None, poses=out["robot_T"].astype(float))
        O = [o.samples @ out["obj_T"][o.name][:3, :3].T + out["obj_T"][o.name][:3, 3] for o in self.objs]
        return np.concatenate([R] + O + [self.static[0]]).astype(np.float32)
