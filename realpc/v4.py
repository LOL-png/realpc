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
    "obj_names": "[P]", "obj_static": "[P] optional bool; immutable library fixtures only (the movable tray is false)", "obj_local": "[No,3] float32 object-frame mesh samples", "obj_index": "[No] int object of each sample",
    "obj_color": "[No,3] uint8", "obj_T": "[P,T,4,4] float32 world pose", "obj_res": "[P,T,2] median |depth residual| (m), n",
    "obj_held": "[P,T] int8 0 none / 1 left / 2 right / 3 carrier", "obj_valid": "[P,T] bool (pose defined)",
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
        objs[-1].metadata = meta
        objs[-1].passive_fixture = bool(meta.get("passive_fixture", False))
    return objs, metas


class StreamConverter:
    def __init__(self, ep, cal, tarm, arms, lib_dir, start=0, cold=True, side_max_lag=2.5, static_voxel=0.008, seg_dir=None):
        """ep: Episode (intrinsics, table region, recorded low_dim); cal: RigCalib; tarm: TorchArm; arms: BimanualArms."""
        # Tested production defaults; explicit environment values retain ablation support.
        for key,value in {"REALPC_SIL":"1","REALPC_CAM_EQUAL":"1","REALPC_SIL_METHOD":"contour","REALPC_COLD_COMPONENTS":"1","REALPC_BROOM_MATERIAL":"1","REALPC_BROOM_CLEAN_MASK":"0","REALPC_BROOM_HOLE":"0","REALPC_WRIST_WEIGHT":"1","REALPC_BROOM_REJECT_BG":"0"}.items():os.environ.setdefault(key,value)
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
        self.init_hypotheses = {}
        prior_path=os.path.join(lib_dir,"grasp_priors.json")
        self.grasp_priors=json.load(open(prior_path)).get("grasps",{}) if os.path.exists(prior_path) else {}
        # object masks of the causal segmenter (scripts/segment_objects.py); offline replay of what runs live
        self.seg = {}
        if seg_dir:
            for c in ("top", "side"):
                p = os.path.join(seg_dir, ep.task, f"{ep.name}_{c}.npz")
                if os.path.exists(p):
                    z = np.load(p)
                    if int(z["start"]) <= start:
                        old_names=list(z["names"]); new_names=[o.name for o in self.objs]
                        lookup=np.arange(256,dtype=np.uint8)
                        for i,n in enumerate(old_names,1):lookup[i]=new_names.index(n)+1 if n in new_names else 0
                        self.seg[c] = (lookup[z["labels"]], int(z["start"]))

    # ------------------------------------------------------------------ first tick: object poses + static scene
    def _init(self, t, frames, q=None, labels=None):
        init = {}
        Ps, Cs = [], []
        region = json.load(open(os.path.join(self.ep.path, "table_region.json")))
        for c, (D, im) in frames.items():
            if c not in ("top","side"):continue
            P, C, _, _ = objlib.depth_points(D, im, self.ep.intrinsics(c), self.cal.T_world_cam(c), stride=2)
            k = objlib.table_crop(P, region, *self.ep.table_plane(), above=-0.01)
            Ps.append(P[k]); Cs.append(C[k])
        P, C = np.concatenate(Ps), np.concatenate(Cs)
        if q is None:
            q = {a: self.cal.q_at(self.ep, a, self.cal.robot_tick(self.ep, "top", t)) for a in ("left", "right")}
        # Robot geometry is known even on a cold start. Do not let white links / black
        # grippers become object registration candidates.
        candidate_free = objlib.robot_free(P, self.arms.points(q), .015)
        for o in self.objs:
            T0 = None
            if getattr(o,"passive_fixture",False):
                from .box_fixture import fit_box,refine_box_pose
                try:
                    T0,_,Q,_,detail=fit_box(P,C)
                    T0,quality=refine_box_pose(T0,Q@T0[:3,:3].T+T0[:3,3],self.metas[o.name]["dimensions_m"])
                    o.initial_valid=True;self.init_info[o.name]={"current_frame_geometry":True,**detail,**quality}
                except ValueError as error:
                    # An occluded tray is unknown, not an episode-specific pose prior.
                    T0=np.eye(4);o.initial_valid=False
                    self.init_info[o.name]={"current_frame_geometry":False,"unobserved":str(error)}
            if T0 is None:
                pts = np.load(os.path.join(self.lib_dir, f"{o.name}_points.npz"))
                Pl, Cl = pts["P"].astype(float), pts["C"].astype(float)
                # colour-gated candidates off the table plane (an object may already be held in the air)
                n_, d_ = self.ep.table_plane()
                pr = objlib.colour_model(Cl, C[::40])(C)
                cand = (pr > 0.3) & ((P @ n_ + d_) > 0.008) & candidate_free
                if cand.sum() < 60:
                    cand = (pr > 0.2) & candidate_free
                if os.environ.get("REALPC_COLD_COMPONENTS", "0") == "1":
                    from .cold_pose import register_components
                    robot = MeshTracker([],{},self.tarm).robot(q,self.T_world_base)
                    grips=[r["fingers"] for r in robot.values() if r["grip"]<.85]
                    # Replay/live masks, when supplied, refer only to this first tick.
                    measured=[]
                    for camera,(D,image) in frames.items():
                        if camera not in ("top","side"):continue
                        lab = labels.get(camera) if labels is not None else None
                        if labels is None and camera in self.seg:
                            L,start0=self.seg[camera]
                            if 0<=t-start0<len(L):
                                lab=L[t-start0]
                        if lab is not None:
                            view=CamView(camera,D,self.ep.intrinsics(camera),self.cal.T_world_cam(camera),labels=lab)
                            measured.append(view.masked_points(self.objs.index(o)+1))
                    M=np.concatenate(measured) if measured else np.zeros((0,3))
                    target,colors=(M,np.full_like(M,.5)) if len(M)>=60 else (P[cand],C[cand])
                    source,source_colors=Pl,Cl
                    if len(M)>=60:
                        # Register the actual mesh being tracked, rather than the
                        # older partial scan used to construct its colour model.
                        source=o.icp_pts
                        source_colors=np.full_like(source,.5)
                    if len(M)<60:
                        # A distinctive material (e.g. green bristles) localizes better
                        # than generic white plastic shared with the robot/table.
                        hsv=cv2.cvtColor((np.clip(Cl,0,1)[None]*255).astype(np.uint8),cv2.COLOR_RGB2HSV)[0]
                        distinct=(hsv[:,1]>90)&(hsv[:,2]>45)
                        if .15<distinct.mean()<.8:
                            material_prob=objlib.colour_model(Cl[distinct],C[::30])(C)
                            choose=(material_prob>.65)&candidate_free&((P@n_+d_)>.008)
                            if choose.sum()>60:
                                source,source_colors=Pl[distinct],Cl[distinct]
                                target,colors=P[choose],C[choose]
                    prior_poses=[]
                    for arm_,r in robot.items():
                        if r["grip"]<.85:
                            prior_poses.extend(r["G"]@np.asarray(g["T_grip_obj"]) for g in self.grasp_priors.get(o.name,{}).get(arm_,[]))
                    T0, fit, rmse, detail = register_components(source,source_colors,target,colors,grippers=grips,contact_points=Pl,initial_poses=prior_poses)
                    detail["other_demo_grasp_hypotheses"]=len(prior_poses)
                    candidates=detail.pop("hypotheses",[])
                    if prior_poses:
                        from scipy.spatial import cKDTree
                        tree=cKDTree(target)
                        prior_error=[float(np.median(tree.query(source@H[:3,:3].T+H[:3,3])[0])) for H in prior_poses]
                        jj=int(np.argmin(prior_error))
                        # Preserve a supported grasp hypothesis instead of allowing
                        # unconstrained ICP to slide a partly visible symmetric head.
                        if len(M)<60 and prior_error[jj]<.020:
                            T0=prior_poses[jj].copy()
                            detail["grasp_seed_median_mm"]=prior_error[jj]*1000
                        candidates.extend(H.tolist() for H in prior_poses)
                    self.init_hypotheses[o.name]=candidates
                else:
                    T0, fit, rmse = objlib.register(Pl, Cl, P[cand], C[cand], voxel=0.005)
                    T0, fit, rmse = objlib.register(Pl, Cl, P[(pr > 0.15) & candidate_free], C[(pr > 0.15) & candidate_free], voxel=0.004, init=T0)
                    detail = {}
                self.init_info[o.name] = {"cold_registration_fitness": fit, "rmse_mm": rmse * 1000, **detail}
            else:
                self.init_info.setdefault(o.name, {"current_frame_geometry": True})
            init[o.name] = T0
        self.trk = MeshTracker(self.objs, init, self.tarm)
        self.trk.initial_hypotheses=self.init_hypotheses
        self.trk.lib_points = {o.name: tuple(np.load(os.path.join(self.lib_dir, f"{o.name}_points.npz"))[k].astype(float) for k in ("P", "C"))
                               for o in self.objs}
        self.trk.region, self.trk.plane = region, self.ep.table_plane()
        # static scene = table-region points not explained by the robot or a library object
        if q is None:
            q = {a: self.cal.q_at(self.ep, a, self.cal.robot_tick(self.ep, "top", t)) for a in ("left", "right")}
        R = self.arms.points(q)
        keep = objlib.robot_free(P, R, 0.03)
        for o in self.objs:
            if not getattr(o,"initial_valid",True):continue
            S = o.icp_pts @ init[o.name][:3, :3].T + init[o.name][:3, 3]
            keep &= objlib.robot_free(P, S, 0.012)
        P, C = P[keep], C[keep]
        kk = np.floor(P / self.static_voxel).astype(np.int64)
        _, idx = np.unique(kk, axis=0, return_index=True)
        self.static = (P[idx].astype(np.float32), (np.clip(C[idx], 0, 1) * 255).astype(np.uint8))
        self.trk.bg_colors = C[idx][::3]                       # static-scene colours: negatives of the colour models

    # ------------------------------------------------------------------ per tick
    def step(self, t, frames, labels=None, q=None, mask_provider=None):
        """Current RGB-D; optional current labels and calibrated, time-aligned joints.
        Live callers supply labels and q; file replay supplies identical values from
        causal SAM2 and joint history. q includes v3 offsets and the fixed 3-tick lag.
        No future image, label or joint value enters the tracker.
        """
        def lab(c):
            if labels is not None:
                return labels.get(c)
            if c not in self.seg:
                return None
            L, s0 = self.seg[c]
            return L[t - s0] if 0 <= t - s0 < len(L) else None
        if self.trk is None:
            self._init(t, frames, q=q, labels={c: lab(c) for c in frames})
        views = [CamView("top", frames["top"][0], self.ep.intrinsics("top"), self.cal.T_world_cam("top"),
                         grey=cv2.cvtColor(frames["top"][1], cv2.COLOR_BGR2GRAY), rgb=frames["top"][1], labels=lab("top"))]
        # side frames are used when their content time matches the top camera's (same lag); both cameras then show the
        # robot at robot_tick(top) = t - lag
        if "side" in frames and abs(self.cal.lag(self.ep, "side", t) - self.cal.lag(self.ep, "top", t)) <= self.side_max_lag:
            views.append(CamView("side", frames["side"][0], self.ep.intrinsics("side"), self.cal.T_world_cam("side"), lowres=(240, 320),
                                 rgb=frames["side"][1], labels=lab("side")))
        tr = self.cal.robot_tick(self.ep, "top", t)
        if q is None:
            q = {a: self.cal.q_at(self.ep, a, tr) for a in ("left", "right")}
        aux=[]
        if any(c.endswith("_wrist") for c in frames):
            LP=self.arms.link_poses(q);A=np.eye(4);A[:3,:3]=[[0,1,0],[1,0,0],[0,0,-1]]
            for arm_ in ("left","right"):
                c=arm_+"_wrist"
                if c not in frames:continue
                G=LP[self.arms.bodies.index((arm_,"gripper"))];Twc=G@A@np.asarray(self.ep.calib["X_"+arm_])
                aux.append(CamView(c,frames[c][0],self.ep.intrinsics(c),Twc,rgb=frames[c][1],lowres=(240,320)))
        # Passive tray: a resting pose needs no visual nudging. Refit near a
        # physical contact, or during initial acquisition; foreground occlusions
        # alone must not move a stationary box.
        if "box" in self.trk.state:
            from scipy.spatial import cKDTree
            st=self.trk.state["box"];box=self.trk.objs["box"]
            contact=False
            if st.valid:
                W=box.samples@st.T[:3,:3].T+st.T[:3,3];tree=cKDTree(W)
                contact=bool(tree.query(self.arms.points(q)[::8])[0].min()<.020)
                for name,other in self.trk.state.items():
                    if name=="box" or not other.valid:continue
                    moving=other.held is not None or (len(other.history)>1 and np.linalg.norm(other.history[-1][:3,3]-other.history[-2][:3,3])>.001)
                    if moving:
                        Q=self.trk.objs[name].samples[::4]@other.T[:3,:3].T+other.T[:3,3]
                        contact |= bool(tree.query(Q)[0].min()<.020)
            if contact:self.box_contact_until=t+10
            st.allow_fixture_update=(t-self.start<5 or t<=getattr(self,"box_contact_until",-1) or not st.valid)
            self.box_diagnostic={"contact":contact,"allow_update":st.allow_fixture_update}
        if "box" in self.trk.state and self.trk.state["box"].allow_fixture_update:
            from .box_fixture import update_box_pose
            pp=[];cc=[]
            for c in ("top","side"):
                if c not in frames:continue
                P,C,_,_=objlib.depth_points(frames[c][0],frames[c][1],self.ep.intrinsics(c),self.cal.T_world_cam(c),stride=4)
                pp.append(P);cc.append(C)
            st=self.trk.state["box"];meta=self.metas["box"]
            if not st.valid:
                from .box_fixture import fit_box,refine_box_pose
                try:
                    T0,_,Q,_,_=fit_box(np.concatenate(pp),np.concatenate(cc));st.T,_=refine_box_pose(T0,Q@T0[:3,:3].T+T0[:3,3],meta["dimensions_m"]);st.valid=True
                except ValueError:pass
            detail={"observed":bool(st.valid)}
            if st.valid:st.T,detail=update_box_pose(st.T,np.concatenate(pp),np.concatenate(cc),meta["dimensions_m"],meta.get("wall_planes"))
            self.box_diagnostic.update(detail)
        poses = self.trk.step(t, views, q, self.T_world_base,aux_views=aux,mask_provider=mask_provider)
        out = {"q": q, "robot_T": self.arms.link_poses(q).astype(np.float32), "obj_T": {}, "obj_res": {}, "obj_held": {}}
        for o in self.objs:
            st = self.trk.state[o.name]
            out["obj_T"][o.name] = poses[o.name].astype(np.float32) if st.valid else np.full((4,4),np.nan,np.float32)
            out.setdefault("obj_valid",{})[o.name]=bool(st.valid and np.isfinite(poses[o.name]).all())
            out["obj_res"][o.name] = st.last_res
            out["obj_held"][o.name] = 0 if st.held is None else (1 if st.held == "left" else 2)
            out.setdefault("obj_relocs", {})[o.name] = st.relocs
            out.setdefault("obj_support", {})[o.name] = (st.support.get("top", 0), st.support.get("side", 0))
            out.setdefault("obj_mode", {})[o.name] = st.mode
            if st.carrier is not None and st.held is None:
                out["obj_held"][o.name] = 3                       # carried by another (held) object
        return out

    def points(self, out):
        """points of one tick (robot | objects | static), fixed order -> [N,3] (what a live consumer uses)."""
        R = self.arms.points(None, poses=out["robot_T"].astype(float))
        O = [o.samples @ out["obj_T"][o.name][:3, :3].T + out["obj_T"][o.name][:3, 3] for o in self.objs]
        return np.concatenate([R] + O + [self.static[0]]).astype(np.float32)

    def point_valid(self,out):
        """Validity matching points(out); unknown objects remain fixed-ID masked slots."""
        return np.concatenate([np.ones(len(self.arms.local),bool)]+[np.full(len(o.samples),out.get("obj_valid",{}).get(o.name,True),bool) for o in self.objs]+[np.ones(len(self.static[0]),bool)])
