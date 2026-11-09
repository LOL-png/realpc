#!/usr/bin/env python
"""Causal mesh-pose tracking of the library objects on one episode (realpc/meshtrack.py). Same code for conversion
(whole episodes) and inference (start anywhere: --start S, cold start = registration of the library to frame S).

  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python scripts/track_objects.py --episode <ep> \
      --lib assets/objects/<task> --calib configs/calib_yam_rig_v3.json --out tracks.npz [--start S --stop E] \
      [--masks <dir with {top,side}_masks.npz>] [--iou_every 5]
Output npz: obj_names, obj_T [P,T,4,4], obj_res [P,T,2] (median |depth residual| m, n), held [P,T] (0 none, 1 left,
2 right), iou [P,T,2] (top, side; NaN without masks), points: pos [N,T,3], group [N] (0 gripper,1 robot,2 static,3 object),
body [N], ticks [T].
"""
import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import trimesh

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from realpc import objlib  # noqa: E402
from realpc.calib_rig import RigCalib  # noqa: E402
from realpc.config import build_arms, load_config  # noqa: E402
from realpc.diffcalib import Renderer, TorchArm  # noqa: E402
from realpc.io import DepthStream, Episode  # noqa: E402
from realpc.meshtrack import CamView, LibObject, MeshTracker  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--episode", required=True); ap.add_argument("--lib", required=True); ap.add_argument("--calib", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--start", type=int, default=0); ap.add_argument("--stop", type=int, default=None)
ap.add_argument("--masks", default=None); ap.add_argument("--iou_every", type=int, default=5)
ap.add_argument("--side_max_lag", type=float, default=2.5); ap.add_argument("--no_flow", action="store_true")
ap.add_argument("--n_samples", type=int, default=1500); ap.add_argument("--cold", action="store_true",
                help="ignore library start poses; register at --start (what inference sees)")
a = ap.parse_args()

ep = Episode(a.episode)
cal = RigCalib.load(a.calib)
cfg = load_config(os.path.join(ROOT, "configs", "yam_bimanual.json")); r = cfg["robot"]
tarm = TorchArm(os.path.join(cfg["_dir"], r["urdf"]), r["arm_joints"], r["finger_joints"])
arms = build_arms(cfg, ep.calib); arms.T_base["right"] = cal.T_world_rb()
T_world_base = {"left": np.eye(4), "right": cal.T_world_rb()}
key = f"{ep.task}/{ep.name}"
T_total = len(ep.low["t_cap"])
stop = min(a.stop or T_total, T_total)

# ------------------------------------------------------------------ library + initial poses
objs, init = [], {}
for f in sorted(glob.glob(os.path.join(a.lib, "*.ply"))):
    meta = json.load(open(f[:-4] + ".json"))
    o = LibObject(meta["name"], trimesh.load(f, process=False), n_samples=a.n_samples)
    objs.append(o)
    T0 = None
    if not a.cold and a.start == 0:
        for sc in meta["scans"]:
            if sc["episode"] == key:
                T0 = np.asarray(sc["T_world_obj_t0"], float)
    if T0 is None:
        # cold start: register the library's coloured points into the scene at --start
        pts = np.load(f[:-4] + "_points.npz")
        from realpc.io import read_depth_frames
        Ps, Cs = [], []
        for c in ("top", "side"):
            D = read_depth_frames(os.path.join(ep.path, f"{c}_depth.mkv"), [a.start])[a.start].astype(np.float32) * float(ep.meta["depth_scale"][c])
            cap = cv2.VideoCapture(ep.video(c)); cap.set(cv2.CAP_PROP_POS_FRAMES, a.start); _, im = cap.read()
            P, C, _, _ = objlib.depth_points(D, im, ep.intrinsics(c), cal.T_world_cam(c))
            region = json.load(open(os.path.join(ep.path, "table_region.json")))
            k = objlib.table_crop(P, region, *ep.table_plane(), above=0.003)
            Ps.append(P[k]); Cs.append(C[k])
        P, C = np.concatenate(Ps), np.concatenate(Cs)
        Tg, fit, rmse = objlib.register(pts["P"].astype(float), pts["C"].astype(float), P, C, voxel=0.005)
        print(f"cold-start registration {meta['name']}: fitness {fit:.2f} rmse {rmse*1000:.1f} mm", flush=True)
        T0 = Tg
    init[o.name] = T0

trk = MeshTracker(objs, init, tarm, cfg={"flow": not a.no_flow})
masks = {}
if a.masks:
    for c in ("top", "side"):
        p = os.path.join(a.masks, f"{c}_masks.npz")
        if os.path.exists(p):
            z = np.load(p)
            masks[c] = (z["masks"], [str(x) for x in z["label_names"]])

# ------------------------------------------------------------------ streaming loop
ds = {c: DepthStream(os.path.join(ep.path, f"{c}_depth.mkv"), a.start) for c in ("top", "side")}
caps = {c: cv2.VideoCapture(ep.video(c)) for c in ("top", "side")}
for c in caps:
    caps[c].set(cv2.CAP_PROP_POS_FRAMES, a.start)
scale = {c: float(ep.meta["depth_scale"][c]) for c in ("top", "side")}
P_n = len(objs)
ticks = list(range(a.start, stop))
obj_T = np.full((P_n, len(ticks), 4, 4), np.nan, np.float32)
obj_res = np.full((P_n, len(ticks), 2), np.nan, np.float32)
held = np.zeros((P_n, len(ticks)), np.int8)
iou = np.full((P_n, len(ticks), 2), np.nan, np.float32)
side_hist = {}                                       # pose history for the lagged side camera IoU
t0 = time.time()
for k, t in enumerate(ticks):
    views = []
    frames = {}
    for c in ("top", "side"):
        _, D = ds[c].next()
        ok, im = caps[c].read()
        frames[c] = (D.astype(np.float32) * scale[c], im)
    # top: content time = t (reference); side: shows the robot at t - lag; only used when the lag is small
    views.append(CamView("top", frames["top"][0], ep.intrinsics("top"), cal.T_world_cam("top"),
                         grey=cv2.cvtColor(frames["top"][1], cv2.COLOR_BGR2GRAY)))
    lag_s = cal.lag(ep, "side", t)
    if abs(lag_s) <= a.side_max_lag:
        views.append(CamView("side", frames["side"][0], ep.intrinsics("side"), cal.T_world_cam("side"), lowres=(240, 320)))
    q = {arm_: cal.q_at(ep, arm_, cal.robot_tick(ep, "top", t)) for arm_ in ("left", "right")}
    poses = trk.step(t, views, q, T_world_base)
    for pi, o in enumerate(objs):
        st = trk.state[o.name]
        obj_T[pi, k] = poses[o.name]
        obj_res[pi, k] = st.last_res
        held[pi, k] = 0 if st.held is None else (1 if st.held == "left" else 2)
    # IoU vs SAM masks (evaluation only; never used by the tracker)
    if masks and k % a.iou_every == 0:
        for ci, c in enumerate(("top", "side")):
            if c not in masks or t >= len(masks[c][0]):
                continue
            M, names = masks[c]
            H, W = M.shape[1:]
            K = ep.intrinsics(c).copy(); s = 320 / W; K[:2] *= s
            R = Renderer(K, int(H * s), 320)
            m_lab = cv2.resize(M[t], (320, int(H * s)), interpolation=cv2.INTER_NEAREST)
            # pose at the camera's content time (side lags)
            tc = t - (cal.lag(ep, c, t) if c == "side" else 0.0)
            kc = int(np.clip(round(tc - a.start), 0, k))
            Tcw = torch.tensor(np.linalg.inv(cal.T_world_cam(c)), dtype=torch.float32, device="cuda")
            for pi, o in enumerate(objs):
                if o.name not in names or not np.isfinite(obj_T[pi, kc, 0, 0]):
                    continue
                Tt = torch.tensor(obj_T[pi, kc], dtype=torch.float32, device="cuda")
                Vw = o.verts_t @ Tt[:3, :3].T + Tt[:3, 3]
                sil, _ = R.raster((Vw @ Tcw[:3, :3].T + Tcw[:3, 3])[None], o.faces_t)
                rm = sil[0].cpu().numpy() > 0.5
                gt = m_lab == names.index(o.name)
                # pixels of the robot are excluded (occluders), other objects too
                valid = ~np.isin(m_lab, [names.index(x) for x in names if x.startswith("robot")])
                if (gt & valid).sum() + (rm & valid).sum() < 30:
                    continue
                iou[pi, k, ci] = ((gt & rm & valid).sum()) / max(1, ((gt | rm) & valid).sum())
    if k % 50 == 0:
        print(f"t {t} ({(time.time()-t0)/(k+1):.2f} s/tick) " + " ".join(
            f"{o.name}:{obj_res[i,k,0]*1000:.1f}mm/{int(obj_res[i,k,1])}{'H' if held[i,k] else ''}" for i, o in enumerate(objs)), flush=True)

# ------------------------------------------------------------------ points: robot FK, object mesh samples, static table
pos, group, body = [], [], []
Ld = arms.local; Bd = arms.body
rob = np.zeros((len(Ld), len(ticks), 3), np.float32)
for k, t in enumerate(ticks):
    q = {arm_: cal.q_at(ep, arm_, cal.robot_tick(ep, "top", t)) for arm_ in ("left", "right")}
    rob[:, k] = arms.points(q)
is_grip = np.array([arms.bodies[b][1] in ("gripper", "tip_left", "tip_right") for b in Bd])
pos.append(rob); group.append(np.where(is_grip, 0, 1)); body.append(Bd)
nb = int(Bd.max()) + 1
for pi, o in enumerate(objs):
    S = o.samples
    P = np.einsum("tij,nj->nti", obj_T[pi, :, :3, :3], S) + obj_T[pi, :, None, :3, 3].transpose(1, 0, 2)
    pos.append(P.astype(np.float32)); group.append(np.full(len(S), 3)); body.append(np.full(len(S), nb + pi))
pos = np.concatenate(pos); group = np.concatenate(group).astype(np.int8); body = np.concatenate(body).astype(np.int32)
np.savez_compressed(a.out, obj_names=np.array([o.name for o in objs]), obj_T=obj_T, obj_res=obj_res, held=held, iou=iou,
                    pos=pos, group=group, body=body, ticks=np.array(ticks), calib=a.calib, lib=a.lib, start=a.start)
summ = {o.name: {"res_med_mm": float(np.nanmedian(obj_res[i, :, 0]) * 1000), "res_p90_mm": float(np.nanpercentile(obj_res[i, :, 0], 90) * 1000),
                 "iou_top_med": float(np.nanmedian(iou[i, :, 0])) if np.isfinite(iou[i, :, 0]).any() else None,
                 "iou_side_med": float(np.nanmedian(iou[i, :, 1])) if np.isfinite(iou[i, :, 1]).any() else None,
                 "held_ticks": int((held[i] > 0).sum())} for i, o in enumerate(objs)}
json.dump(summ, open(a.out[:-4] + "_summary.json", "w"), indent=1)
print(json.dumps(summ, indent=1))
