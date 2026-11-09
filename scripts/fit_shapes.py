#!/usr/bin/env python
"""Fit clean object models by multi-view silhouette + depth (realpc/shapefit.py) and write a library v2.

  LD_PRELOAD=... python scripts/fit_shapes.py --task sweep --name debris --model box --out assets/objects_v2/sweep \
      [--episodes ...] [--frames_per_ep 8]
Models: box | mug (cylinder + C handle) | deform (smoothed v1 surface + normal offsets, Laplacian).
Frames: our SAM2 object masks (scripts/segment_objects.py) + depth + the tracked v1 pose of that tick (converted v1
dataset); robot pixels (SAM label or FK render) and other objects are ignored (occluders).
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
import trimesh

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from realpc.calib_rig import RigCalib  # noqa: E402
from realpc.config import load_config  # noqa: E402
from realpc.diffcalib import Renderer, TorchArm  # noqa: E402
from realpc.io import Episode, read_depth_frames  # noqa: E402
from realpc import shapefit as sf  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True); ap.add_argument("--name", required=True); ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--episodes", nargs="*", default=None)
ap.add_argument("--frames_per_ep", type=int, default=8); ap.add_argument("--iters", type=int, default=600)
ap.add_argument("--lib", default=os.path.join(ROOT, "assets", "objects"))
ap.add_argument("--conv", default=os.path.expanduser("~/data_ext/realpc_out/v1"))
ap.add_argument("--seg", default=os.path.expanduser("~/data_ext/realpc_v4/seg"))
ap.add_argument("--data", default=os.path.expanduser("~/data_ext/agentwm_probe/hf"))
ap.add_argument("--calib", default=os.path.join(ROOT, "configs", "calib_yam_rig_v3.json"))
ap.add_argument("--min_px", type=int, default=250)
a = ap.parse_args()
cal = RigCalib.load(a.calib)
cfg = load_config(os.path.join(ROOT, "configs", "yam_bimanual.json")); r = cfg["robot"]
tarm = TorchArm(os.path.join(cfg["_dir"], r["urdf"]), r["arm_joints"], r["finger_joints"])
eps = a.episodes or ([f"sweep/episode_{i:04d}" for i in range(0, 50, 3)] if a.task == "sweep" else [f"{a.task}/episode_0000"])
rng = np.random.default_rng(0)
frames, meta_frames = [], []
for e in eps:
    ep = Episode(os.path.join(a.data, e))
    d = np.load(os.path.join(a.conv, e + ".npz"))
    names = [str(x) for x in d["obj_names"]]
    if a.name not in names:
        continue
    pi = names.index(a.name)
    ticks = d["ticks"]
    for c in ("top", "side"):
        p = os.path.join(a.seg, ep.task, f"{ep.name}_{c}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p); L = z["labels"]; sn = [str(x) for x in z["names"]]
        lid = sn.index(a.name) + 1
        others = [i + 1 for i, n in enumerate(sn) if n != a.name] + [254, 255]
        px = (L[:, ::2, ::2] == lid).reshape(len(L), -1).sum(1) * 4
        cand = np.flatnonzero(px >= a.min_px)
        if len(cand) == 0:
            continue
        # spread over the episode: static start + the rest
        pick = np.unique(np.r_[cand[:2], rng.choice(cand, min(a.frames_per_ep - 2, len(cand)), replace=False)])
        tk = [int(ticks[k]) for k in pick]
        D = read_depth_frames(os.path.join(ep.path, f"{c}_depth.mkv"), tk)
        Wn = 1280 if c == "top" else 640
        h = int(round((720 if c == "top" else 480) * 320 / Wn))
        K = ep.intrinsics(c).copy(); K[:2] *= 320 / Wn
        Rr = Renderer(K, h, 320)
        for k, t in zip(pick, tk):
            T = d["obj_T"][pi, k].astype(float)
            if not np.isfinite(T).all():
                continue
            lab = cv2.resize(L[k], (320, h), interpolation=cv2.INTER_NEAREST)
            mask = lab == lid
            ign = np.isin(lab, others)
            # FK robot render (calibrated) as occluder too
            for s in ("left", "right"):
                q = d["q_left" if s == "left" else "q_right"][k]
                V = tarm.verts(torch.tensor(q[None], dtype=torch.float32, device="cuda"))[0]
                Tb = torch.tensor(cal.T_cam_base(c, s), dtype=torch.float32, device="cuda")
                sil, _ = Rr.raster((V @ Tb[:3, :3].T + Tb[:3, 3])[None], tarm.faces)
                ign |= cv2.dilate((sil[0].cpu().numpy() > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~mask
            dep = cv2.resize(D[t].astype(np.float32) * float(ep.meta["depth_scale"][c]), (320, h), interpolation=cv2.INTER_NEAREST)
            T_co = np.linalg.inv(cal.T_world_cam(c)) @ T
            frames.append(sf.Frame(mask, ~ign, dep, K, T_co, h, 320))
            meta_frames.append((e, c, t))
print(f"{a.task}/{a.name}: {len(frames)} frames from {len(set(m[0] for m in meta_frames))} episodes", flush=True)

v1 = trimesh.load(os.path.join(a.lib, a.task, f"{a.name}.ply"), process=False)
pts = np.load(os.path.join(a.lib, a.task, f"{a.name}_points.npz"))


def run(model, iters, **kw):
    dp = sf.fit(model, frames, iters=iters, **kw)
    return dp, sf.eval_iou(model, frames, dp)


if a.model == "box":
    obb = v1.bounding_box_oriented
    T = obb.primitive.transform
    from scipy.spatial.transform import Rotation as Rot
    model = sf.PrimitiveModel([{"type": "box", "size": list(np.asarray(obb.primitive.extents) * 0.85),
                                "rot": Rot.from_matrix(T[:3, :3]).as_rotvec().tolist(), "pos": T[:3, 3].tolist()}])
    dp, iou = run(model, a.iters)
elif a.model == "mug":
    from scipy.spatial.transform import Rotation as Rot
    obb = v1.bounding_box_oriented; T = obb.primitive.transform; ext = np.asarray(obb.primitive.extents)
    best = None
    for ax in range(3):                      # cylinder axis candidates: the three OBB axes
        R = T[:3, :3].copy()
        perm = [(ax + 1) % 3, (ax + 2) % 3, ax]
        Rc = R[:, perm]
        if np.linalg.det(Rc) < 0:
            Rc[:, 0] *= -1
        rad = 0.4 * min(ext[perm[0]], ext[perm[1]])
        m = sf.PrimitiveModel([{"type": "cyl", "size": [rad, rad, 0.8 * ext[ax]], "rot": Rot.from_matrix(Rc).as_rotvec().tolist(),
                                "pos": T[:3, 3].tolist()}])
        dp, iou = run(m, a.iters // 2)
        print(f"  axis {ax}: IoU median {np.median(iou):.3f}", flush=True)
        if best is None or np.median(iou) > best[0]:
            best = (np.median(iou), m, dp)
    cyl = best[1]
    # handle: side where the observed mug points stick out beyond the fitted cylinder
    Rc = sf.so3_exp(cyl.rot[0].detach()).cpu().numpy(); c0 = cyl.pos[0].detach().cpu().numpy()
    rad = float(cyl.log_size.exp()[0, 0]); hgt = float(cyl.log_size.exp()[0, 2])
    P = (pts["P"] - c0) @ Rc
    radial = np.linalg.norm(P[:, :2], axis=1)
    out = P[(radial > rad + 0.008) & (np.abs(P[:, 2]) < hgt / 2)]
    az = np.arctan2(out[:, 1].mean(), out[:, 0].mean()) if len(out) > 20 else 0.0
    u = np.array([np.cos(az), np.sin(az), 0.0])
    reach = float(np.clip(np.percentile(np.linalg.norm(out[:, :2], axis=1), 90) - rad, 0.015, 0.04)) if len(out) > 20 else 0.025
    w = np.cross([0, 0, 1.0], u)
    Rh = np.stack([u, w, [0, 0, 1.0]], 1)
    parts = [{"type": "cyl", "size": [rad, rad, hgt], "rot": Rot.from_matrix(Rc).as_rotvec().tolist(), "pos": c0.tolist()}]
    for zc, sz, mx in ((0.3 * hgt, [reach, 0.012, 0.01], [0.045, 0.015, 0.012]), (-0.3 * hgt, [reach, 0.012, 0.01], [0.045, 0.015, 0.012]),
                       (0.0, [0.01, 0.012, 0.6 * hgt], [0.012, 0.015, 0.09])):
        x = rad + (reach / 2 if sz[0] == reach else reach)
        p_local = np.array([x, 0, zc])
        parts.append({"type": "box", "size": sz, "max": mx, "min": [0.006, 0.008, 0.006], "rot": Rot.from_matrix(Rc @ Rh).as_rotvec().tolist(),
                      "pos": (c0 + Rc @ (Rh @ p_local)).tolist()})
    model = sf.PrimitiveModel(parts)
    dp, iou = run(model, a.iters)
elif a.model == "broom":
    from scipy.spatial.transform import Rotation as Rot
    P, C = pts["P"], pts["C"]
    hsv = __import__("cv2").cvtColor((np.clip(C, 0, 1)[None] * 255).astype(np.uint8), __import__("cv2").COLOR_RGB2HSV)[0]
    green = (hsv[:, 0] > 35) & (hsv[:, 0] < 90) & (hsv[:, 1] > 60)
    white = ~green
    def obb_part(Q, shrink=1.0, mx=None):
        o = trimesh.points.PointCloud(Q).bounding_box_oriented
        T = o.primitive.transform
        d = {"type": "box", "size": list(np.maximum(np.asarray(o.primitive.extents) * shrink, 0.008)),
             "rot": Rot.from_matrix(T[:3, :3]).as_rotvec().tolist(), "pos": T[:3, 3].tolist()}
        if mx is not None:
            d["max"] = mx
        return d
    G = P[green]; W = P[white]
    gc = G.mean(0)
    dist = np.linalg.norm(W - gc, axis=1)
    head = W[dist < np.percentile(dist, 40)]; handle = W[dist >= np.percentile(dist, 40)]
    parts = [obb_part(G, 0.9), obb_part(head, 0.9), obb_part(handle, 0.85, mx=[0.25, 0.04, 0.03])]
    model = sf.PrimitiveModel(parts)
    dp, iou = run(model, a.iters)
elif a.model == "ffd":
    base = v1.copy()
    trimesh.smoothing.filter_taubin(base, iterations=60)
    model = sf.FFDModel(base, n=4)
    dp, iou = run(model, a.iters, smooth_w=50.0, lr=1e-3)
else:  # deform
    base = sf.smooth_remesh(v1, voxel=0.004, sigma=1.5)
    if len(base.faces) > 12000:
        base = base.simplify_quadric_decimation(face_count=12000) if hasattr(base, "simplify_quadric_decimation") else base
    model = sf.DeformModel(base)
    dp, iou = run(model, a.iters, smooth_w=2e4, lr=5e-4)

mesh = model.mesh()
mesh.merge_vertices()
if not mesh.is_watertight:
    mesh = sf.smooth_remesh(mesh, voxel=0.002, sigma=0.6)
# vertex colours from the coloured v1 points
from scipy.spatial import cKDTree  # noqa: E402
_, j = cKDTree(pts["P"]).query(mesh.vertices)
mesh.visual.vertex_colors = np.c_[(np.clip(pts["C"][j], 0, 1) * 255).astype(np.uint8), np.full(len(j), 255, np.uint8)]
meta = json.load(open(os.path.join(a.lib, a.task, f"{a.name}.json")))
v1_iou = sf.eval_iou(sf.DeformModel(v1), frames)
meta.update({"version": "v2 (clean model fitted by multi-view silhouette + depth)", "model": a.model,
             "fit_frames": len(frames), "fit_episodes": sorted(set(m[0] for m in meta_frames)),
             "fit_iou_median": float(np.median(iou)), "v1_mesh_iou_median_same_frames": float(np.median(v1_iou)),
             "watertight": bool(mesh.is_watertight), "volume_m3": float(mesh.volume), "extents_m": mesh.extents.tolist(),
             "n_vertices": int(len(mesh.vertices)), "n_faces": int(len(mesh.faces))})
if hasattr(model, "types"):
    meta["parts"] = [{"type": t, "size_m": s.tolist()} for t, s in zip(model.types, model.log_size.exp().detach().cpu().numpy())]
os.makedirs(a.out, exist_ok=True)
mesh.export(os.path.join(a.out, f"{a.name}.ply"))
json.dump(meta, open(os.path.join(a.out, f"{a.name}.json"), "w"), indent=1)
np.savez_compressed(os.path.join(a.out, f"{a.name}_points.npz"), P=pts["P"], C=pts["C"])
print(f"saved {a.name}: {a.model}, extents {np.round(mesh.extents * 100, 1)} cm, watertight {mesh.is_watertight}, "
      f"fit IoU median {np.median(iou):.3f} (v1 mesh {np.median(v1_iou):.3f})", flush=True)
