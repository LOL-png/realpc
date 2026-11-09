#!/usr/bin/env python
"""Bootstrap step of the object library: mesh v0 -> causal pose tracks -> every camera view of the episode transformed
into the OBJECT frame (T_obj_cam = T_world_obj(t)^-1 T_world_cam) -> space carving with all views -> mesh v1.

Held and static ticks both add views (held objects are seen from many angles). Carving is voted: a voxel is removed
when at least `--min_free` views see through it and they are >= `--free_frac` of the views that observe it at all.
Image masks (if given) also carve: a voxel projecting onto a pixel labelled as background/table is free.

  LD_PRELOAD=... python scripts/refine_object_lib.py --lib <dir> --name mug --tracks <track_objects npz> \
      --episode <ep> --calib <json> [--masks <dir>] [--every 3] [--max_res 0.008] --out <dir>
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import trimesh

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from realpc import objlib  # noqa: E402
from realpc.calib_rig import RigCalib  # noqa: E402
from realpc.io import Episode, read_depth_frames  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--lib", required=True); ap.add_argument("--name", required=True); ap.add_argument("--tracks", nargs="+", required=True)
ap.add_argument("--episodes", nargs="+", required=True); ap.add_argument("--calib", required=True); ap.add_argument("--masks", nargs="*", default=[])
ap.add_argument("--every", type=int, default=3); ap.add_argument("--max_res", type=float, default=0.008)
ap.add_argument("--voxel", type=float, default=0.0025); ap.add_argument("--min_free", type=int, default=2)
ap.add_argument("--free_frac", type=float, default=0.25); ap.add_argument("--out", required=True)
ap.add_argument("--side_max_lag", type=float, default=2.5); ap.add_argument("--tick_max", type=int, default=None)
a = ap.parse_args()
cal = RigCalib.load(a.calib)
mesh0 = trimesh.load(os.path.join(a.lib, f"{a.name}.ply"), process=False)
meta = json.load(open(os.path.join(a.lib, f"{a.name}.json")))
pts = np.load(os.path.join(a.lib, f"{a.name}_points.npz"))
lo = mesh0.bounds[0] - 0.02; hi = mesh0.bounds[1] + 0.02
shape = np.ceil((hi - lo) / a.voxel).astype(int)
ii = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), -1).reshape(-1, 3)
X = lo + (ii + 0.5) * a.voxel
free = np.zeros(len(X), np.int32); seen = np.zeros(len(X), np.int32); surf = np.zeros(len(X), np.int32)
Psurf, Csurf = [pts["P"]], [pts["C"]]
nviews = 0
for ti, (trk, epp) in enumerate(zip(a.tracks, a.episodes)):
    d = np.load(trk, allow_pickle=True)
    names = [str(x) for x in d["obj_names"]]
    pi = names.index(a.name)
    ticks = d["ticks"]; T_wo = d["obj_T"][pi]; res = d["obj_res"][pi]
    ep = Episode(epp)
    use = [k for k in range(0, len(ticks), a.every) if np.isfinite(T_wo[k, 0, 0]) and (np.isfinite(res[k, 0]) and res[k, 0] < a.max_res or a.tick_max is not None)
           and (a.tick_max is None or ticks[k] < a.tick_max)]
    tk = [int(ticks[k]) for k in use]
    masks = {}
    if ti < len(a.masks) and a.masks[ti]:
        for c in ("top", "side"):
            p = os.path.join(a.masks[ti], f"{c}_masks.npz")
            if os.path.exists(p):
                z = np.load(p); masks[c] = (z["masks"], [str(x) for x in z["label_names"]])
    for c in ("top", "side"):
        D = read_depth_frames(os.path.join(ep.path, f"{c}_depth.mkv"), tk)
        sc = float(ep.meta["depth_scale"][c]); K = ep.intrinsics(c)
        cap = cv2.VideoCapture(ep.video(c))
        for k, t in zip(use, tk):
            if c == "side" and abs(cal.lag(ep, "side", t)) > a.side_max_lag:
                continue
            Dm = D[t].astype(np.float32) * sc
            Toc = np.linalg.inv(T_wo[k].astype(float)) @ cal.T_world_cam(c)
            Tco = np.linalg.inv(Toc)
            Pc = X @ Tco[:3, :3].T + Tco[:3, 3]
            z = Pc[:, 2]; H, W = Dm.shape
            u = np.round(K[0, 0] * Pc[:, 0] / np.maximum(z, 1e-6) + K[0, 2]).astype(int)
            v = np.round(K[1, 1] * Pc[:, 1] / np.maximum(z, 1e-6) + K[1, 2]).astype(int)
            inb = (z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            dd = np.zeros(len(X)); dd[inb] = Dm[v[inb], u[inb]]
            val = inb & (dd > 0.1)
            fr = val & (dd > z + 0.008)
            if c in masks and t < len(masks[c][0]):
                lab = masks[c][0][t]; nm = masks[c][1]
                bg = np.isin(lab, [nm.index(x) for x in ("background", "table") if x in nm])
                fr |= inb & bg[np.clip(v, 0, H - 1), np.clip(u, 0, W - 1)] & ~(val & (dd < z - 0.01))
                # pixels of ANOTHER object at (or behind) the voxel's depth: the voxel is not this object
                oth = np.isin(lab, [i for i, x in enumerate(nm) if x not in ("background", "table", a.name) and not x.startswith("robot")])
                fr |= inb & oth[np.clip(v, 0, H - 1), np.clip(u, 0, W - 1)] & val & (dd > z - 0.01)
            sv = val & (np.abs(dd - z) < 0.006)
            if c in masks and t < len(masks[c][0]):
                own = masks[c][0][t] == masks[c][1].index(a.name) if a.name in masks[c][1] else np.zeros_like(masks[c][0][t], bool)
                sv &= own[np.clip(v, 0, H - 1), np.clip(u, 0, W - 1)]
            free += fr; seen += (val | fr); surf += sv
            nviews += 1
            # observed surface points of the object in this view (for colours / coverage)
            if k % (a.every * 4) == 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, t); ok, im = cap.read()
                P, C, _, _ = objlib.depth_points(Dm, im, K, Toc, stride=3)
                dm, _ = objlib.cKDTree(mesh0.vertices).query(P, distance_upper_bound=0.006)
                if c in masks and t < len(masks[c][0]):
                    pass
                Psurf.append(P[np.isfinite(dm)]); Csurf.append(C[np.isfinite(dm)])
    print(f"{epp}: {len(use)} ticks with residual < {a.max_res*1000:.0f} mm", flush=True)
occ0 = mesh0.voxelized(a.voxel).fill().is_filled(X)
remove = (free >= a.min_free) & (free >= a.free_frac * np.maximum(seen, 1))
occ = (occ0 | (surf >= 3)) & ~remove
occ = occ.reshape(shape)
occ = objlib.object_occupancy(occ, lo, a.voxel, X[(surf >= 2)] if a.masks else np.concatenate([pts["P"]] + [p for p in Psurf[1:]]), near=0.02)
P = np.concatenate(Psurf); C = np.concatenate(Csurf)
mesh = objlib.occupancy_mesh(occ, lo, a.voxel, seed_pts=pts["P"])
_, j = objlib.cKDTree(P).query(mesh.vertices)
mesh.visual.vertex_colors = np.c_[(np.clip(C[j], 0, 1) * 255).astype(np.uint8), np.full(len(j), 255, np.uint8)]
dsurf, _ = objlib.cKDTree(P).query(mesh.vertices)
meta.update({"version": "v1 (bootstrap: tracked views in the object frame)", "watertight": bool(mesh.is_watertight),
             "volume_m3": float(mesh.volume), "extents_m": mesh.extents.tolist(), "n_vertices": int(len(mesh.vertices)),
             "n_faces": int(len(mesh.faces)), "refine_views": nviews, "refine_tracks": a.tracks,
             "observed_surface_frac": float((dsurf < 0.005).mean())})
os.makedirs(a.out, exist_ok=True)
objlib.save_object(mesh, meta, a.out, a.name)
np.savez_compressed(os.path.join(a.out, f"{a.name}_points.npz"), P=pts["P"], C=pts["C"])
print(f"saved {a.name} v1: watertight {mesh.is_watertight}, extents {np.round(mesh.extents, 3)}, views {nviews}", flush=True)
