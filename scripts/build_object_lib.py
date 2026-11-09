#!/usr/bin/env python
"""Build the object mesh library of one task from many episodes (realpc/objlib.py).

  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python scripts/build_object_lib.py --task sweep \
      --ref sweep/episode_0000 --episodes sweep/episode_0001 ... --calib configs/calib_yam_rig_v3.json \
      --masks_root ~/data_ext/masks --out assets/objects/sweep --objects broom dustpan debris

Static ticks: the first ticks of an episode, before either gripper moves more than 2 cm (objects untouched).
"""
import argparse
import datetime
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from realpc.calib_rig import RigCalib  # noqa: E402
from realpc.config import build_arms, load_config  # noqa: E402
from realpc.io import Episode, read_depth_frames  # noqa: E402
from realpc import objlib  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True); ap.add_argument("--ref", required=True); ap.add_argument("--episodes", nargs="*", default=[])
ap.add_argument("--calib", default=None); ap.add_argument("--masks_root", default=os.path.expanduser("~/data_ext/masks"))
ap.add_argument("--data", default=os.path.expanduser("~/data_ext/agentwm_probe/hf")); ap.add_argument("--out", required=True)
ap.add_argument("--objects", nargs="+", required=True); ap.add_argument("--n_static", type=int, default=3)
ap.add_argument("--cams", nargs="+", default=["top", "side"]); ap.add_argument("--voxel", type=float, default=0.002)
ap.add_argument("--debug_dir", default=None)
a = ap.parse_args()

cfg = load_config(os.path.join(ROOT, "configs", "yam_bimanual.json"))


def static_ticks(ep, n):
    """first ticks before any fingertip moved 2 cm (FK eef positions)."""
    pl, pr = ep.low["eef_pos_left"], ep.low["eef_pos_right"]
    mv = np.maximum(np.linalg.norm(pl - pl[0], axis=1), np.linalg.norm(pr - pr[0], axis=1))
    last = int(np.argmax(mv > 0.02)) if (mv > 0.02).any() else len(mv)
    last = max(1, last - 2)
    return sorted(set(np.linspace(0, last - 1, n).astype(int).tolist()))


def scene(ep, cal, ticks, masks=None, above=0.003):
    """fused scene points of the static ticks -> P, C, label (or -1), cam id."""
    arms = build_arms(cfg, ep.calib)
    arms.T_base["right"] = cal.T_world_rb()
    Ld, Bd = arms.dense_local(800)
    region = json.load(open(os.path.join(ep.path, "table_region.json")))
    n, d = ep.table_plane()
    Ps, Cs, Ls, Ks, views = [], [], [], [], []
    for ci, c in enumerate(a.cams):
        D = read_depth_frames(os.path.join(ep.path, f"{c}_depth.mkv"), ticks)
        sc = float(ep.meta.get("depth_scale", {}).get(c, 0.001))
        cap = cv2.VideoCapture(ep.video(c))
        for t in ticks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, t); ok, im = cap.read()
            lab = None
            if masks is not None and c in masks:
                lab = masks[c][t]
            Dm = D[t].astype(np.float32) * sc
            views.append((Dm, ep.intrinsics(c), cal.T_world_cam(c)))
            P, C, L, _ = objlib.depth_points(Dm, im, ep.intrinsics(c), cal.T_world_cam(c), stride=2, labels=lab)
            q = {arm_: ep.low[f"joint_pos_{arm_}"][t] for arm_ in ("left", "right")}
            LP = arms.link_poses(q)
            R = np.einsum("nij,nj->ni", LP[Bd, :3, :3], Ld) + LP[Bd, :3, 3]
            keep = objlib.table_crop(P, region, n, d, above=above) & objlib.robot_free(P, R)
            Ps.append(P[keep]); Cs.append(C[keep]); Ks.append(np.full(keep.sum(), ci))
            Ls.append(L[keep].astype(int) if L is not None else np.full(keep.sum(), -1))
    return np.concatenate(Ps), np.concatenate(Cs), np.concatenate(Ls), np.concatenate(Ks), views


def voxel_mean(P, C, v):
    k = np.floor(P / v).astype(np.int64)
    _, inv = np.unique(k, axis=0, return_inverse=True)
    inv = inv.ravel()
    cnt = np.bincount(inv)
    Pm = np.stack([np.bincount(inv, P[:, i]) for i in range(3)], 1) / cnt[:, None]
    Cm = np.stack([np.bincount(inv, C[:, i]) for i in range(3)], 1) / cnt[:, None]
    return Pm, Cm, cnt


ref = Episode(os.path.join(a.data, a.ref))
cal = RigCalib.load(a.calib) if a.calib else RigCalib.recorder(ref)
mk = {}
for c in a.cams:
    p = os.path.join(a.masks_root, a.ref, f"{c}_masks.npz")
    if os.path.exists(p):
        z = np.load(p)
        mk[c] = z["masks"]; names = [str(x) for x in z["label_names"]]
ticks0 = static_ticks(ref, a.n_static)
P0, C0, L0, K0, V0 = scene(ref, cal, ticks0, mk, above=-0.002)   # labelled: keep flat parts touching the table
n_t, d_t = ref.table_plane()
print(f"ref {a.ref} ticks {ticks0}: {len(P0)} scene points", flush=True)
lib = {}
for name in a.objects:
    lid = names.index(name)
    sel = L0 == lid
    P, C, cnt = voxel_mean(P0[sel], C0[sel], 0.003)
    P, C = P[cnt >= 1], C[cnt >= 1]
    # clean: largest cluster within 1.5 cm
    pc = objlib.to_pcd(P, C)
    lab = np.asarray(pc.cluster_dbscan(eps=0.012, min_points=8))
    if (lab >= 0).any():
        big = np.bincount(lab[lab >= 0]).argmax()
        P, C = P[lab == big], C[lab == big]
    cm = objlib.colour_model(C, C0[~sel & (L0 != lid)])
    lib[name] = dict(P=P, C=C, cm=cm, scans=[("ref", a.ref, np.eye(4), len(P), 1.0, 0.0)], parts=[(P, C)], views=list(V0))
    print(f"  {name}: ref partial scan {len(P)} pts", flush=True)

for e in a.episodes:
    ep = Episode(os.path.join(a.data, e))
    ticks = static_ticks(ep, a.n_static)
    P1, C1, _, _, V1 = scene(ep, cal, ticks)
    Pv, Cv, _ = voxel_mean(P1, C1, 0.003)
    for name, o in lib.items():
        pr = o["cm"](Cv)
        cand = pr > 0.3
        if cand.sum() < 50:
            print(f"  {e} {name}: no colour candidates", flush=True); continue
        T, fit, rmse = objlib.register(o["P"], o["C"], Pv[cand], Cv[cand], voxel=0.005)
        # refine on all points (colour gate softer)
        T, fit2, rmse2 = objlib.register(o["P"], o["C"], Pv[pr > 0.15], Cv[pr > 0.15], voxel=0.004, init=T)
        ok = fit2 > 0.55 and rmse2 < 0.004
        print(f"  {e} {name}: fitness {fit2:.2f} rmse {rmse2*1000:.1f} mm -> {'OK' if ok else 'rejected'}", flush=True)
        if not ok:
            continue
        Ti = np.linalg.inv(T)
        Pr = o["P"] @ T[:3, :3].T + T[:3, 3]
        dd, _ = objlib.cKDTree(Pr).query(Pv, distance_upper_bound=0.004)
        near = np.isfinite(dd) & (pr > 0.4)
        Pc = Pv[near] @ Ti[:3, :3].T + Ti[:3, 3]
        o["parts"].append((Pc, Cv[near]))
        o["views"] += [(D_, K_, Ti @ T_) for D_, K_, T_ in V1]
        o["scans"].append(("reg", e, T, int(near.sum()), fit2, rmse2))

os.makedirs(a.out, exist_ok=True)
for name, o in lib.items():
    P = np.concatenate([p for p, _ in o["parts"]]); C = np.concatenate([c for _, c in o["parts"]])
    P, C, cnt = voxel_mean(P, C, a.voxel)
    # support: a voxel must be seen by >= 2 points overall (removes isolated noise)
    keep = cnt >= (2 if len(o["parts"]) > 2 else 1)
    P, C = P[keep], C[keep]
    pc = objlib.to_pcd(P, C)
    _, ind = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    P, C = P[ind], C[ind]
    lo = P.min(0) - 0.015; hi = P.max(0) + 0.015
    occ, lo, vx, seen = objlib.carve(o["views"], lo, hi, voxel=a.voxel, plane=(n_t, d_t))
    occ = objlib.object_occupancy(occ, lo, vx, P, near=0.012)
    occ = objlib.footprint_fill(occ, lo, vx, P, (n_t, d_t))
    mesh = objlib.occupancy_mesh(occ, lo, vx, seed_pts=P)
    # how much of the final surface was actually observed (a fused point within 5 mm)
    dsurf, _ = objlib.cKDTree(P).query(mesh.vertices)
    observed_frac = float((dsurf < 0.005).mean())
    # canonical object frame: centroid + PCA axes of the mesh, z = table normal side kept (world-aligned at ref t0)
    ctr = mesh.centroid.copy()
    T_world_obj = np.eye(4); T_world_obj[:3, 3] = ctr
    mesh.apply_translation(-ctr)
    Pobj = P - ctr
    # vertex colours: nearest fused point
    _, j = objlib.cKDTree(Pobj).query(mesh.vertices)
    mesh.visual.vertex_colors = np.c_[(C[j] * 255).astype(np.uint8), np.full(len(j), 255, np.uint8)]
    # surface-to-depth residual of every scan (scan points in the object frame -> mesh surface)
    for si, (Pp, _) in enumerate(o["parts"]):
        dres = objlib.surface_residual(mesh, Pp - ctr)
        sc = list(o["scans"][si]); sc.append(float(np.median(dres))); sc.append(float(np.percentile(dres, 90)))
        o["scans"][si] = tuple(sc)
    meta = {"task": a.task, "name": name, "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "watertight": bool(mesh.is_watertight), "volume_m3": float(mesh.volume), "extents_m": mesh.extents.tolist(),
            "n_vertices": int(len(mesh.vertices)), "n_faces": int(len(mesh.faces)), "fused_points": int(len(P)),
            "frame": "object frame = mesh centroid, axes parallel to the world (left arm base) frame of the reference episode at its first tick",
            "T_world_obj_ref": T_world_obj.tolist(), "reference_episode": a.ref, "calibration": cal.name,
            "shape": "space carving (top+side depth of the static ticks of every registered episode; table plane as floor), largest component containing the observed points; unseen regions stay filled (conservative)",
            "observed_surface_frac": observed_frac, "n_views": len(o["views"]),
            "scans": [{"kind": k, "episode": e, "fitness": f, "rmse_mm": r * 1000, "points": n_,
                       "surf_res_med_mm": rm * 1000, "surf_res_p90_mm": r9 * 1000,
                       "T_world_obj_t0": (np.asarray(T) @ T_world_obj).tolist()} for k, e, T, n_, f, r, rm, r9 in o["scans"]],
            "n_scans": len(o["scans"]),
            "surf_res_med_mm_all": float(np.median([x[6] for x in o["scans"]]) * 1000)}
    objlib.save_object(mesh, meta, a.out, name)
    np.savez_compressed(os.path.join(a.out, f"{name}_points.npz"), P=Pobj.astype(np.float32), C=C.astype(np.float32))
    print(f"saved {name}: watertight {mesh.is_watertight}, {len(mesh.faces)} faces, extents {np.round(mesh.extents, 3)}, "
          f"{len(o['scans'])} scans, surface->depth residual median {meta['surf_res_med_mm_all']:.1f} mm", flush=True)
