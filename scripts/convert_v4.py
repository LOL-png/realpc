#!/usr/bin/env python
"""Batch conversion of episodes with the v4 streaming converter (realpc/v4.py) -> one npz + QA json per episode.

  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python scripts/convert_v4.py --episodes <ep dir> ... \
      --out_dir ~/data_ext/realpc_out/v1 [--calib configs/calib_yam_rig_v3.json] [--start S --stop E] [--cold]
Library per task: assets/objects/<task>/ (no library = robot + static only; greycloth adds the cloth v1 tracks).
"""
import argparse
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from realpc.calib_rig import RigCalib  # noqa: E402
from realpc.config import build_arms, load_config  # noqa: E402
from realpc.diffcalib import TorchArm  # noqa: E402
from realpc.io import DepthStream, Episode  # noqa: E402
from realpc.v4 import SCHEMA, StreamConverter  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", nargs="+", required=True); ap.add_argument("--out_dir", required=True)
ap.add_argument("--calib", default=os.path.join(ROOT, "configs", "calib_yam_rig_v3.json"))
ap.add_argument("--assets", default=os.path.join(ROOT, "assets", "objects"))
ap.add_argument("--start", type=int, default=0); ap.add_argument("--stop", type=int, default=None); ap.add_argument("--cold", action="store_true")
ap.add_argument("--cloth_tracks", default=os.path.expanduser("~/point_tracker/reports/REALDATA_SE3_1108/tracks/cloth_greycloth.npz"))
ap.add_argument("--cloth_valid_until", type=int, default=160, help="cloth v1 is valid before the fold (report table)")
ap.add_argument('--cloth_cot_inputs',help='Zero-lookahead CoTracker current packets')
ap.add_argument("--cloth_causal_inputs", help="Current-frame RGB/depth/FK/mask packets for CausalClothTracker; overrides old cloth export")
ap.add_argument("--cloth_causal_config", help="cloth_causal_track config.json (same tracker parameters as live inference)")
ap.add_argument("--skip_existing", action="store_true")
ap.add_argument("--seg_dir", default=os.path.expanduser("~/data_ext/realpc_v4/seg"), help="causal object masks (segment_objects.py)")
a = ap.parse_args()

cfg = load_config(os.path.join(ROOT, "configs", "yam_bimanual.json")); r = cfg["robot"]
tarm = TorchArm(os.path.join(cfg["_dir"], r["urdf"]), r["arm_joints"], r["finger_joints"])
cal = RigCalib.load(a.calib)
commit = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
LOW_KEYS = ["eef_pos_left", "eef_quat_left", "gripper_left", "action_joint_pos_left", "action_eef_pos_left", "action_eef_quat_left",
            "action_eef_delta_left", "action_gripper_left", "engaged_left", "eef_pos_right", "eef_quat_right", "gripper_right",
            "action_joint_pos_right", "action_eef_pos_right", "action_eef_quat_right", "action_eef_delta_right",
            "action_gripper_right", "engaged_right", "joint_pos_left", "joint_pos_right"]

for epp in a.episodes:
    ep = Episode(epp)
    out_p = os.path.join(a.out_dir, ep.task, f"{ep.name}.npz")
    if a.skip_existing and os.path.exists(out_p):
        continue
    t0 = time.time()
    arms = build_arms(cfg, ep.calib)
    lib = os.path.join(a.assets, ep.task)
    if not os.path.isdir(lib):
        lib = os.path.join(a.out_dir, "_empty_lib"); os.makedirs(lib, exist_ok=True)
    conv = StreamConverter(ep, cal, tarm, arms, lib, start=a.start, cold=a.cold, seg_dir=a.seg_dir)
    T_total = len(ep.low["t_cap"]); stop = min(a.stop or T_total, T_total)
    ticks = np.arange(a.start, stop)
    ds = {c: DepthStream(os.path.join(ep.path, f"{c}_depth.mkv"), a.start) for c in ("top", "side")}
    caps = {c: cv2.VideoCapture(ep.video(c)) for c in ("top", "side")}
    for c in caps:
        caps[c].set(cv2.CAP_PROP_POS_FRAMES, a.start)
    sc = {c: float(ep.meta["depth_scale"][c]) for c in ("top", "side")}
    P = len(conv.objs)
    names = [o.name for o in conv.objs]
    robot_T = np.zeros((len(ticks), len(arms.bodies), 4, 4), np.float32)
    obj_T = np.full((P, len(ticks), 4, 4), np.nan, np.float32); obj_res = np.full((P, len(ticks), 2), np.nan, np.float32)
    obj_held = np.zeros((P, len(ticks)), np.int8)
    obj_support = np.zeros((P, len(ticks), 2), np.int32); obj_mode = np.full((P, len(ticks)), "", dtype="<U16")
    qL = np.zeros((len(ticks), 7)); qR = np.zeros((len(ticks), 7))
    cloth = None; cloth_samples=[]; cloth_observed=[]; cloth_metrics=[]
    if ep.task == 'greycloth' and a.cloth_causal_inputs:
        if a.start != 0:raise ValueError('Causal cloth conversion must start at t0 or restore its past state')
        from realpc.cloth_causal import CausalClothTracker, at_image
        from realpc.cloth_inputs import project
        options=json.load(open(a.cloth_causal_config)) if a.cloth_causal_config else {}
        cloth=CausalClothTracker.from_config(options)
    for k, t in enumerate(ticks):
        frames = {}
        for c in ("top", "side"):
            _, D = ds[c].next(); ok, im = caps[c].read()
            frames[c] = (D.astype(np.float32) * sc[c], im)
        o = conv.step(int(t), frames)
        if cloth is not None:
            # Identical live step API: never preload a future cloth packet.
            with np.load(os.path.join(a.cloth_causal_inputs,'frames',f'{t:04d}.npz')) as z:
                packet={key:z[key] for key in z.files}
            if a.cloth_cot_inputs:
                with np.load(os.path.join(a.cloth_cot_inputs,f'{t:04d}.npz')) as z:packet.update({key:z[key] for key in z.files})
            _,cm,views=cloth.step(int(t),packet,iters=options.get('iters',5))
            S=cloth.cm.samples()[:4000];observed=np.zeros(len(S),bool)
            for c in ('top','side'):
                uv,zz=project(S,views[c]['K'],views[c]['T']);dd,inside=at_image(views[c]['D'],uv)
                observed |= inside&(dd>.08)&(np.abs(zz-dd)<.012)
            cloth_samples.append(S.astype(np.float32));cloth_observed.append(observed);cloth_metrics.append(cm)
        if k % 50 == 0:
            print(f"{ep.task}/{ep.name} tick={t} elapsed={time.time()-t0:.1f}s", flush=True)
        robot_T[k] = o["robot_T"]; qL[k] = o["q"]["left"]; qR[k] = o["q"]["right"]
        for i, n in enumerate(names):
            obj_T[i, k] = o["obj_T"][n]; obj_res[i, k] = o["obj_res"][n]; obj_held[i, k] = o["obj_held"][n]
            obj_support[i, k] = o.get("obj_support", {}).get(n, (0, 0)); obj_mode[i, k] = o.get("obj_mode", {}).get(n, "")
    for c in ds:
        ds[c].close()
    res = dict(ticks=ticks, t_cap=ep.low["t_cap"][ticks], q_left=qL.astype(np.float32), q_right=qR.astype(np.float32),
               robot_local=arms.local.astype(np.float32), robot_body=arms.body.astype(np.int16),
               robot_bodies=np.array([f"{x}/{y}" for x, y in arms.bodies]), robot_T=robot_T,
               obj_names=np.array(names), obj_T=obj_T, obj_res=obj_res, obj_held=obj_held,
               obj_valid=np.isfinite(obj_T[:, :, 0, 0]), obj_support=obj_support, obj_mode=obj_mode,
               static_points=conv.static[0], static_color=conv.static[1],
               task=ep.task, episode=ep.name, calib=os.path.basename(a.calib), const_lag=cal.const_lag, commit=commit,
               causal=True, start=a.start, cold=a.cold)
    for kk in LOW_KEYS:
        if kk in ep.low:
            res["low_" + kk] = ep.low[kk][ticks]
    if conv.objs:
        res["obj_local"] = np.concatenate([o.samples for o in conv.objs]).astype(np.float32)
        res["obj_index"] = np.concatenate([np.full(len(o.samples), i) for i, o in enumerate(conv.objs)]).astype(np.int16)
        cols = []
        for o in conv.objs:
            vc = np.asarray(o.mesh.visual.vertex_colors)[:, :3] if hasattr(o.mesh.visual, "vertex_colors") else np.full((len(o.mesh.vertices), 3), 128)
            _, j = __import__("scipy.spatial", fromlist=["cKDTree"]).cKDTree(o.mesh.vertices).query(o.samples)
            cols.append(vc[j])
        res["obj_color"] = np.concatenate(cols).astype(np.uint8)
    # cloth (greycloth): cloth v1 tracks (old recorder calibration) mapped into the v3 world: every camera-measured
    # point moves by the same rig correction C = T_world_top(v3) . T_world_top(recorder)^-1
    if cloth is not None:
        res['cloth_points']=np.stack(cloth_samples,axis=1)
        res['cloth_observed']=np.stack(cloth_observed,axis=1)
        # Whole-frame gate remains conservative during unverified material folds.
        res['cloth_valid']=np.array([min(m['cameras'][c]['surface_coverage_10mm'] for c in ('top','side'))>.85
            and m['edge_ratio_p5_p50_p95'][2]<1.35 for m in cloth_metrics])
        res['cloth_variant']=np.array('causal_material_mesh_v2');res['cloth_coordinate_frame']=np.array('v3_world')
        res['cloth_geometry_observed_only']=np.array(True)
        from scipy.spatial import cKDTree
        keep=cKDTree(res['cloth_points'][:,0]).query(res['static_points'])[0]>.015
        res['static_points'],res['static_color']=res['static_points'][keep],res['static_color'][keep]
    elif ep.task == "greycloth" and os.path.exists(a.cloth_tracks) and ep.name == "episode_0000":
        d = np.load(a.cloth_tracks, allow_pickle=True)
        cl = d["group"] == 4
        Pc = d["pos"][cl]; ct = d["ticks"]
        C = cal.T_world_cam("top") @ np.linalg.inv(RigCalib.recorder(ep).T_world_cam("top"))
        idx = np.clip(np.searchsorted(ct, ticks), 0, len(ct) - 1)
        Pk = Pc[:, idx]
        Pk = Pk @ C[:3, :3].T.astype(np.float32) + C[:3, 3].astype(np.float32)
        res["cloth_points"] = Pk.astype(np.float32)
        res["cloth_valid"] = (ticks < a.cloth_valid_until) & (ticks >= ct[0]) & np.isfinite(Pk).all((0, 2))
        # the cloth lies on the table at the first tick: those points are cloth, not static scene
        from scipy.spatial import cKDTree
        k0 = int(np.argmax(np.isfinite(Pk).all((0, 2))))
        dd, _ = cKDTree(Pk[:, k0]).query(res["static_points"], distance_upper_bound=0.015)
        keep = ~np.isfinite(dd)
        res["static_points"], res["static_color"] = res["static_points"][keep], res["static_color"][keep]
    os.makedirs(os.path.dirname(out_p), exist_ok=True)
    np.savez_compressed(out_p, **res)
    if cloth is not None:
        json.dump(cloth_metrics,open(out_p.replace('.npz','_cloth_metrics.json'),'w'))
    qa = {"episode": f"{ep.task}/{ep.name}", "ticks": int(len(ticks)), "seconds": round(time.time() - t0, 1),
          "n_robot": int(len(arms.local)), "n_static": int(len(res["static_points"])),
          "n_object": int(sum(len(o.samples) for o in conv.objs)), "n_cloth": int(res["cloth_points"].shape[0]) if "cloth_points" in res else 0,
          "objects": {n: {"init": conv.init_info.get(n), "res_med_mm": float(np.nanmedian(obj_res[i, :, 0]) * 1000) if np.isfinite(obj_res[i, :, 0]).any() else None,
                          "res_p90_mm": float(np.nanpercentile(obj_res[i, :, 0], 90) * 1000) if np.isfinite(obj_res[i, :, 0]).any() else None,
                          "held_ticks": int((obj_held[i] > 0).sum()),
                          "max_disp_mm": float(np.nanmax(np.linalg.norm(obj_T[i, :, :3, 3] - obj_T[i, 0, :3, 3], axis=1)) * 1000)} for i, n in enumerate(names)}}
    json.dump(qa, open(out_p[:-4] + "_qa.json", "w"), indent=1)
    print(json.dumps(qa), flush=True)
json.dump(SCHEMA, open(os.path.join(a.out_dir, "schema.json"), "w"), indent=1)
