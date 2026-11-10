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
ap.add_argument("--start", type=int, default=0); ap.add_argument("--stop", type=int, default=None); ap.add_argument("--cold", action="store_true", default=True)
ap.add_argument("--cloth_tracks", default=os.path.expanduser("~/point_tracker/reports/REALDATA_SE3_1108/tracks/cloth_greycloth.npz"))
ap.add_argument("--cloth_valid_until", type=int, default=160, help="cloth v1 is valid before the fold (report table)")
ap.add_argument("--skip_existing", action="store_true")
ap.add_argument("--online-seg", action="store_true", help="run the same current-frame segmenter used live instead of replaying cached masks")
ap.add_argument("--sam-ckpt",default="/home/howardji/realpc/.model_cache/sam2/sam2.1_hiera_small.pt")
ap.add_argument("--seg_dir", default=os.path.expanduser("~/data_ext/realpc_v4/seg"), help="causal object masks (segment_objects.py)")
a = ap.parse_args()

cfg = load_config(os.path.join(ROOT, "configs", "yam_bimanual.json")); r = cfg["robot"]
tarm = TorchArm(os.path.join(cfg["_dir"], r["urdf"]), r["arm_joints"], r["finger_joints"])
cal = RigCalib.load(a.calib)
commit = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
def max_displacement_mm(T):
    valid=np.isfinite(T).all((1,2));T=T[valid]
    return float(np.max(np.linalg.norm(T[:,:3,3]-T[0,:3,3],axis=1))*1000) if len(T) else None

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
    segmenter=None
    if a.online_seg:
        from sam2.build_sam import build_sam2_video_predictor
        from realpc.online_masks import OnlineObjectSegmenter
        model=build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_s.yaml",a.sam_ckpt,device="cuda")
        conv.seg={};segmenter=OnlineObjectSegmenter(conv,model)
    T_total = len(ep.low["t_cap"]); stop = min(a.stop or T_total, T_total)
    ticks = np.arange(a.start, stop)
    cams=["top","side"]+([c for c in ("left_wrist","right_wrist") if os.path.exists(os.path.join(ep.path,c+"_depth.mkv"))] if ep.task=="sweep" and os.environ.get("REALPC_BROOM_MATERIAL","0")=="1" else [])
    ds = {c: DepthStream(os.path.join(ep.path, f"{c}_depth.mkv"), a.start) for c in cams}
    caps = {c: cv2.VideoCapture(ep.video(c)) for c in cams}
    for c in caps:
        caps[c].set(cv2.CAP_PROP_POS_FRAMES, a.start)
    sc = {c: float(ep.meta["depth_scale"][c]) for c in cams}
    P = len(conv.objs)
    names = [o.name for o in conv.objs]
    robot_T = np.zeros((len(ticks), len(arms.bodies), 4, 4), np.float32)
    obj_T = np.full((P, len(ticks), 4, 4), np.nan, np.float32); obj_res = np.full((P, len(ticks), 2), np.nan, np.float32)
    obj_held = np.zeros((P, len(ticks)), np.int8)
    obj_support = np.zeros((P, len(ticks), 2), np.int32); obj_mode = np.full((P, len(ticks)), "", dtype="<U16")
    broom_diagnostics=[]
    qL = np.zeros((len(ticks), 7)); qR = np.zeros((len(ticks), 7))
    for k, t in enumerate(ticks):
        frames = {}
        for c in cams:
            _, D = ds[c].next(); ok, im = caps[c].read()
            frames[c] = (D.astype(np.float32) * sc[c], im)
        labels=segmenter.step(int(t),frames) if segmenter is not None else None
        o = conv.step(int(t), frames,labels=labels)
        if hasattr(conv.trk,"broom_refiner"):broom_diagnostics.append({"tick":int(t),**conv.trk.broom_refiner.stats})
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
               obj_names=np.array(names), obj_static=np.array([bool(conv.metas[o.name].get("immobile",False)) for o in conv.objs]), obj_T=obj_T, obj_res=obj_res, obj_held=obj_held,
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
    if ep.task == "greycloth" and os.path.exists(a.cloth_tracks) and ep.name == "episode_0000":
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
    if broom_diagnostics:json.dump(broom_diagnostics,open(out_p[:-4]+"_broom.json","w"),indent=1)
    qa = {"episode": f"{ep.task}/{ep.name}", "ticks": int(len(ticks)), "seconds": round(time.time() - t0, 1),
          "n_robot": int(len(arms.local)), "n_static": int(len(res["static_points"])),
          "n_object": int(sum(len(o.samples) for o in conv.objs)), "n_cloth": int(res["cloth_points"].shape[0]) if "cloth_points" in res else 0,
          "objects": {n: {"init": conv.init_info.get(n), "res_med_mm": float(np.nanmedian(obj_res[i, :, 0]) * 1000) if np.isfinite(obj_res[i, :, 0]).any() else None,
                          "res_p90_mm": float(np.nanpercentile(obj_res[i, :, 0], 90) * 1000) if np.isfinite(obj_res[i, :, 0]).any() else None,
                          "held_ticks": int((obj_held[i] > 0).sum()),
                          "max_disp_mm": max_displacement_mm(obj_T[i])} for i, n in enumerate(names)}}
    json.dump(qa, open(out_p[:-4] + "_qa.json", "w"), indent=1)
    print(json.dumps(qa), flush=True)
json.dump(SCHEMA, open(os.path.join(a.out_dir, "schema.json"), "w"), indent=1)
