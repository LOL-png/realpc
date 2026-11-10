#!/usr/bin/env python
"""Batch conversion of episodes with the v4 streaming converter (realpc/v4.py) -> one npz + QA json per episode.

  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python scripts/convert_v4.py --episodes <ep dir> ... \
      --out_dir ~/data_ext/realpc_out/v1 [--calib configs/calib_yam_rig_v3.json] [--start S --stop E] [--cold]
Library per task: assets/objects/<task>/ (no library = robot + static only; greycloth adds the cloth v1 tracks).
"""
import argparse
import hashlib
from pathlib import Path
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
ap.add_argument('--cloth_cot_inputs',help='Zero-lookahead CoTracker current packets')
ap.add_argument("--cloth_causal_inputs", help="Current-frame RGB/depth/FK/mask packets for CausalClothTracker; overrides old cloth export")
ap.add_argument("--cloth_causal_config", help="cloth_causal_track config.json (same tracker parameters as live inference)")
ap.add_argument("--skip_existing", action="store_true")
ap.add_argument("--preview-every",type=int,default=0,help="save atomic partial pose snapshots for quick visual validation")
ap.add_argument("--save-seg-dir",help="optional QA cache of labels used at each tick; export excluded from timing")
ap.add_argument("--sam-interval",type=int,default=0,help="Periodic SAM refresh plus failures; 0=events only")
ap.add_argument("--depth-mesh",action="store_true",help="depth/mesh tracking; SAM only at initialization or failed-fit recovery")
ap.add_argument("--online-seg", action="store_true", help="run the same current-frame segmenter used live instead of replaying cached masks")
ap.add_argument("--sam-ckpt",default="/home/howardji/realpc/.model_cache/sam2/sam2.1_hiera_small.pt")
ap.add_argument("--seg_dir", default=os.path.expanduser("~/data_ext/realpc_v4/seg"), help="causal object masks (segment_objects.py)")
a = ap.parse_args()
if a.depth_mesh:
    if a.online_seg:ap.error("--depth-mesh and --online-seg are separate modes")
    os.environ["REALPC_DEPTH_MESH"]="1"
    os.environ["REALPC_SAM_INTERVAL"]=str(a.sam_interval)

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
    conv = StreamConverter(ep, cal, tarm, arms, lib, start=a.start, cold=a.cold, seg_dir=None if a.depth_mesh else a.seg_dir)
    segmenter=None;depth_pipeline=None
    if a.depth_mesh:
        from realpc.depth_pipeline import DepthMeshPipeline
        depth_pipeline=DepthMeshPipeline(conv,a.sam_ckpt)
    if a.online_seg:
        from sam2.build_sam import build_sam2_video_predictor
        from realpc.online_masks import OnlineObjectSegmenter
        model=build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_s.yaml",a.sam_ckpt,device="cuda")
        conv.seg={};segmenter=OnlineObjectSegmenter(conv,model)
    os.makedirs(os.path.dirname(out_p),exist_ok=True)
    source_files=list(Path(ROOT,"realpc").glob("*.py"))+[Path(a.calib)]+list(Path(lib).glob("*.ply"))+list(Path(lib).glob("*.json"))
    run_manifest={"commit":commit,"initialization":"current RGB-D","online_seg":a.online_seg,"depth_mesh":a.depth_mesh,"start":a.start,"code_asset_sha256":{str(p.relative_to(Path(ROOT).resolve())) if p.is_relative_to(Path(ROOT).resolve()) else str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},"settings":{k:os.environ.get(k) for k in ["REALPC_SAM_INTERVAL","REALPC_SIL","REALPC_CAM_EQUAL","REALPC_SIL_METHOD","REALPC_COLD_COMPONENTS","REALPC_BROOM_MATERIAL","REALPC_BROOM_CLEAN_MASK","REALPC_BROOM_HOLE","REALPC_WRIST_WEIGHT","REALPC_BROOM_REJECT_BG"]}}
    json.dump(run_manifest,open(out_p[:-4]+"_run.json","w"),indent=2)
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
    broom_diagnostics=[];step_times=[];input_times=[];used_masks={}
    qL = np.zeros((len(ticks), 7)); qR = np.zeros((len(ticks), 7))
    cloth = None; cloth_pipeline=None; cloth_samples=[]; cloth_observed=[]; cloth_metrics=[]
    if ep.task == 'greycloth' and a.cloth_causal_inputs:
        if a.start != 0:raise ValueError('Causal cloth conversion must start at t0 or restore its past state')
        from realpc.cloth_causal import CausalClothTracker, at_image
        from realpc.cloth_inputs import project
        options=json.load(open(a.cloth_causal_config)) if a.cloth_causal_config else {}
        if options.get('mask_interval') is not None:
            if a.cloth_cot_inputs:raise ValueError('Sparse cloth pipeline must generate its own tracks; omit cached cloth_cot_inputs')
            from realpc.cloth_pipeline import CausalClothPipeline
            cloth_pipeline=CausalClothPipeline(options);cloth=cloth_pipeline.tracker
        else:cloth=CausalClothTracker.from_config(options)
    for k, t in enumerate(ticks):
        input_start=time.perf_counter()
        frames = {}
        for c in cams:
            _, D = ds[c].next(); ok, im = caps[c].read()
            frames[c] = (D.astype(np.float32) * sc[c], im)
        input_times.append(time.perf_counter()-input_start);step_start=time.perf_counter()
        labels=segmenter.step(int(t),frames) if segmenter is not None else None
        o = depth_pipeline.step(int(t),frames) if depth_pipeline is not None else conv.step(int(t), frames,labels=labels)
        step_times.append(time.perf_counter()-step_start)
        if a.save_seg_dir:
            for v in conv.trk.prev:
                if v.labels is not None:
                    used_masks.setdefault(v.name,[]).append(cv2.resize(v.labels,(640,360 if v.name=="top" else 480),interpolation=0))
        if hasattr(conv.trk,"broom_refiner"):broom_diagnostics.append({"tick":int(t),**conv.trk.broom_refiner.stats})
        if cloth is not None:
            # Identical live step API: never preload a future cloth packet.
            with np.load(os.path.join(a.cloth_causal_inputs,'frames',f'{t:04d}.npz')) as z:
                packet={key:z[key] for key in z.files}
            if a.cloth_cot_inputs:
                with np.load(os.path.join(a.cloth_cot_inputs,f'{t:04d}.npz')) as z:packet.update({key:z[key] for key in z.files})
            if cloth_pipeline is not None:_,cm,views=cloth_pipeline.step(int(t),packet)
            else:_,cm,views=cloth.step(int(t),packet,iters=options.get('iters',5))
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
        if a.preview_every and k>0 and (k+1)%a.preview_every==0:
            temporary=out_p[:-4]+"_partial_tmp.npz";preview=out_p[:-4]+"_partial.npz"
            np.savez_compressed(temporary,ticks=ticks[:k+1],obj_names=np.array(names),obj_T=obj_T[:,:k+1],obj_valid=np.isfinite(obj_T[:,:k+1,0,0]),q_left=qL[:k+1].astype(np.float32),q_right=qR[:k+1].astype(np.float32),task=ep.task,episode=ep.name,partial=True)
            os.replace(temporary,preview)
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
    if broom_diagnostics:json.dump(broom_diagnostics,open(out_p[:-4]+"_broom.json","w"),indent=1)
    if cloth is not None:
        json.dump(cloth_metrics,open(out_p.replace('.npz','_cloth_metrics.json'),'w'))
        if cloth_pipeline is not None:json.dump(cloth_pipeline.summary(),open(out_p.replace('.npz','_cloth_pipeline.json'),'w'),indent=2)
    qa = {"episode": f"{ep.task}/{ep.name}", "ticks": int(len(ticks)), "seconds": round(time.time() - t0, 1),
          "n_robot": int(len(arms.local)), "n_static": int(len(res["static_points"])),
          "n_object": int(sum(len(o.samples) for o in conv.objs)), "n_cloth": int(res["cloth_points"].shape[0]) if "cloth_points" in res else 0,
          "objects": {n: {"init": conv.init_info.get(n), "res_med_mm": float(np.nanmedian(obj_res[i, :, 0]) * 1000) if np.isfinite(obj_res[i, :, 0]).any() else None,
                          "res_p90_mm": float(np.nanpercentile(obj_res[i, :, 0], 90) * 1000) if np.isfinite(obj_res[i, :, 0]).any() else None,
                          "held_ticks": int((obj_held[i] > 0).sum()),
                          "max_disp_mm": max_displacement_mm(obj_T[i])} for i, n in enumerate(names)}}
    qa["timing"]={"step_seconds":sum(step_times),"step_fps":len(step_times)/sum(step_times),"steady_median_ms":float(np.median(step_times[1:])*1000),"steady_p90_ms":float(np.percentile(step_times[1:],90)*1000),"input_seconds":sum(input_times),"end_to_end_fps":len(ticks)/qa["seconds"]}
    if depth_pipeline is not None:qa["depth_pipeline"]=depth_pipeline.summary()
    json.dump(qa, open(out_p[:-4] + "_qa.json", "w"), indent=1)
    print(json.dumps(qa), flush=True)
    if a.save_seg_dir:
        target=Path(a.save_seg_dir)/ep.task;target.mkdir(parents=True,exist_ok=True)
        for c,labels_used in used_masks.items():
            np.savez_compressed(target/(ep.name+'_'+c+'.npz'),labels=np.stack(labels_used),start=a.start,names=np.array(names),source='depth_geometry' if a.depth_mesh else 'SAM_current')
json.dump(SCHEMA, open(os.path.join(a.out_dir, "schema.json"), "w"), indent=1)
