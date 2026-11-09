#!/usr/bin/env python
"""Side-by-side calibration verification video + metrics: FK robot meshes rendered over the top and side images,
recorder calibration (lag 0) vs a v3 calibration (per-arm camera poses + fitted lag).

  python scripts/calib_video.py --episode <ep dir> --calib configs/calib_yam_rig_v3.json --out video.mp4 \
      [--masks <dir with {top,side}_masks.npz>] [--stride 2] [--json metrics.json] [--stills <dir>]
Metrics (per camera / arm, before vs after): IoU vs SAM robot masks (if given), and arm depth |dz| (median over
rendered arm pixels with a valid depth reading within 10 cm of the render, or on SAM robot pixels when masks are given).
"""
import argparse
import json
import os
import subprocess
import sys

import cv2
import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from realpc.calib_rig import RigCalib  # noqa: E402
from realpc.config import load_config  # noqa: E402
from realpc.diffcalib import Renderer, TorchArm  # noqa: E402
from realpc.io import Episode, read_depth_frames  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--episode", required=True); ap.add_argument("--calib", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--masks", default=None); ap.add_argument("--stride", type=int, default=2); ap.add_argument("--json", default=None)
ap.add_argument("--stills", default=None); ap.add_argument("--still_ticks", type=int, nargs="*", default=None)
ap.add_argument("--max_frames", type=int, default=100000)
a = ap.parse_args()

ep = Episode(a.episode)
cfg = load_config(os.path.join(ROOT, "configs", "yam_bimanual.json")); r = cfg["robot"]
arm = TorchArm(os.path.join(cfg["_dir"], r["urdf"]), r["arm_joints"], r["finger_joints"])
new = RigCalib.load(a.calib)
old = RigCalib.recorder(ep)
cams = ("top", "side")
T = len(ep.low["t_cap"])
ticks = list(range(0, T, a.stride))[: a.max_frames]
masks = {}
if a.masks:
    for c in cams:
        p = os.path.join(a.masks, f"{c}_masks.npz")
        if os.path.exists(p):
            masks[c] = np.load(p)["masks"]
depth = {c: read_depth_frames(os.path.join(ep.path, f"{c}_depth.mkv"), ticks) for c in cams}
dscale = {c: float(ep.meta.get("depth_scale", {}).get(c, 0.001)) for c in cams}
caps = {c: cv2.VideoCapture(ep.video(c)) for c in cams}
rend = {}
size = {"top": (360, 640), "side": (480, 640)}
for c in cams:
    K = ep.intrinsics(c).copy()
    H0 = 720 if c == "top" else 480
    s = size[c][0] / H0
    K[:2] *= s
    rend[c] = (Renderer(K, *size[c]), s)
COL = {"left": (0, 0, 255), "right": (255, 0, 255)}
stats = {f"{lab}/{c}/{arm_}": {"iou": [], "dz": []} for lab in ("before", "after") for c in cams for arm_ in ("left", "right")}


@torch.no_grad()
def render(cal, c, t):
    R, s = rend[c]
    out = {}
    for arm_ in ("left", "right"):
        tq = cal.robot_tick(ep, c, t)
        q = cal.q_at(ep, arm_, tq)
        V = arm.verts(torch.tensor(q[None], dtype=torch.float32, device="cuda"))   # q_at applies the joint offsets
        M = torch.tensor(cal.T_cam_base(c, arm_), dtype=torch.float32, device="cuda")
        Vc = V @ M[:3, :3].T + M[:3, 3]
        sil, z = R.raster(Vc, arm.faces)
        out[arm_] = (sil[0].cpu().numpy() > 0.5, z[0].cpu().numpy())
    return out


enc = None
frames_imgs = {c: {} for c in cams}
for fi, t in enumerate(ticks):
    imgs = {}
    for c in cams:
        caps[c].set(cv2.CAP_PROP_POS_FRAMES, t) if fi == 0 or a.stride > 1 else None
        ok, im = caps[c].read()
        if a.stride > 1:
            for _ in range(a.stride - 1):
                caps[c].grab()
        imgs[c] = cv2.resize(im, (size[c][1], size[c][0]))
    tiles = []
    for c in cams:
        row = []
        D = depth[c].get(t)
        Dm = None if D is None else cv2.resize(D.astype(np.float32) * dscale[c], (size[c][1], size[c][0]), interpolation=cv2.INTER_NEAREST)
        M = None
        if c in masks and t < len(masks[c]):
            M = cv2.resize(masks[c][t], (size[c][1], size[c][0]), interpolation=cv2.INTER_NEAREST)
        for lab, cal in (("before", old), ("after", new)):
            rd = render(cal, c, t)
            im = imgs[c].copy()
            for arm_, (m, z) in rd.items():
                col = np.array(COL[arm_], np.uint8)
                im[m] = (0.35 * col + 0.65 * im[m]).astype(np.uint8)
                cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(im, cs, -1, COL[arm_], 1)
                st = stats[f"{lab}/{c}/{arm_}"]
                if M is not None:
                    gt = M == (2 if arm_ == "left" else 3)
                    if gt.sum() > 200 and m.sum() > 200:
                        st["iou"].append(float((gt & m).sum() / (gt | m).sum()))
                if Dm is not None and m.sum() > 200:
                    sel = m & (Dm > 0.1) & (z > 0)
                    if M is not None:
                        sel &= (M == (2 if arm_ == "left" else 3))
                    dz = Dm[sel] - z[sel]
                    dz = dz[np.abs(dz) < 0.1]
                    if len(dz) > 100:
                        st["dz"].append(float(np.median(np.abs(dz))))
            if M is not None:
                for lid, col in ((2, (0, 255, 0)), (3, (255, 255, 0))):
                    cs, _ = cv2.findContours((M == lid).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    cv2.drawContours(im, cs, -1, col, 1)
            lagtxt = f" lag {cal.lag(ep, c, t):+.1f}" if lab == "after" else ""
            cv2.putText(im, f"{c} {lab}{lagtxt}  t={t}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(im, f"{c} {lab}{lagtxt}  t={t}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            row.append(im)
        tiles.append(np.concatenate(row, 1))
    frame = np.concatenate(tiles, 0)
    if a.stills and (a.still_ticks is None or t in a.still_ticks):
        frames_imgs["all"] = frames_imgs.get("all", {})
        frames_imgs["all"][t] = frame
    if enc is None:
        Hh, Ww = frame.shape[:2]
        enc = subprocess.Popen(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{Ww}x{Hh}",
                                "-r", str(30 // a.stride), "-i", "pipe:0", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", a.out],
                               stdin=subprocess.PIPE)
    enc.stdin.write(frame.tobytes())
enc.stdin.close(); enc.wait()
summ = {}
for k, v in stats.items():
    summ[k] = {"iou_med": float(np.median(v["iou"])) if v["iou"] else None, "n_iou": len(v["iou"]),
               "dz_med_mm": float(np.median(v["dz"]) * 1000) if v["dz"] else None, "n_dz": len(v["dz"])}
print(json.dumps(summ, indent=1))
if a.json:
    json.dump({"episode": f"{ep.task}/{ep.name}", "calib": a.calib, "metrics": summ}, open(a.json, "w"), indent=1)
if a.stills and "all" in frames_imgs:
    os.makedirs(a.stills, exist_ok=True)
    for t, im in frames_imgs["all"].items():
        cv2.imwrite(os.path.join(a.stills, f"calib_{ep.task}_{ep.name}_t{t:04d}.jpg"), im)
print("saved", a.out)
