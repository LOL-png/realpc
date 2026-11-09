#!/usr/bin/env python
"""Side-by-side strip for chosen ticks: camera image | FK arm render at lag 0 | at the best lag (from lag_validate npz,
peak of the mean IoU curve in a +-15 tick window around the tick).

  python scripts/lag_strip.py --episode <ep> --lagval <npz> --cam side --ticks 186 198 ... --calib recorder --out <png>
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from realpc.calib_rig import RigCalib  # noqa: E402
from realpc.config import load_config  # noqa: E402
from realpc.diffcalib import Renderer, TorchArm, interp_q  # noqa: E402
from realpc.io import Episode  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--episode", required=True); ap.add_argument("--lagval", required=True); ap.add_argument("--cam", default="side")
ap.add_argument("--ticks", type=int, nargs="+", required=True); ap.add_argument("--calib", default="recorder"); ap.add_argument("--out", required=True)
ap.add_argument("--masks", default=None)
a = ap.parse_args()
ep = Episode(a.episode)
cal = RigCalib.recorder(ep) if a.calib == "recorder" else RigCalib.load(a.calib)
cfg = load_config(os.path.join(ROOT, "configs", "yam_bimanual.json")); r = cfg["robot"]
arm = TorchArm(os.path.join(cfg["_dir"], r["urdf"]), r["arm_joints"], r["finger_joints"])
d = np.load(a.lagval); I, L, T = d["iou"], d["lags"], d["ticks"]
M = np.load(os.path.join(a.masks, f"{a.cam}_masks.npz"))["masks"] if a.masks else None
cap = cv2.VideoCapture(ep.video(a.cam))
rows = []
COL = {"left": (0, 0, 255), "right": (255, 0, 255)}
for t in a.ticks:
    w = (np.abs(T - t) <= 15) & np.isfinite(I).all(1)
    best = int(L[np.argmax(np.nanmean(I[w], 0))]) if w.any() else 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, t); ok, im = cap.read()
    H, W = im.shape[:2]
    R = Renderer(ep.intrinsics(a.cam), H, W)
    tiles = [im.copy()]
    cv2.putText(tiles[0], f"{a.cam} image t={t}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    for lag in (0, best):
        o = im.copy()
        for s in ("left", "right"):
            q = interp_q(torch.tensor(ep.low[f"joint_pos_{s}"], dtype=torch.float32, device="cuda"), torch.tensor([float(t - lag)], device="cuda"))
            V = arm.verts(q)
            Tc = torch.tensor(cal.T_cam_base(a.cam, s), dtype=torch.float32, device="cuda")
            sil, _ = R.raster(V @ Tc[:3, :3].T + Tc[:3, 3], arm.faces)
            m = sil[0].cpu().numpy() > 0.5
            o[m] = (0.4 * np.array(COL[s]) + 0.6 * o[m]).astype(np.uint8)
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(o, cs, -1, COL[s], 2)
        if M is not None:
            for lid, col in ((2, (0, 255, 0)), (3, (255, 255, 0))):
                cs, _ = cv2.findContours((M[t] == lid).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(o, cs, -1, col, 1)
        iou = np.nanmean(I[w], 0)[list(L).index(lag)] if w.any() and lag in L else np.nan
        cv2.putText(o, f"FK at q(t{-lag:+d})  lag {lag:+d}  IoU {iou:.2f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        tiles.append(o)
    rows.append(np.concatenate(tiles, 1))
cv2.imwrite(a.out, np.concatenate(rows, 0))
print("saved", a.out)
