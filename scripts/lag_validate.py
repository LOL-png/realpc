#!/usr/bin/env python
"""Validate camera lags: silhouette IoU of the rendered FK arms vs SAM robot masks as a function of the lag.

For ticks t (every --every) and lags L in [--lmin, --lmax]: IoU(render(q(t - L)), mask(t)) per arm (pixels of objects
and the other arm ignored). Writes iou [n_ticks, n_lags] (mean over the visible arms) + the fitted lag at each tick.

  python scripts/lag_validate.py --episode <ep> --masks <dir> --calib <json> --cam side --out <npz>
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
ap.add_argument("--episode", required=True); ap.add_argument("--masks", required=True); ap.add_argument("--calib", required=True)
ap.add_argument("--cam", default="side"); ap.add_argument("--out", required=True)
ap.add_argument("--every", type=int, default=6); ap.add_argument("--lmin", type=int, default=-30); ap.add_argument("--lmax", type=int, default=70)
ap.add_argument("--lstep", type=int, default=2)
a = ap.parse_args()
ep = Episode(a.episode)
cal = RigCalib.load(a.calib) if a.calib != "recorder" else RigCalib.recorder(ep)
cfg = load_config(os.path.join(ROOT, "configs", "yam_bimanual.json")); r = cfg["robot"]
arm = TorchArm(os.path.join(cfg["_dir"], r["urdf"]), r["arm_joints"], r["finger_joints"])
M = np.load(os.path.join(a.masks, f"{a.cam}_masks.npz"))["masks"]
H, W = M.shape[1:]
hh = int(round(H * 320 / W))
K = ep.intrinsics(a.cam).copy(); K[:2] *= 320 / W
R = Renderer(K, hh, 320)
lags = np.arange(a.lmin, a.lmax + 1, a.lstep)
ticks = np.arange(abs(a.lmin) if a.lmin < 0 else 0, len(M), a.every)
Q = {s: torch.tensor(ep.low[f"joint_pos_{s}"], dtype=torch.float32, device="cuda") for s in ("left", "right")}
out = np.full((len(ticks), len(lags)), np.nan, np.float32)
with torch.no_grad():
    for i, t in enumerate(ticks):
        m = cv2.resize(M[t], (320, hh), interpolation=cv2.INTER_NEAREST)
        acc = np.zeros(len(lags)); n = 0
        for s, lid, oid in (("left", 2, 3), ("right", 3, 2)):
            gt = m == lid
            if gt.sum() < 150:
                continue
            valid = ~((m >= 4) | (m == oid))
            tt = torch.tensor(t - lags, dtype=torch.float32, device="cuda")
            V = arm.verts(interp_q(Q[s], tt))
            Tc = torch.tensor(cal.T_cam_base(a.cam, s), dtype=torch.float32, device="cuda")
            Vc = V @ Tc[:3, :3].T + Tc[:3, 3]
            sil, _ = R.raster(Vc, arm.faces)
            sil = sil.cpu().numpy() > 0.5
            inter = (sil & gt & valid).sum((1, 2)); uni = ((sil | gt) & valid).sum((1, 2))
            acc += inter / np.maximum(uni, 1); n += 1
        if n:
            out[i] = acc / n
best = np.where(np.isfinite(out).any(1), lags[np.nanargmax(np.nan_to_num(out, nan=-1), 1)], np.nan)
np.savez_compressed(a.out, iou=out, lags=lags, ticks=ticks, best=best, calib=a.calib, cam=a.cam)
fl = np.isfinite(out).any(1)
print(f"{ep.task}/{ep.name} {a.cam}: frames {fl.sum()}, best-lag median {np.nanmedian(best):.1f}, p10/p90 "
      f"{np.nanpercentile(best,10):.0f}/{np.nanpercentile(best,90):.0f}; mean IoU curve peak at {lags[np.nanargmax(np.nanmean(out[fl],0))]} "
      f"(IoU {np.nanmax(np.nanmean(out[fl],0)):.3f} vs lag0 {np.nanmean(out[fl], 0)[list(lags).index(0)]:.3f})")
