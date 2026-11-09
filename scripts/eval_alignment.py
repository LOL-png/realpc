#!/usr/bin/env python
"""Acceptance check of object alignment over a WHOLE episode: per-frame silhouette IoU of each tracked library mesh vs
the SAM2 object masks (Astra, evaluation only; the tracker never sees them), in the top and side cameras.

A frame counts for an object/camera when the mask or the render covers >= `--min_px` pixels (at 320 px width) outside
robot pixels. Pass = IoU >= `--thr`. Robot pixels are excluded (occluders); the side camera is compared at its content
time (t - lag).

  python scripts/eval_alignment.py --converted <npz> --episode <ep> --masks <dir> --out <json> [--every 2]
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from realpc.calib_rig import RigCalib  # noqa: E402
from realpc.diffcalib import Renderer, TorchArm  # noqa: E402
from realpc.config import load_config  # noqa: E402
from realpc.io import Episode  # noqa: E402
from realpc.v4 import load_library  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--converted", required=True); ap.add_argument("--episode", required=True); ap.add_argument("--masks", default=None)
ap.add_argument("--out", required=True); ap.add_argument("--every", type=int, default=2); ap.add_argument("--thr", type=float, default=0.5)
ap.add_argument("--min_px", type=int, default=60)
ap.add_argument("--seg_dir", default=None, help="use our own SAM2 masks (segment_objects.py) instead of --masks: self-consistency QA")
ap.add_argument("--calib", default=os.path.join(ROOT, "configs", "calib_yam_rig_v3.json"))
ap.add_argument("--assets", default=os.path.join(ROOT, "assets", "objects"))
a = ap.parse_args()
d = np.load(a.converted, allow_pickle=True)
ep = Episode(a.episode); cal = RigCalib.load(a.calib)
objs, _ = load_library(os.path.join(a.assets, ep.task), n_samples=100)
names = [str(x) for x in d["obj_names"]]
cfg = load_config(os.path.join(ROOT, "configs", "yam_bimanual.json")); rr = cfg["robot"]
tarm = TorchArm(os.path.join(cfg["_dir"], rr["urdf"]), rr["arm_joints"], rr["finger_joints"])
ticks = d["ticks"]
res = {}
per_frame = {}
for c in ("top", "side"):
    if a.seg_dir:
        z = np.load(os.path.join(a.seg_dir, ep.task, f"{ep.name}_{c}.npz"))
        M = z["labels"]; ln = ["background"] + [str(x) for x in z["names"]]
        ln = ln + [f"pad{i}" for i in range(len(ln), 254)] + ["robot_left", "robot_right"]
    else:
        z = np.load(os.path.join(a.masks, f"{c}_masks.npz")); M = z["masks"]; ln = [str(x) for x in z["label_names"]]
    H, W = M.shape[1:]; s = 320 / W; h = int(round(H * s))
    W_native = 1280 if c == "top" else 640
    K = ep.intrinsics(c).copy(); K[:2] *= 320 / W_native
    R = Renderer(K, h, 320)
    Tcw = torch.tensor(np.linalg.inv(cal.T_world_cam(c)), dtype=torch.float32, device="cuda")
    robot_ids = [i for i, x in enumerate(ln) if x.startswith("robot")]
    for o in objs:
        if o.name not in ln:
            continue
        pi = names.index(o.name); lid = ln.index(o.name)
        ious, tks = [], []
        for k in range(0, len(ticks), a.every):
            t = int(ticks[k])
            if t >= len(M):
                break
            tc = t - (cal.lag(ep, c, t) - cal.lag(ep, "top", t))           # content time relative to the top camera
            kc = int(np.clip(round(tc - ticks[0]), 0, len(ticks) - 1))
            T = d["obj_T"][pi, kc]
            if not np.isfinite(T).all():
                continue
            m = cv2.resize(M[t], (320, h), interpolation=cv2.INTER_NEAREST)
            # VISIBLE silhouette: z-test against the calibrated FK robot and the other tracked objects
            zs = []
            for oo in objs:
                To = d["obj_T"][names.index(oo.name), kc]
                if not np.isfinite(To).all():
                    zs.append(None); continue
                Tt = torch.tensor(To, dtype=torch.float32, device="cuda")
                Vw = oo.verts_t @ Tt[:3, :3].T + Tt[:3, 3]
                _, z = R.raster((Vw @ Tcw[:3, :3].T + Tcw[:3, 3])[None], oo.faces_t)
                zs.append(z[0].cpu().numpy())
            zr = np.full((h, 320), np.inf)
            for s_ in ("left", "right"):
                q = d["q_left" if s_ == "left" else "q_right"][kc]
                V = tarm.verts(torch.tensor(q[None], dtype=torch.float32, device="cuda"))[0]
                Tb = torch.tensor(cal.T_cam_base(c, s_), dtype=torch.float32, device="cuda")
                _, z = R.raster((V @ Tb[:3, :3].T + Tb[:3, 3])[None], tarm.faces)
                z = z[0].cpu().numpy(); zr = np.minimum(zr, np.where(z > 0, z, np.inf))
            me = zs[[oo.name for oo in objs].index(o.name)]
            rm = (me > 0) & (me < zr)
            for oo, zo in zip(objs, zs):
                if oo.name != o.name and zo is not None:
                    rm &= ~((zo > 0) & (zo < me))
            valid = ~np.isin(m, robot_ids)
            gt = (m == lid) & valid; rmv = rm & valid
            if gt.sum() < a.min_px and rmv.sum() < a.min_px:
                continue
            ious.append(float((gt & rmv).sum() / max(1, (gt | rmv).sum()))); tks.append(t)
        ious = np.array(ious)
        res[f"{o.name}/{c}"] = {"frames": int(len(ious)), "iou_median": float(np.median(ious)) if len(ious) else None,
                                "pass_frac": float((ious >= a.thr).mean()) if len(ious) else None}
        per_frame[f"{o.name}/{c}"] = {"ticks": tks, "iou": ious.tolist()}
out = {"episode": f"{ep.task}/{ep.name}", "converted": a.converted, "thr": a.thr, "summary": res, "per_frame": per_frame}
json.dump(out, open(a.out, "w"))
for k, v in res.items():
    print(f"{ep.task} {k}: frames {v['frames']}, IoU median {v['iou_median']}, pass(IoU>={a.thr}) {v['pass_frac']}")
