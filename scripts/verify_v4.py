#!/usr/bin/env python
"""Verification video of a v4 converted episode: [top image + overlays | side image + overlays | 3D points].

Overlays: FK robot meshes (red left / magenta right outlines) and library meshes at their tracked poses (filled), on the
camera images; the 3D panel splats the converted points (fixed identity) from a fixed virtual view, coloured by group
(robot grey, objects by id, static by image colour, cloth white; held objects outlined in the title).

  python scripts/verify_v4.py --converted <npz> --episode <ep dir> --out <mp4> [--stride 2] [--stills_dir d --still_ticks ...]
"""
import argparse
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
from realpc.dataset import load  # noqa: E402
from realpc.diffcalib import Renderer, TorchArm  # noqa: E402
from realpc.io import Episode  # noqa: E402
from realpc.v4 import load_library  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--converted", required=True); ap.add_argument("--episode", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--stride", type=int, default=2); ap.add_argument("--calib", default=os.path.join(ROOT, "configs", "calib_yam_rig_v3.json"))
ap.add_argument("--assets", default=os.path.join(ROOT, "assets", "objects"))
ap.add_argument("--stills_dir", default=None); ap.add_argument("--still_ticks", type=int, nargs="*", default=[])
a = ap.parse_args()
d = load(a.converted)
ep = Episode(a.episode)
cal = RigCalib.load(a.calib)
cfg = load_config(os.path.join(ROOT, "configs", "yam_bimanual.json")); r = cfg["robot"]
tarm = TorchArm(os.path.join(cfg["_dir"], r["urdf"]), r["arm_joints"], r["finger_joints"])
objs = []
if os.path.isdir(os.path.join(a.assets, ep.task)):
    objs, _ = load_library(os.path.join(a.assets, ep.task), n_samples=200)
names = [str(x) for x in d.get("obj_names", [])]
OCOL = [(0, 200, 255), (255, 140, 0), (60, 255, 60), (255, 0, 200), (200, 255, 0)]
T_wb = {"left": np.eye(4), "right": cal.T_world_rb()}
size = {"top": (360, 640), "side": (480, 640)}
R = {}
for c in ("top", "side"):
    K = ep.intrinsics(c).copy(); W0 = 1280 if c == "top" else 640; K[:2] *= 640 / W0
    R[c] = Renderer(K, *size[c])
# fixed 3D view: from the front-left above the table, looking at the table centre
ctr = d["static_points"].mean(0)
eye = ctr + np.array([-0.9, 0.0, 0.75])
fwd = ctr - eye; fwd /= np.linalg.norm(fwd)
right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right); up = np.cross(right, fwd)
Rv = np.stack([right, -up, fwd]); Kv = np.array([[560, 0, 320], [0, 560, 240], [0, 0, 1.0]])
col = d["color"].astype(np.uint8).copy()
g = d["group"]
col[g <= 1] = (110, 110, 110)
for i in range(len(names)):
    col[(g == 3) & (d["obj_id"] == i)] = OCOL[i % len(OCOL)][::-1]
col[g == 4] = (235, 235, 235)
ticks = d["ticks"]
caps = {c: cv2.VideoCapture(ep.video(c)) for c in ("top", "side")}
for c in caps:
    caps[c].set(cv2.CAP_PROP_POS_FRAMES, int(ticks[0]))
enc = None
cur = int(ticks[0])
for k in range(0, len(ticks), a.stride):
    t = int(ticks[k])
    ims = {}
    for c in ("top", "side"):
        while cur <= t:
            ok, im = caps[c].read() if c == "top" else (None, None)
            break
    # simple seek per frame (robust; videos are short)
    for c in ("top", "side"):
        caps[c].set(cv2.CAP_PROP_POS_FRAMES, t); ok, im = caps[c].read()
        ims[c] = cv2.resize(im, (640, size[c][0]))
    tiles = []
    for c in ("top", "side"):
        im = ims[c]
        Tcw = torch.tensor(np.linalg.inv(cal.T_world_cam(c)), dtype=torch.float32, device="cuda")
        tc = t - (cal.lag(ep, c, t) if c == "side" else 0.0)       # side content time
        kc = int(np.clip(round(tc - ticks[0]), 0, len(ticks) - 1))
        for i, o in enumerate(objs):
            if o.name not in names:
                continue
            Tt = torch.tensor(d["obj_T"][names.index(o.name), kc], dtype=torch.float32, device="cuda")
            if not torch.isfinite(Tt).all():
                continue
            Vw = o.verts_t @ Tt[:3, :3].T + Tt[:3, 3]
            sil, _ = R[c].raster((Vw @ Tcw[:3, :3].T + Tcw[:3, 3])[None], o.faces_t)
            m = sil[0].cpu().numpy() > 0.5
            cc = np.array(OCOL[i % len(OCOL)], np.uint8)
            im[m] = (0.45 * cc + 0.55 * im[m]).astype(np.uint8)
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(im, cs, -1, OCOL[i % len(OCOL)], 2)
        for s, cc in (("left", (0, 0, 255)), ("right", (255, 0, 255))):
            q = (d["q_left"] if s == "left" else d["q_right"])[kc]
            V = tarm.verts(torch.tensor(q[None], dtype=torch.float32, device="cuda"))[0]
            Tb = torch.tensor(cal.T_cam_base(c, s), dtype=torch.float32, device="cuda")
            sil, _ = R[c].raster((V @ Tb[:3, :3].T + Tb[:3, 3])[None], tarm.faces)
            cs, _ = cv2.findContours((sil[0].cpu().numpy() > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(im, cs, -1, cc, 1)
        held = [n for i, n in enumerate(names) if d["obj_held"][i, kc] > 0]
        txt = f"{ep.task}/{ep.name} {c} t={t}" + (f"  held: {','.join(held)}" if held else "")
        cv2.putText(im, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3); cv2.putText(im, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        tiles.append(im)
    # 3D panel
    P = d["points"][k]
    ok = d["valid"][k]
    Pc = (P[ok] - eye) @ Rv.T
    z = Pc[:, 2]
    uv = (Pc @ Kv.T)[:, :2] / np.maximum(z, 1e-3)[:, None]
    cc = col[ok]
    pan = np.full((480, 640, 3), 30, np.uint8)
    inb = (z > 0.1) & (uv[:, 0] >= 0) & (uv[:, 0] < 639) & (uv[:, 1] >= 0) & (uv[:, 1] < 479)
    o_ = np.argsort(-z[inb])
    uu = uv[inb][o_].astype(int); c3 = cc[inb][o_]
    for dx in (0, 1):
        for dy in (0, 1):
            pan[np.clip(uu[:, 1] + dy, 0, 479), np.clip(uu[:, 0] + dx, 0, 639)] = c3
    cv2.putText(pan, f"converted points t={t} (N={int(ok.sum())})", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    top = cv2.copyMakeBorder(tiles[0], 0, 120, 0, 0, cv2.BORDER_CONSTANT)
    frame = np.concatenate([top, tiles[1], pan], 1)
    if a.stills_dir and t in a.still_ticks:
        os.makedirs(a.stills_dir, exist_ok=True)
        cv2.imwrite(os.path.join(a.stills_dir, f"verify_{ep.task}_{ep.name}_t{t:04d}.jpg"), frame)
    if enc is None:
        H, W = frame.shape[:2]
        enc = subprocess.Popen(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                                "-r", str(max(1, 22 // a.stride)), "-i", "pipe:0", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", a.out],
                               stdin=subprocess.PIPE)
    enc.stdin.write(frame.tobytes())
enc.stdin.close(); enc.wait()
print("saved", a.out)
