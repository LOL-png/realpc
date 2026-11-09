#!/usr/bin/env python
"""Overlay library meshes at given poses on the top + side images of one tick (filled + outline per object).

  python scripts/object_overlay.py --episode <ep> --lib <dir> --calib <json|recorder> --tick 0 --out <png>
         [--tracks <track_objects npz>]   (poses from tracks; default: the library's per-episode start poses)
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch
import trimesh

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from realpc.calib_rig import RigCalib  # noqa: E402
from realpc.diffcalib import Renderer  # noqa: E402
from realpc.io import Episode  # noqa: E402

COLS = [(0, 200, 255), (255, 120, 0), (0, 255, 120), (255, 0, 200), (200, 255, 0)]


def overlay(ep, cal, objs, poses, t, cams=("top", "side"), width=640):
    tiles = []
    for c in cams:
        cap = cv2.VideoCapture(ep.video(c)); cap.set(cv2.CAP_PROP_POS_FRAMES, t); ok, im = cap.read()
        H, W = im.shape[:2]
        s = width / W
        im = cv2.resize(im, (width, int(round(H * s))))
        K = ep.intrinsics(c).copy(); K[:2] *= s
        R = Renderer(K, im.shape[0], width)
        Tcw = torch.tensor(np.linalg.inv(cal.T_world_cam(c)), dtype=torch.float32, device="cuda")
        out = im.copy()
        for i, (name, (V, F)) in enumerate(objs.items()):
            if name not in poses or not np.isfinite(poses[name]).all():
                continue
            Tt = torch.tensor(poses[name], dtype=torch.float32, device="cuda")
            Vw = V @ Tt[:3, :3].T + Tt[:3, 3]
            sil, _ = R.raster((Vw @ Tcw[:3, :3].T + Tcw[:3, 3])[None], F)
            m = sil[0].cpu().numpy() > 0.5
            col = np.array(COLS[i % len(COLS)], np.uint8)
            out[m] = (0.45 * col + 0.55 * out[m]).astype(np.uint8)
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(out, cs, -1, COLS[i % len(COLS)], 2)
        cv2.putText(out, f"{ep.task}/{ep.name} {c} t={t}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(out, f"{ep.task}/{ep.name} {c} t={t}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        tiles.append(out)
    h = max(x.shape[0] for x in tiles)
    tiles = [cv2.copyMakeBorder(x, 0, h - x.shape[0], 0, 0, cv2.BORDER_CONSTANT) for x in tiles]
    return np.concatenate(tiles, 1)


def load_lib(lib):
    objs, metas = {}, {}
    for f in sorted(glob.glob(os.path.join(lib, "*.ply"))):
        meta = json.load(open(f[:-4] + ".json"))
        m = trimesh.load(f, process=False)
        objs[meta["name"]] = (torch.tensor(np.asarray(m.vertices), dtype=torch.float32, device="cuda"),
                              torch.tensor(np.asarray(m.faces), dtype=torch.int64, device="cuda"))
        metas[meta["name"]] = meta
    return objs, metas


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", nargs="+", required=True); ap.add_argument("--lib", required=True); ap.add_argument("--calib", required=True)
    ap.add_argument("--tick", type=int, default=0); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    objs, metas = load_lib(a.lib)
    rows = []
    for e in a.episodes:
        ep = Episode(e)
        cal = RigCalib.recorder(ep) if a.calib == "recorder" else RigCalib.load(a.calib)
        key = f"{ep.task}/{ep.name}"
        poses = {}
        for n, m in metas.items():
            for sc in m["scans"]:
                if sc["episode"] == key:
                    poses[n] = np.asarray(sc["T_world_obj_t0"], float)
        rows.append(overlay(ep, cal, objs, poses, a.tick))
    cv2.imwrite(a.out, np.concatenate(rows, 0))
    print("saved", a.out)


def poses_from_tracks(npz, t):
    d = np.load(npz, allow_pickle=True)
    k = int(np.searchsorted(d["ticks"], t))
    return {str(n): d["obj_T"][i, k].astype(float) for i, n in enumerate(d["obj_names"])}
