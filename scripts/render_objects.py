#!/usr/bin/env python
"""Render stills of library meshes (4 views each, vertex colours, Phong shading) into one PNG per task.

  python scripts/render_objects.py --lib assets/objects/<task> --out <png>
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np
import torch
import trimesh

ap = argparse.ArgumentParser()
ap.add_argument("--lib", required=True); ap.add_argument("--out", required=True); ap.add_argument("--size", type=int, default=320)
a = ap.parse_args()

from pytorch3d.renderer import (FoVPerspectiveCameras, MeshRasterizer, MeshRenderer, PointLights, RasterizationSettings,  # noqa: E402
                                SoftPhongShader, TexturesVertex, look_at_view_transform)
from pytorch3d.structures import Meshes  # noqa: E402

rows = []
for f in sorted(glob.glob(os.path.join(a.lib, "*.ply"))):
    m = trimesh.load(f, process=False)
    meta = json.load(open(f[:-4] + ".json"))
    V = torch.tensor(np.asarray(m.vertices) - m.bounds.mean(0), dtype=torch.float32, device="cuda")
    Fc = torch.tensor(np.asarray(m.faces), dtype=torch.int64, device="cuda")
    col = np.asarray(m.visual.vertex_colors)[:, :3] / 255.0 if hasattr(m.visual, "vertex_colors") else np.full((len(V), 3), 0.7)
    tex = TexturesVertex(torch.tensor(col, dtype=torch.float32, device="cuda")[None])
    mesh = Meshes([V], [Fc], tex)
    r = float(np.linalg.norm(m.extents)) * 1.3
    tiles = []
    for elev, azim in ((30, 30), (30, 150), (75, 0), (-35, 90)):
        el, az = np.radians(elev), np.radians(azim)
        eye = (r * np.cos(el) * np.cos(az), r * np.cos(el) * np.sin(az), r * np.sin(el))
        R, T = look_at_view_transform(eye=(eye,), at=((0, 0, 0),), up=((0, 0, 1),))
        cams = FoVPerspectiveCameras(R=R, T=T, fov=45, znear=0.005, zfar=10.0, device="cuda")
        rd = MeshRenderer(MeshRasterizer(cams, RasterizationSettings(image_size=a.size, faces_per_pixel=1)),
                          SoftPhongShader(device="cuda", cameras=cams, lights=PointLights(location=[list(eye)], ambient_color=((0.5, 0.5, 0.5),), device="cuda")))
        img = (rd(mesh)[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)[:, :, ::-1].copy()
        tiles.append(img)
    row = np.concatenate(tiles, 1)
    txt = f"{meta['task']}/{meta['name']}  {np.round(np.array(meta['extents_m']) * 100, 1)} cm  watertight={meta['watertight']}  scans={len(meta['scans'])}"
    cv2.putText(row, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    rows.append(row)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
cv2.imwrite(a.out, np.concatenate(rows, 0))
print("saved", a.out)
