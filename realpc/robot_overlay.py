"""Project the FK-posed URDF meshes into the cameras: filled semi-transparent silhouette + outline, and accuracy numbers.

  outline->edge distance  for every outline pixel of the rendered arm, the distance (px) to the nearest Canny edge of
                          the photo (chamfer, capped at 30 px); median per arm
  depth residual          on silhouette pixels with a valid depth reading: measured depth - rendered depth (m); the
                          median |residual| and the share within 2 cm (the arm may be occluded or out of the sensor range)
`corr` = {arm: 4x4} world-frame correction of the camera for that arm (hand-eye refinement): points seen by this camera
are mapped by corr[arm] into the robot's world, so the arm is rendered with T_wc' = corr[arm] @ T_wc.
"""
from __future__ import annotations

import cv2
import numpy as np
import trimesh


class MeshProjector:
    def __init__(self, robot, max_faces=6000, seed=0):
        self.robot = robot
        self.meshes = {}
        rng = np.random.default_rng(seed)
        for k, (fn, To, sc) in robot.visual.items():
            m = trimesh.load(fn, process=False, force="mesh")
            F = np.asarray(m.faces)
            if len(F) > max_faces:
                F = F[rng.choice(len(F), max_faces, replace=False)]
            V = np.asarray(m.vertices) * np.asarray(sc)
            self.meshes[k] = (V @ To[:3, :3].T + To[:3, 3], F)

    def render(self, qmaps: dict, T_world_base: dict, T_wc, K, H, W, corr=None):
        """-> {arm: (mask bool HxW, depth HxW float (inf outside))}"""
        out = {}
        for arm, qm in qmaps.items():
            Tc = (corr or {}).get(arm, np.eye(4)) @ T_wc
            Ti = np.linalg.inv(Tc)
            fk = self.robot.fk(qm)
            mask = np.zeros((H, W), np.uint8)
            depth = np.full((H, W), np.inf, np.float32)
            for link, (V, F) in self.meshes.items():
                M = Ti @ T_world_base[arm] @ fk[link]
                Vc = V @ M[:3, :3].T + M[:3, 3]
                z = Vc[:, 2]
                uv = (Vc @ K.T)[:, :2] / np.maximum(z, 1e-3)[:, None]
                tri_ok = (z[F] > 0.05).all(1)
                for f in F[tri_ok]:
                    pts = uv[f].astype(np.int32)
                    if np.abs(pts).max() > 4 * max(H, W):
                        continue
                    cv2.fillConvexPoly(mask, pts, 1)
                # depth: splat vertices (dense enough for a residual statistic)
                ok = (z > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
                u, v = uv[ok, 0].astype(int), uv[ok, 1].astype(int)
                np.minimum.at(depth, (v, u), z[ok].astype(np.float32))
            # fill depth holes inside the mask by a min-filter (vertex splats are sparse)
            d = depth.copy(); d[~np.isfinite(d)] = 0
            dd = cv2.dilate(np.where(d > 0, 1.0 / np.maximum(d, 1e-3), 0).astype(np.float32), np.ones((9, 9), np.uint8))
            depth = np.where(mask > 0, np.where(dd > 0, 1.0 / np.maximum(dd, 1e-6), np.inf), np.inf).astype(np.float32)
            out[arm] = (mask > 0, depth)
        return out


def edge_distance(img_bgr, mask):
    edges = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0), 40, 120)
    dt = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 5)
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    H, W = mask.shape
    pts = np.concatenate([c[:, 0] for c in cs]) if cs else np.zeros((0, 2), int)
    pts = pts[(pts[:, 0] > 2) & (pts[:, 0] < W - 3) & (pts[:, 1] > 2) & (pts[:, 1] < H - 3)]   # ignore image border
    if len(pts) == 0:
        return np.nan, 0
    return float(np.median(np.minimum(dt[pts[:, 1], pts[:, 0]], 30))), int(len(pts))


def depth_residual(measured, rendered, mask):
    m = mask & (measured > 0.25) & np.isfinite(rendered)
    if m.sum() < 50:
        return np.nan, np.nan, int(m.sum())
    r = measured[m] - rendered[m]
    return float(np.median(np.abs(r))), float((np.abs(r) < 0.02).mean()), int(m.sum())


def draw(img, renders, colors={"left": (0, 0, 255), "right": (255, 0, 255)}, alpha=0.35):
    out = img.copy()
    for arm, (mask, _) in renders.items():
        c = np.array(colors[arm], np.uint8)
        out[mask] = (alpha * c + (1 - alpha) * out[mask]).astype(np.uint8)
        cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cs, -1, colors[arm], 2)
    return out
