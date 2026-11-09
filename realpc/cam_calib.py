"""Camera calibration from the robot by 2D reprojection (no depth): align the FK-posed URDF mesh silhouette to the image.

For each camera, ONE camera pose in the world (= left arm base) is refined; both arms are used, the right arm through the
fixed base relation T_lb_rb. Per round:
  1. render each arm's silhouette at the current pose; take its outline pixels (away from the image border) and lift them
     to 3D with the rendered depth -> world points X on the arm's occluding contour;
  2. robust Levenberg-Marquardt over the 6-DoF camera correction: residual = distance transform of the photo's Canny
     edges at the projection of X (bilinear, capped), soft-L1 loss;
then re-render (the contour moves with the pose) and repeat. Evaluated by the outline->edge distance on held-out frames.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot


def edge_dt(img_bgr, cap=20.0):
    e = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0), 40, 120)
    return np.minimum(cv2.distanceTransform((e == 0).astype(np.uint8), cv2.DIST_L2, 5), cap).astype(np.float32)


def bilinear(img, u, v):
    H, W = img.shape
    u = np.clip(u, 0, W - 1.001); v = np.clip(v, 0, H - 1.001)
    u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
    du, dv = u - u0, v - v0
    return (img[v0, u0] * (1 - du) * (1 - dv) + img[v0, u0 + 1] * du * (1 - dv) + img[v0 + 1, u0] * (1 - du) * dv
            + img[v0 + 1, u0 + 1] * du * dv)


def contour_points(mask, depth, K, T_wc, border=4, step=2):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return np.zeros((0, 3))
    pts = np.concatenate([c[:, 0] for c in cs])[::step]
    H, W = mask.shape
    pts = pts[(pts[:, 0] > border) & (pts[:, 0] < W - border) & (pts[:, 1] > border) & (pts[:, 1] < H - border)]
    z = depth[pts[:, 1], pts[:, 0]]
    ok = np.isfinite(z) & (z > 0.05)
    pts, z = pts[ok], z[ok]
    x = (pts[:, 0] - K[0, 2]) / K[0, 0] * z
    y = (pts[:, 1] - K[1, 2]) / K[1, 1] * z
    return np.stack([x, y, z], 1) @ T_wc[:3, :3].T + T_wc[:3, 3]


def se3(x):
    T = np.eye(4); T[:3, :3] = Rot.from_rotvec(x[:3]).as_matrix(); T[:3, 3] = x[3:]
    return T


def calibrate_camera(samples, proj, arms, K, T_wc0, rounds=4, f_scale=3.0, prior_rad=np.radians(2.0), prior_m=0.02):
    """samples: list of (img_bgr, {arm: qmap}). Returns refined T_wc (camera pose in the world) and a log.
    The correction is applied in the camera frame: T_wc = T_wc0 @ se3(x)."""
    dts = [edge_dt(img) for img, _ in samples]
    x = np.zeros(6)
    log = []
    for rd in range(rounds):
        T = T_wc0 @ se3(x)
        Xs = []
        for (img, qm), dt in zip(samples, dts):
            H, W = img.shape[:2]
            r = proj.render(qm, arms.T_base, T, K, H, W)
            X = [contour_points(m, d, K, T) for m, d in r.values() if m.sum() > 300]
            Xs.append(np.concatenate(X) if X else np.zeros((0, 3)))

        def res(v):
            Ti = np.linalg.inv(T_wc0 @ se3(v))
            out = []
            for X, dt in zip(Xs, dts):
                if not len(X):
                    continue
                pc = X @ Ti[:3, :3].T + Ti[:3, 3]
                uv = (pc @ K.T)[:, :2] / np.maximum(pc[:, 2:], 1e-3)
                out.append(bilinear(dt, uv[:, 0], uv[:, 1]))
            return np.concatenate(out)

        def res_prior(v):                     # soft prior: the recorder calibration is cm-accurate
            r = res(v)
            w = np.sqrt(len(r)) * f_scale
            return np.concatenate([r, w * np.r_[v[:3] / prior_rad, v[3:] / prior_m]])
        r0 = res(x)
        sol = least_squares(res_prior, x, loss="soft_l1", f_scale=f_scale, x_scale=np.array([0.01] * 3 + [0.01] * 3), max_nfev=60)
        x = sol.x
        r1 = res(x)
        log.append({"round": rd, "n_contour_pts": int(len(r0)), "median_px_before": float(np.median(r0)),
                    "median_px_after": float(np.median(r1)), "rot_deg": float(np.degrees(np.linalg.norm(x[:3]))),
                    "trans_mm": float(np.linalg.norm(x[3:]) * 1000)})
    return T_wc0 @ se3(x), log
