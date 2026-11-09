"""Verification video / stills of converted tracks.

Layout (2 x 2, full resolution, 2240 x 1440):
  [ top camera RGB + tracked points (1280 x 720) | side camera RGB + tracked points (960 x 720) ]
  [ fixed 3D view: tracked (object colours) over the measured cloud (light grey) | per-object pose residual vs time ]
Objects that move during the episode get distinct solid colours (deformable = magenta); objects that never move are
drawn faint; the table is omitted. The robot (FK points, incl. attached hardware) is drawn on the arm, semi-transparent
(links orange, gripper yellow). No trails.
"""
from __future__ import annotations

import os
import subprocess

import cv2
import numpy as np

PALETTE = [(40, 40, 230), (230, 200, 40), (60, 220, 60), (0, 200, 255), (255, 90, 60), (200, 120, 255), (40, 140, 255),
           (180, 255, 120), (255, 255, 100), (120, 60, 200)]          # BGR, distinct
MAGENTA = (255, 0, 255)
ROBOT, GRIPPER, FAINT, MEASURED = (0, 140, 255), (0, 230, 255), (200, 200, 200), (205, 205, 205)


def project(P, T_world_cam, K):
    Ti = np.linalg.inv(T_world_cam)
    pc = P @ Ti[:3, :3].T + Ti[:3, 3]
    z = pc[:, 2]
    uv = (pc @ K.T)[:, :2] / np.maximum(z, 1e-6)[:, None]
    return uv, z


def splat(img, P, cols, T, K, r=1, alpha=1.0, zsort=True):
    """draw points (solid squares of half-size r) with optional transparency; far points first."""
    H, W = img.shape[:2]
    if len(P) == 0:
        return img
    uv, z = project(P, T, K)
    ok = np.isfinite(uv).all(1) & (z > 0.05) & (uv[:, 0] >= r) & (uv[:, 0] < W - r) & (uv[:, 1] >= r) & (uv[:, 1] < H - r)
    if not ok.any():
        return img
    u, v = uv[ok, 0].astype(int), uv[ok, 1].astype(int)
    c = np.asarray(cols, np.uint8)[ok] if np.ndim(cols) == 2 else np.tile(np.asarray(cols, np.uint8), (ok.sum(), 1))
    o = np.argsort(-z[ok]) if zsort else np.arange(ok.sum())
    layer = img.copy() if alpha < 1 else img
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            layer[v[o] + dy, u[o] + dx] = c[o]
    if alpha < 1:
        mask = np.zeros(img.shape[:2], bool)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                mask[v + dy, u + dx] = True
        img[mask] = (alpha * layer[mask] + (1 - alpha) * img[mask]).astype(np.uint8)
    return img


def robot_layer(P, cols, T, K, H, W, radius=0.008, rmax=14):
    """rasterise robot points as metric discs: colour image, mask and z-buffer (for occluding objects behind the arm)."""
    zb = np.full((H, W), np.inf, np.float32)
    col = np.zeros((H, W, 3), np.uint8)
    uv, z = project(P, T, K)
    ok = np.isfinite(uv).all(1) & (z > 0.05)
    uv, z, c = uv[ok], z[ok], np.asarray(cols)[ok]
    rad = np.clip((K[0, 0] * radius / z).astype(int), 1, rmax)
    for r in np.unique(rad):
        m = rad == r
        u, v, zz, cc = uv[m, 0].astype(int), uv[m, 1].astype(int), z[m], c[m]
        o = np.argsort(-zz)
        u, v, zz, cc = u[o], v[o], zz[o], cc[o]
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy > r * r:
                    continue
                uu, vv = u + dx, v + dy
                inb = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
                uu, vv, z2, c2 = uu[inb], vv[inb], zz[inb], cc[inb]
                closer = z2 < zb[vv, uu]
                zb[vv[closer], uu[closer]] = z2[closer]
                col[vv[closer], uu[closer]] = c2[closer]
    return col, np.isfinite(zb), zb


def visible(P, zb, T, K, margin=0.01):
    H, W = zb.shape
    uv, z = project(P, T, K)
    ok = np.isfinite(uv).all(1) & (z > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    vis = np.zeros(len(P), bool)
    i = np.flatnonzero(ok)
    vis[i] = z[i] < zb[uv[i, 1].astype(int), uv[i, 0].astype(int)] + margin
    return vis


def virtual_cam(center, dist=1.0, elev=38, azim=200, f=820, W=1280, H=720):
    e, a = np.radians(elev), np.radians(azim)
    eye = center + dist * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    zc = center - eye; zc /= np.linalg.norm(zc)
    xc = np.cross(zc, [0, 0, 1.0]); xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    T = np.eye(4); T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = xc, yc, zc, eye
    return T, np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])


class VideoReader:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path) if path and os.path.exists(path) else None
        self.pos, self.last = -1, None

    def get(self, tick):
        if self.cap is None:
            return None
        if tick != self.pos + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, tick)
        ok, img = self.cap.read()
        self.pos = tick
        if ok:
            self.last = img
        return img if ok else self.last


class FFWriter:
    """H.264 mp4 through ffmpeg (yuv420p)."""

    def __init__(self, path, W, H, fps):
        self.p = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps),
                                   "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "veryfast", path],
                                  stdin=subprocess.PIPE)

    def write(self, img):
        self.p.stdin.write(np.ascontiguousarray(img).tobytes())

    def close(self):
        self.p.stdin.close(); self.p.wait()


def label(img, text, org, scale=0.7):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


class TrackRenderer:
    def __init__(self, ep, tracks, title="", moved_thresh=0.02, min_pts=150, arms=None, side_map=None):
        self.ep, self.tr, self.title = ep, tracks, title
        nr = tracks.n_robot
        self.nb = int(tracks.body[:nr].max()) + 1
        grp = tracks.group.astype(int)
        self.robot = np.flatnonzero(grp <= 1)
        self.gripper_mask = grp[self.robot] == 0
        objp = np.flatnonzero(grp >= 3)
        oid = tracks.body[objp] - self.nb
        P = tracks.pos[objp]
        alive_any = np.isfinite(P[..., 0])
        first = np.argmax(alive_any, 1)
        disp = np.nanmax(np.linalg.norm(P - P[np.arange(len(P)), first][:, None], axis=-1), axis=1)
        moved_ids = []
        for o in np.unique(oid):
            m = oid == o
            if (np.nanmedian(np.nan_to_num(disp[m])) > moved_thresh and m.sum() >= min_pts) or (grp[objp][m] == 4).any():
                moved_ids.append((m.sum(), o))
        moved_ids = [o for _, o in sorted(moved_ids, reverse=True)]
        self.obj_color, k = {}, 0
        for o in moved_ids:
            if (grp[objp][oid == o] == 4).any():
                self.obj_color[o] = MAGENTA
            else:
                self.obj_color[o] = PALETTE[k % len(PALETTE)]; k += 1
        self.moved_ids, self.objp, self.oid = moved_ids, objp, oid
        self.cols = np.array([self.obj_color.get(o, FAINT) for o in oid], np.uint8).reshape(-1, 3)
        self.is_moved = np.isin(oid, moved_ids)
        sel = objp[self.is_moved] if self.is_moved.any() else objp
        c = np.nanmedian(tracks.pos[sel].reshape(-1, 3), axis=0)
        self.Tv, self.Kv = virtual_cam(np.nan_to_num(c))
        self.vt, self.vs = VideoReader(ep.video("top")), VideoReader(ep.video("side"))
        self.arms = arms
        self.side_map = side_map                     # top tick -> side frame (realpc.sync); None = same index
        if arms is not None:
            from .robot_overlay import MeshProjector
            self.proj = MeshProjector(arms.robot, max_faces=3000)
        self.plot_bg = self._plot_background()

    def _plot_background(self, W=960, H=720):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        tr = self.tr
        fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
        t = np.arange(tr.pos.shape[1])
        for o in self.moved_ids:
            if o < len(tr.obj_res):
                r = tr.obj_res[o, :, 0] * 1000
                c = np.array(self.obj_color[o][::-1]) / 255
                ax.plot(t, r, color=c, lw=1.6, label=f"obj {o}")
                held = tr.held[self.objp[self.oid == o]].any(0)
                if held.any():
                    ax.fill_between(t, 0, 1.5, where=held, color=c, alpha=0.6, step="mid")
        ax.set_ylim(0, 30); ax.set_xlim(0, len(t) - 1)
        ax.set_xlabel("step"); ax.set_ylabel("pose residual (mm)")
        ax.set_title("per-object residual (median measured -> object); bar = held", fontsize=11)
        ax.grid(alpha=0.3)
        if self.moved_ids:
            ax.legend(loc="upper right", fontsize=9)
        fig.tight_layout()
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3][..., ::-1].copy()
        bb = ax.get_window_extent()
        self._ax_px = (bb.x0, bb.x1)
        self._ax_py = (H - bb.y1, H - bb.y0)
        plt.close(fig)
        return cv2.resize(img, (W, H))

    def camera_panel(self, k, cam, size):
        tr, ep = self.tr, self.ep
        tick = int(tr.ticks[k])
        P = tr.pos[:, k]
        rob = P[self.robot]
        rcol = np.where(self.gripper_mask[:, None], np.array(GRIPPER, np.uint8), np.array(ROBOT, np.uint8)).astype(np.uint8)
        op = P[self.objp]
        alive = np.isfinite(op).all(1)
        ft = int(self.side_map[min(tick, len(self.side_map) - 1)]) if (cam == "side" and self.side_map is not None) else tick
        img = (self.vt if cam == "top" else self.vs).get(ft)
        K = ep.intrinsics(cam)
        img = np.zeros((720, 1280, 3), np.uint8) if img is None else img.copy()
        Tc = ep.low[f"cam_pose_{cam}"][tick]
        H_, W_ = img.shape[:2]
        if self.arms is not None:                    # FK mesh silhouette (occludes objects behind the arm)
            rd = self.proj.render({arm: self.arms.qmap(ep.low[f"joint_pos_{arm}"][tick]) for arm in self.arms.T_base},
                                  self.arms.T_base, Tc, K, H_, W_)
            zb = np.full((H_, W_), np.inf, np.float32)
            for m_, d_ in rd.values():
                zb = np.minimum(zb, np.where(m_, np.where(np.isfinite(d_), d_, 0.0), np.inf).astype(np.float32))
        else:
            rd = None
            rc, rmask, zb = robot_layer(rob, rcol, Tc, K, H_, W_, radius=0.004, rmax=6)
        vis = alive.copy()
        vis[alive] = visible(op[alive], zb, Tc, K)
        st = vis & ~self.is_moved
        splat(img, op[st], self.cols[st], Tc, K, r=0, alpha=0.3)
        mv = vis & self.is_moved
        splat(img, op[mv], self.cols[mv], Tc, K, r=1)
        if rd is not None:
            from .robot_overlay import draw as draw_mesh
            img = draw_mesh(img, rd, colors={"left": ROBOT, "right": ROBOT}, alpha=0.3)
        else:
            splat(img, rob, rcol, Tc, K, r=1, alpha=0.5)
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA if img.shape[1] > size[0] else cv2.INTER_LINEAR)
        return img

    def view3d(self, k, extra=None):
        tr, ep = self.tr, self.ep
        P = tr.pos[:, k]
        rob = P[self.robot]
        rcol = np.where(self.gripper_mask[:, None], np.array(GRIPPER, np.uint8), np.array(ROBOT, np.uint8)).astype(np.uint8)
        op = P[self.objp]
        alive = np.isfinite(op).all(1)
        v = np.full((720, 1280, 3), 255, np.uint8)
        if extra is None:
            f = ep.frame(int(tr.frame_index[k]))
            splat(v, f.points[::2], MEASURED, self.Tv, self.Kv, r=0, zsort=False)
            st = alive & ~self.is_moved
            splat(v, op[st], (150, 150, 150), self.Tv, self.Kv, r=0)
            mv = alive & self.is_moved
            splat(v, op[mv], self.cols[mv], self.Tv, self.Kv, r=1)
            for o in self.moved_ids:
                m = alive & (self.oid == o)
                if m.sum() > 10:
                    uv, z = project(np.nanmedian(op[m], 0)[None], self.Tv, self.Kv)
                    if np.isfinite(uv).all() and 0 < uv[0, 0] < 1200 and 20 < uv[0, 1] < 700:
                        cv2.putText(v, f"obj {o}", (int(uv[0, 0]) + 14, int(uv[0, 1])), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                    self.obj_color[o], 2, cv2.LINE_AA)
        else:
            Pe, Ce = extra
            splat(v, Pe, Ce, self.Tv, self.Kv, r=1)
        rc, rmask, _ = robot_layer(rob, rcol, self.Tv, self.Kv, 720, 1280, radius=0.003, rmax=3)
        v[rmask] = (0.5 * rc[rmask] + 0.5 * v[rmask]).astype(np.uint8)
        return v

    def frame(self, k):
        tr = self.tr
        top = self.camera_panel(k, "top", (1280, 720)); label(top, "top camera", (14, 32))
        side = self.camera_panel(k, "side", (960, 720)); label(side, "side camera", (14, 32))
        v = self.view3d(k)
        cv2.putText(v, "3D: tracked (colour) over measured cloud (grey)", (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2,
                    cv2.LINE_AA)
        plot = self.plot_bg.copy()
        T = tr.pos.shape[1]
        x0, x1 = self._ax_px
        x = int(x0 + (x1 - x0) * k / max(1, T - 1))
        cv2.line(plot, (x, int(self._ax_py[0])), (x, int(self._ax_py[1])), (0, 0, 0), 1)
        out = np.concatenate([np.concatenate([top, side], 1), np.concatenate([v, plot], 1)], 0)
        label(out, f"{self.title}  step {k}/{T - 1}", (14, 706), 0.75)
        return out


def render_tracks(ep, tracks, out_path, title="", fps=15, stride=1, arms=None, side_map=None, **kw):
    R = TrackRenderer(ep, tracks, title, arms=arms, side_map=side_map)
    wr = None
    for k in range(0, tracks.pos.shape[1], stride):
        fr = R.frame(k)
        if wr is None:
            wr = FFWriter(out_path, fr.shape[1], fr.shape[0], fps)
        wr.write(fr)
    wr.close()


def render_still(ep, tracks, out_png, k, title=""):
    cv2.imwrite(out_png, TrackRenderer(ep, tracks, title).frame(k))


def harness_cloud(depth_m, rgb, K, T_wc, table_n, table_d, stride=2):
    """refined depth image -> world points coloured by height above the table (turbo, 0..25 cm)."""
    D = depth_m[::stride, ::stride]
    v, u = np.nonzero((D > 0.25) & (D < 1.6))
    z = D[v, u]; u = u * stride; v = v * stride
    P = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], 1) @ T_wc[:3, :3].T + T_wc[:3, 3]
    h = P @ np.asarray(table_n) + table_d
    keep = h > -0.02
    c = cv2.applyColorMap(np.clip(h[keep] / 0.25 * 255, 0, 255).astype(np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0]
    return P[keep], c


def compare_still(ep, tracks, harness_depth_m, k, out_png, title_a="v3 causal", title_b="harness offline (visible layer)",
                  cam="top", arms=None):
    """2 x 2: [v3 on top photo | harness cloud on top photo] / [v3 3D | harness 3D] (same virtual camera)."""
    R = TrackRenderer(ep, tracks, arms=arms)
    tick = int(tracks.ticks[k])
    K = ep.intrinsics("top"); T_wc = ep.low["cam_pose_top"][tick]
    a = R.camera_panel(k, cam, (1280, 720)); label(a, f"{title_a} ({cam} camera): objects (colour), robot = FK mesh", (14, 32))
    rgb = R.vt.get(tick)
    Pb, Cb = harness_cloud(harness_depth_m, rgb, K, T_wc, *ep.table_plane())
    b = (R.vt if cam == "top" else R.vs).get(tick).copy()
    splat(b, Pb[::2], Cb[::2], ep.low[f"cam_pose_{cam}"][tick], ep.intrinsics(cam), r=1, zsort=True)
    b = cv2.resize(b, (1280, 720))
    label(b, f"{title_b}: refined depth, height colour", (14, 32))
    va = R.view3d(k); cv2.putText(va, f"{title_a} (3D, over measured grey)", (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)
    vb = R.view3d(k, extra=(Pb, Cb)); cv2.putText(vb, f"{title_b} (3D, no correspondence)", (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)
    out = np.concatenate([np.concatenate([a, b], 1), np.concatenate([va, vb], 1)], 0)
    label(out, f"{ep.task}/{ep.name}  step {k}  tick {tick}", (14, 706), 0.75)
    cv2.imwrite(out_png, out)
