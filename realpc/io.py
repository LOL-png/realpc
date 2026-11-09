"""Episode reader for the recorder's layout (one folder per episode):

  fused/<tick>.npz   points (N,3) f32, colors (N,3) u8, t (s), cam_poses (C,4,4), q_left (7), q_right (7)
                     (fused cloud of all RGB-D cameras in the LEFT arm base frame, x forward, z up)
  low_dim.npz        per tick: t_wall, joint_pos_<arm> (7), cam_pose_<role> ...
  calib.json         T_lb_rb (right base in the left base frame), table_n / table_d (n.x + d = 0), T_lb_<cam>
  meta.json          intrinsics, camera roles
  top.mp4            optional, for overlays

Only `Frame` objects reach the tracker, so a live system feeds the same structure (see README, streaming).
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

import numpy as np


@dataclass
class Frame:
    tick: int
    t: float
    points: np.ndarray          # (N,3) float32, world (= left arm base) frame
    colors: np.ndarray          # (N,3) uint8
    q: dict                     # {"left": (7,), "right": (7,)}  joints + normalised gripper (1 = open)
    views: list | None = None   # optional depth views [(depth_m HxW, K 3x3, T_world_cam 4x4)] for carving


class Episode:
    def __init__(self, path: str):
        self.path = path
        self.files = sorted(glob.glob(os.path.join(path, "fused", "*.npz")))
        if not self.files and not os.path.exists(os.path.join(path, "low_dim.npz")):
            raise FileNotFoundError(f"no fused/*.npz or low_dim.npz in {path}")
        self.seq = [int(os.path.basename(f)[:-4]) for f in self.files]       # fusion counter, NOT the video tick
        self.calib = json.load(open(os.path.join(path, "calib.json")))
        self.meta = json.load(open(os.path.join(path, "meta.json"))) if os.path.exists(os.path.join(path, "meta.json")) else {}
        lp = os.path.join(path, "low_dim.npz")
        self.low = dict(np.load(lp)) if os.path.exists(lp) else None
        if not self.files:                            # video-only download: one "frame" per video tick
            self.times = np.asarray(self.low["t_cap"], float)
            self.seq = list(range(len(self.times)))
            self.ticks = list(range(len(self.times)))
            self.name = os.path.basename(os.path.normpath(path))
            self.task = os.path.basename(os.path.dirname(os.path.normpath(path)))
            return
        self.times = np.array([float(np.load(f)["t"]) for f in self.files])
        # video / low_dim tick of each fused cloud: the tick whose top-camera capture time equals the cloud's time
        if self.low is not None and "t_cap" in self.low:
            self.ticks = [int(np.argmin(np.abs(self.low["t_cap"] - t))) for t in self.times]
        else:
            self.ticks = list(range(len(self.files)))
        self.name = os.path.basename(os.path.normpath(path))
        self.task = os.path.basename(os.path.dirname(os.path.normpath(path)))

    def __len__(self):
        return len(self.files)

    def frame(self, i: int) -> Frame:
        d = np.load(self.files[i])
        return Frame(self.ticks[i], float(d["t"]), d["points"].astype(np.float32), d["colors"],
                     {"left": d["q_left"].astype(np.float64), "right": d["q_right"].astype(np.float64)})

    def cam_pose(self, i: int, cam: str = "top") -> np.ndarray:
        roles = self.meta.get("cameras", ["top", "side", "left_wrist", "right_wrist"])
        return np.load(self.files[i])["cam_poses"][roles.index(cam)]

    def intrinsics(self, cam: str = "top") -> np.ndarray:
        return np.asarray(self.meta["intrinsics"][cam], float)

    def T_world_base(self) -> dict:
        return {"left": np.eye(4), "right": np.asarray(self.calib["T_lb_rb"], float)}

    def table_plane(self):
        return np.asarray(self.calib["table_n"], float), float(self.calib["table_d"])

    def video(self, cam: str = "top"):
        p = os.path.join(self.path, f"{cam}.mp4")
        return p if os.path.exists(p) else None


def decode_depth_video(mkv_path: str, cache_npy: str) -> np.ndarray:
    """16-bit FFV1 depth video -> uint16 memmap [T,H,W] cached at cache_npy (decoded once with ffmpeg)."""
    import subprocess
    if not os.path.exists(cache_npy):
        info = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                               "-of", "csv=p=0", mkv_path], capture_output=True, text=True).stdout.strip().split(",")
        W, H = int(info[0]), int(info[1])
        os.makedirs(os.path.dirname(os.path.abspath(cache_npy)), exist_ok=True)
        proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", mkv_path, "-f", "rawvideo", "-pix_fmt", "gray16le", "-"],
                                stdout=subprocess.PIPE)
        frames = []
        while True:
            buf = proc.stdout.read(W * H * 2)
            if len(buf) < W * H * 2:
                break
            frames.append(np.frombuffer(buf, np.uint16).reshape(H, W))
        proc.wait()
        np.save(cache_npy + ".part.npy", np.stack(frames))
        os.replace(cache_npy + ".part.npy", cache_npy)
    return np.load(cache_npy, mmap_mode="r")


def backproject(depth_m: np.ndarray, K: np.ndarray, T_world_cam: np.ndarray, stride: int = 1, zmin=0.1, zmax=3.0):
    """depth image (m) -> world points [n,3] and pixel coords [n,2] (pixels with zmin < z < zmax)."""
    D = depth_m[::stride, ::stride]
    v, u = np.nonzero((D > zmin) & (D < zmax))
    z = D[v, u]
    u = u * stride; v = v * stride
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    P = np.stack([x, y, z], 1) @ T_world_cam[:3, :3].T + T_world_cam[:3, 3]
    return P.astype(np.float32), np.stack([u, v], 1)


def attach_views(ep, frame, sources):
    """add the depth views of `sources` (DepthSource list) at this frame's tick."""
    frame.views = [(s.depth(frame.tick), s.K, s.pose(frame.tick)) for s in sources]
    return frame


class DepthSource:
    """Per-tick depth images of one camera from `<cam>_depth.mkv` (offline replay of what a live camera delivers)."""

    def __init__(self, ep: "Episode", cam: str, cache_dir: str):
        self.ep, self.cam = ep, cam
        self.D = decode_depth_video(os.path.join(ep.path, f"{cam}_depth.mkv"),
                                    os.path.join(cache_dir, f"{ep.task}_{ep.name}_{cam}_depth.npy"))
        self.scale = float(ep.meta.get("depth_scale", {}).get(cam, 0.001))
        self.K = ep.intrinsics(cam)

    def depth(self, tick: int) -> np.ndarray:
        return self.D[tick].astype(np.float32) * self.scale

    def pose(self, tick: int) -> np.ndarray:
        return np.asarray(self.ep.low[f"cam_pose_{self.cam}"][tick], float)


def read_depth_frames(mkv_path: str, indices) -> dict:
    """decode the 16-bit depth video once, keeping only `indices` (no cache on disk) -> {tick: uint16 HxW}."""
    import subprocess
    want = set(int(i) for i in indices)
    info = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                           "-of", "csv=p=0", mkv_path], capture_output=True, text=True).stdout.strip().split(",")
    W, H = int(info[0]), int(info[1])
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", mkv_path, "-f", "rawvideo", "-pix_fmt", "gray16le", "-"],
                            stdout=subprocess.PIPE)
    out, t = {}, 0
    last = max(want) if want else -1
    while t <= last:
        buf = proc.stdout.read(W * H * 2)
        if len(buf) < W * H * 2:
            break
        if t in want:
            out[t] = np.frombuffer(buf, np.uint16).reshape(H, W).copy()
        t += 1
    proc.kill(); proc.wait()
    return out


class DepthStream:
    """sequential 16-bit depth reader (ffmpeg pipe), the offline stand-in for a live depth camera: next() -> (tick, uint16)."""

    def __init__(self, mkv_path: str, start: int = 0):
        import subprocess
        info = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                               "-of", "csv=p=0", mkv_path], capture_output=True, text=True).stdout.strip().split(",")
        self.W, self.H = int(info[0]), int(info[1])
        self.proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", mkv_path, "-f", "rawvideo", "-pix_fmt", "gray16le", "-"],
                                     stdout=subprocess.PIPE)
        self.t = -1
        while self.t < start - 1:
            self.next()

    def next(self):
        buf = self.proc.stdout.read(self.W * self.H * 2)
        if len(buf) < self.W * self.H * 2:
            return None
        self.t += 1
        return self.t, np.frombuffer(buf, np.uint16).reshape(self.H, self.W)

    def close(self):
        self.proc.kill(); self.proc.wait()
