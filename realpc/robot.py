"""Robot model: URDF forward kinematics and fixed surface points per link.

Points are sampled once on each link's visual mesh (in the link frame) and moved by FK every step, so a robot point is
the same physical point in every frame (exact correspondence). Only revolute / prismatic / fixed joints are needed.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np


def rpy_to_mat(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def origin_tf(el):
    T = np.eye(4)
    if el is None:
        return T
    xyz = [float(v) for v in el.get("xyz", "0 0 0").split()]
    rpy = [float(v) for v in el.get("rpy", "0 0 0").split()]
    T[:3, :3] = rpy_to_mat(*rpy)
    T[:3, 3] = xyz
    return T


def axis_angle(axis, q):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * K @ K


@dataclass
class Joint:
    name: str
    type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float


class URDFRobot:
    """Minimal URDF FK. `fk(qmap)` -> {link: 4x4 pose in the robot base frame}."""

    def __init__(self, urdf_path: str):
        self.path = urdf_path
        root = ET.parse(urdf_path).getroot()
        self.links, self.visual = [], {}
        for ln in root.findall("link"):
            nm = ln.get("name")
            self.links.append(nm)
            vis = ln.find("visual")
            if vis is not None and vis.find("geometry/mesh") is not None:
                m = vis.find("geometry/mesh")
                sc = [float(v) for v in m.get("scale", "1 1 1").split()]
                self.visual[nm] = (os.path.join(os.path.dirname(urdf_path), m.get("filename")), origin_tf(vis.find("origin")), sc)
        self.joints = []
        for j in root.findall("joint"):
            lim = j.find("limit")
            self.joints.append(Joint(j.get("name"), j.get("type"), j.find("parent").get("link"), j.find("child").get("link"),
                                     origin_tf(j.find("origin")),
                                     np.array([float(v) for v in (j.find("axis").get("xyz") if j.find("axis") is not None else "1 0 0").split()]),
                                     float(lim.get("lower", 0)) if lim is not None else 0.0,
                                     float(lim.get("upper", 0)) if lim is not None else 0.0))
        children = {j.child for j in self.joints}
        self.root = [l for l in self.links if l not in children][0]
        # topological order
        order, frontier = [], [self.root]
        while frontier:
            p = frontier.pop(0)
            for j in self.joints:
                if j.parent == p:
                    order.append(j)
                    frontier.append(j.child)
        self.order = order

    def fk(self, qmap: dict) -> dict:
        T = {self.root: np.eye(4)}
        for j in self.order:
            q = float(qmap.get(j.name, 0.0))
            M = np.eye(4)
            if j.type in ("revolute", "continuous"):
                M[:3, :3] = axis_angle(j.axis, q)
            elif j.type == "prismatic":
                M[:3, 3] = np.asarray(j.axis) / np.linalg.norm(j.axis) * q
            T[j.child] = T[j.parent] @ j.origin @ M
        return T

    def sample_surface(self, n_per_link: dict | int, seed: int = 0) -> dict:
        """{link: [n,3] points in the link frame} sampled evenly on the visual mesh (area weighted)."""
        import trimesh
        out = {}
        for k, (fn, To, sc) in self.visual.items():
            n = n_per_link if isinstance(n_per_link, int) else n_per_link.get(k, 0)
            if n <= 0:
                continue
            m = trimesh.load(fn, process=False, force="mesh")
            pts, _ = trimesh.sample.sample_surface_even(m, n * 2, seed=seed)
            pts = np.asarray(pts)[:n] * np.asarray(sc)
            if len(pts) < n:                                  # sample_surface_even may return fewer
                extra, _ = trimesh.sample.sample_surface(m, n - len(pts), seed=seed)
                pts = np.concatenate([pts, np.asarray(extra) * np.asarray(sc)])
            out[k] = (pts @ To[:3, :3].T + To[:3, 3]).astype(np.float32)
        return out


class BimanualArms:
    """Two copies of one arm model. Arm 'left' defines the world frame; 'right' sits at T_world_rightbase.

    q: per arm a vector [j1..j6, gripper] with gripper normalised (1 = open, 0 = closed) as recorded.
    gripper_map: finger joint value = closed + g * (open - closed) for each finger joint.
    """

    def __init__(self, urdf_path, T_world_base: dict, arm_joints, finger_joints: dict, points_per_link: dict, seed=0):
        self.robot = URDFRobot(urdf_path)
        self.T_base = {k: np.asarray(v, float) for k, v in T_world_base.items()}
        self.arm_joints = list(arm_joints)
        self.finger = finger_joints                                          # {joint: [closed, open]}
        local = self.robot.sample_surface(points_per_link, seed=seed)
        self.link_names = [l for l in self.robot.links if l in local]
        # flat point table: arm, link, local coords
        P, link_id, arm_id = [], [], []
        self.bodies = []                                                     # (arm, link)
        for ai, arm in enumerate(self.T_base):
            for l in self.link_names:
                P.append(local[l]); link_id.append(np.full(len(local[l]), len(self.bodies))); arm_id.append(np.full(len(local[l]), ai))
                self.bodies.append((arm, l))
        self.local = np.concatenate(P).astype(np.float32)
        self._urdf_path, self._seed = urdf_path, seed
        self.body = np.concatenate(link_id)
        self.arm_of = np.concatenate(arm_id)

    def qmap(self, q):
        m = {j: float(v) for j, v in zip(self.arm_joints, q[: len(self.arm_joints)])}
        g = float(np.clip(q[len(self.arm_joints)], 0, 1))
        for j, (c, o) in self.finger.items():
            m[j] = c + g * (o - c)
        return m

    def link_poses(self, qs: dict) -> np.ndarray:
        """qs {arm: q} -> [n_bodies, 4, 4] world poses."""
        out = []
        for arm in self.T_base:
            T = self.robot.fk(self.qmap(qs[arm]))
            for l in self.link_names:
                out.append(self.T_base[arm] @ T[l])
        return np.stack(out)

    def points(self, qs: dict, poses=None) -> np.ndarray:
        Tb = self.link_poses(qs) if poses is None else poses
        R, t = Tb[self.body, :3, :3], Tb[self.body, :3, 3]
        return (np.einsum("nij,nj->ni", R, self.local) + t).astype(np.float32)

    def dense_local(self, n_per_link=1500):
        """denser surface samples (local coords, body ids) for removing robot points from measurements."""
        if not hasattr(self, "_dense_cache"):
            local = self.robot.sample_surface(n_per_link, seed=self._seed + 1)
            L, B = [], []
            for bi, (arm, l) in enumerate(self.bodies):
                L.append(local[l]); B.append(np.full(len(local[l]), bi))
            self._dense_cache = (np.concatenate(L).astype(np.float64), np.concatenate(B))
        return self._dense_cache
