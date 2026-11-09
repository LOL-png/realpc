"""Differentiable robot<->camera calibration ("calibration as an ML model").

The FK-posed YAM link meshes are rendered with pytorch3d (soft silhouettes + hard z-buffer) and compared with image robot
masks (SAM2, per arm) and the camera depth. Free parameters:
  * per camera c and arm a: the camera pose relative to that arm's base, T_cam<-base_a = exp(xi[c,a]) @ T0[c,a]
    (left and right separately, so a base offset of one arm shows up as a disagreement between the two)
  * per camera and episode: a piecewise-linear time lag L(t) (ticks; the image at tick t shows the robot at q(t - L(t)))
  * optional per-joint zero offsets (shared by both cameras)
Losses: 1 - soft IoU per arm (pixels of the other arm / of objects are ignored, they may occlude), a chamfer-style
distance-transform term for long-range gradients, robust arm-pixel depth, and priors.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .robot import URDFRobot


# ----------------------------------------------------------------------------------------------- se3 helpers (torch)
def hat(w):
    z = torch.zeros_like(w[..., 0])
    return torch.stack([torch.stack([z, -w[..., 2], w[..., 1]], -1),
                        torch.stack([w[..., 2], z, -w[..., 0]], -1),
                        torch.stack([-w[..., 1], w[..., 0], z], -1)], -2)


def so3_exp(w):
    th = w.norm(dim=-1, keepdim=True).clamp_min(1e-9)[..., None]
    K = hat(w / th[..., 0])
    I = torch.eye(3, dtype=w.dtype, device=w.device).expand(K.shape)
    return I + torch.sin(th) * K + (1 - torch.cos(th)) * K @ K


def se3_exp(xi):
    """xi [...,6] = (rot(3), trans(3)) -> [...,4,4] (rotation about the origin, then translation)."""
    R = so3_exp(xi[..., :3])
    T = torch.zeros(xi.shape[:-1] + (4, 4), dtype=xi.dtype, device=xi.device)
    T[..., :3, :3] = R
    T[..., :3, 3] = xi[..., 3:]
    T[..., 3, 3] = 1
    return T


def axis_rot(axis, q):
    """axis [3] tensor, q [...] -> [...,3,3]"""
    return so3_exp(q[..., None] * axis)


# ----------------------------------------------------------------------------------------------- arm model
def cluster_decimate(m, target_faces):
    """vertex-clustering decimation (robust on triangle soups): grow the cell until <= target faces."""
    import trimesh
    V, Fc = np.asarray(m.vertices), np.asarray(m.faces)
    cell = 0.001
    while True:
        key = np.floor(V / cell).astype(np.int64)
        _, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.ravel()
        cnt = np.bincount(inv)
        Vn = np.stack([np.bincount(inv, V[:, i]) for i in range(3)], 1) / cnt[:, None]
        Fn = inv[Fc]
        Fn = Fn[(Fn[:, 0] != Fn[:, 1]) & (Fn[:, 1] != Fn[:, 2]) & (Fn[:, 0] != Fn[:, 2])]
        Fn = np.unique(np.sort(Fn, 1), axis=0) if len(Fn) else Fn
        if len(Fn) <= target_faces or cell > 0.02:
            return trimesh.Trimesh(Vn, Fn, process=False)
        cell *= 1.3


class TorchArm:
    """Differentiable FK + decimated link meshes of one URDF arm (all links with a visual mesh)."""

    def __init__(self, urdf_path, arm_joints, finger_joints, faces_per_link=1500, device="cuda", seed=0):
        import trimesh
        self.robot = URDFRobot(urdf_path)
        self.device = device
        self.arm_joints = list(arm_joints)
        self.finger = finger_joints
        self.links = list(self.robot.visual.keys())
        V, Fc, vid = [], [], []
        off = 0
        for li, k in enumerate(self.links):
            fn, To, sc = self.robot.visual[k]
            m = trimesh.load(fn, process=False, force="mesh")
            m.merge_vertices()
            if len(m.faces) > faces_per_link:
                m = cluster_decimate(m, faces_per_link)
            v = np.asarray(m.vertices) * np.asarray(sc)
            v = v @ To[:3, :3].T + To[:3, 3]
            V.append(v); Fc.append(np.asarray(m.faces) + off); vid.append(np.full(len(v), li)); off += len(v)
        self.verts_local = torch.tensor(np.concatenate(V), dtype=torch.float32, device=device)
        self.faces = torch.tensor(np.concatenate(Fc), dtype=torch.int64, device=device)
        self.vlink = torch.tensor(np.concatenate(vid), dtype=torch.int64, device=device)
        self.jorder = []
        for j in self.robot.order:
            self.jorder.append(dict(name=j.name, type=j.type, parent=j.parent, child=j.child,
                                    origin=torch.tensor(j.origin, dtype=torch.float32, device=device),
                                    axis=torch.tensor(j.axis / np.linalg.norm(j.axis), dtype=torch.float32, device=device)))
        rng = np.random.default_rng(seed)
        self.sub = torch.tensor(rng.choice(len(self.verts_local), min(3000, len(self.verts_local)), replace=False), device=device)

    def link_tf(self, q, offsets=None):
        """q [B,7] (6 joints + normalised gripper) -> {link: [B,4,4]} in the arm base frame."""
        B = q.shape[0]
        qm = {}
        for i, j in enumerate(self.arm_joints):
            qm[j] = q[:, i] + (offsets[i] if offsets is not None else 0.0)
        g = q[:, len(self.arm_joints)].clamp(0, 1)
        for j, (c, o) in self.finger.items():
            qm[j] = c + g * (o - c)
        dt = q.dtype
        eye = torch.eye(4, device=self.device, dtype=dt).expand(B, 4, 4)
        T = {self.robot.root: eye}
        for j in self.jorder:
            M = torch.eye(4, device=self.device, dtype=dt).repeat(B, 1, 1)
            v = qm.get(j["name"], torch.zeros(B, device=self.device, dtype=dt))
            ax = j["axis"].to(dt)
            if j["type"] in ("revolute", "continuous"):
                M[:, :3, :3] = axis_rot(ax, v)
            elif j["type"] == "prismatic":
                M[:, :3, 3] = ax * v[:, None]
            T[j["child"]] = T[j["parent"]] @ j["origin"].to(dt) @ M
        return T

    def verts(self, q, offsets=None):
        """[B,V,3] mesh vertices in the arm base frame."""
        T = self.link_tf(q, offsets)
        Ts = torch.stack([T[k] for k in self.links], 1)             # [B,L,4,4]
        Tv = Ts[:, self.vlink]                                       # [B,V,4,4]
        return torch.einsum("bvij,vj->bvi", Tv[..., :3, :3], self.verts_local.to(Tv.dtype)) + Tv[..., :3, 3]


# ----------------------------------------------------------------------------------------------- renderer
class Renderer:
    """pytorch3d rasteriser for OpenCV pinhole cameras; inputs are vertices already in the OpenCV camera frame."""

    def __init__(self, K, H, W, device="cuda"):
        self.K = torch.tensor(np.asarray(K), dtype=torch.float32, device=device)
        self.H, self.W, self.device = H, W, device

    def _ndc(self, Vc):
        # pytorch3d NDC: +x left, +y up; the shorter image side spans [-1, 1]
        s = min(self.H, self.W) / 2.0
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        z = Vc[..., 2]
        u = fx * Vc[..., 0] / z.clamp_min(1e-4) + cx
        v = fy * Vc[..., 1] / z.clamp_min(1e-4) + cy
        x = -(u - self.W / 2.0) / s
        y = -(v - self.H / 2.0) / s
        return torch.stack([x, y, z], -1)

    def raster(self, Vc, faces, sigma=None, fpp=1, znear=0.03):
        """Vc [B,V,3] -> (soft or hard silhouette [B,H,W], zbuf [B,H,W] (0 = empty))."""
        from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes
        from pytorch3d.structures import Meshes
        P = self._ndc(Vc)
        # drop triangles with a vertex in front of the near plane (the top camera sees the arms very close)
        fz = Vc[:, faces, 2].amin(-1)                                # [B,F]
        outs_s, outs_z = [], []
        for b in range(Vc.shape[0]):
            fk = faces[fz[b] > znear]
            m = Meshes(verts=[P[b]], faces=[fk])
            blur = 0.0 if sigma is None else math.log(1.0 / 1e-4 - 1.0) * sigma
            pix, zbuf, bary, dists = rasterize_meshes(m, image_size=(self.H, self.W), blur_radius=blur,
                                                      faces_per_pixel=fpp, perspective_correct=False,
                                                      cull_backfaces=False, bin_size=None, max_faces_per_bin=100000)
            if sigma is None:
                s = (pix[..., 0] >= 0).float()
            else:
                prob = torch.sigmoid(-dists / sigma) * (pix >= 0).float()
                s = 1.0 - torch.prod(1.0 - prob, dim=-1)
            # depth along z of the first face; zbuf is interpolated (differentiable w.r.t. the vertices)
            z = torch.where(pix[..., 0] >= 0, zbuf[..., 0], torch.zeros_like(zbuf[..., 0]))
            outs_s.append(s[0]); outs_z.append(z[0])
        return torch.stack(outs_s), torch.stack(outs_z)

    def project(self, Vc):
        z = Vc[..., 2].clamp_min(1e-3)
        return torch.stack([self.K[0, 0] * Vc[..., 0] / z + self.K[0, 2], self.K[1, 1] * Vc[..., 1] / z + self.K[1, 2]], -1)


def soft_iou(p, m, valid):
    inter = (p * m * valid).sum((-1, -2))
    union = ((p + m - p * m) * valid).sum((-1, -2))
    return inter / union.clamp_min(1.0)


def interp_q(qseq, tt):
    """qseq [T,D] tensor, tt [B] fractional ticks -> [B,D] linear interpolation (clamped)."""
    T = qseq.shape[0]
    tt = tt.clamp(0, T - 1 - 1e-4)
    i0 = tt.floor().long()
    w = (tt - i0.float())[:, None]
    return qseq[i0] * (1 - w) + qseq[(i0 + 1).clamp(max=T - 1)] * w
