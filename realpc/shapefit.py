"""Clean object models fitted by multi-view silhouette + depth (differentiable rendering, pytorch3d).

Models (all in the v1 library object frame, so per-episode start poses and object masks stay valid):
  * Primitive composite: parts = boxes / cylinders with learnable size and part pose (block = box; mug = cylinder +
    C-shaped handle of three boxes).
  * Smooth deformable surface: a heavily smoothed, re-meshed version of the fused v1 mesh whose vertices move along
    their normals (one offset each), with Laplacian smoothness (broom, dustpan, stand).
Data: frames (episode, camera, tick) with the object's mask, depth and the tracked v1 pose; each frame gets a small
learnable pose correction (the tracked poses carry cm-level noise). Losses: 1 - soft IoU (robot pixels and other objects
ignored: they may occlude), robust depth on mask pixels, priors.
"""
from __future__ import annotations

import numpy as np
import torch
import trimesh

from .diffcalib import Renderer, se3_exp, so3_exp, soft_iou

DEV = "cuda"


# ----------------------------------------------------------------------------------------------- templates
def box_template():
    m = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    return np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int64)


def cyl_template(n=40):
    m = trimesh.creation.cylinder(radius=1.0, height=1.0, sections=n)
    return np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int64)


class PrimitiveModel(torch.nn.Module):
    """parts: list of dict(type 'box'|'cyl', size (box: sx,sy,sz full; cyl: r,r,h), rot (axis-angle), pos)."""

    def __init__(self, parts, sym=None):
        super().__init__()
        self.types = [p["type"] for p in parts]
        self.smin = torch.tensor([p.get("min", [1e-3] * 3) for p in parts], dtype=torch.float32)
        self.smax = torch.tensor([p.get("max", [1.0] * 3) for p in parts], dtype=torch.float32)
        self.log_size = torch.nn.Parameter(torch.tensor(np.log([p["size"] for p in parts]), dtype=torch.float32))
        self.rot = torch.nn.Parameter(torch.tensor([p.get("rot", [0, 0, 0]) for p in parts], dtype=torch.float32))
        self.pos = torch.nn.Parameter(torch.tensor([p["pos"] for p in parts], dtype=torch.float32))
        self.tmpl = {"box": box_template(), "cyl": cyl_template()}
        F, off = [], 0
        for t in self.types:
            v, f = self.tmpl[t]
            F.append(f + off); off += len(v)
        self.faces = torch.tensor(np.concatenate(F), device=DEV)

    def verts(self):
        out = []
        size = torch.minimum(torch.maximum(self.log_size.exp(), self.smin.to(self.log_size.device)), self.smax.to(self.log_size.device))
        for i, t in enumerate(self.types):
            v = torch.tensor(self.tmpl[t][0], device=DEV)
            s = size[i] if t == "box" else torch.stack([size[i, 0], size[i, 0], size[i, 2]])
            R = so3_exp(self.rot[i])
            out.append((v * s) @ R.T + self.pos[i])
        return torch.cat(out)

    def mesh(self):
        """watertight union of the parts (exported)."""
        V = self.verts().detach().cpu().numpy()
        parts, off = [], 0
        for t in self.types:
            n = len(self.tmpl[t][0])
            parts.append(trimesh.Trimesh(V[off: off + n], self.tmpl[t][1], process=True)); off += n
        try:
            m = trimesh.boolean.union(parts, engine="manifold")
        except Exception:
            m = trimesh.util.concatenate(parts)
        return m


class FFDModel(torch.nn.Module):
    """smooth global deformation of a (smoothed) mesh: trilinear lattice of n^3 control points over its bounding box
    (cannot create spikes; fixes size / proportions)."""

    def __init__(self, mesh: trimesh.Trimesh, n=4):
        super().__init__()
        self.mesh0 = mesh
        V = np.asarray(mesh.vertices, np.float32)
        lo, hi = V.min(0) - 0.01, V.max(0) + 0.01
        self.lo = torch.tensor(lo, device=DEV); self.span = torch.tensor(hi - lo, device=DEV)
        self.u = (torch.tensor(V, device=DEV) - self.lo) / self.span                 # [V,3] in [0,1]
        self.n = n
        self.d = torch.nn.Parameter(torch.zeros(n, n, n, 3, device=DEV))
        self.base = torch.tensor(V, device=DEV)
        self.faces = torch.tensor(np.asarray(mesh.faces), dtype=torch.int64, device=DEV)

    def verts(self):
        # trilinear interpolation of the control-point displacements
        g = self.u * (self.n - 1)
        i0 = g.floor().clamp(0, self.n - 2).long(); f = g - i0
        out = torch.zeros_like(self.base)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                    out = out + w[:, None] * self.d[i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz]
        return self.base + out

    def smooth_loss(self):
        d = self.d
        return ((d[1:] - d[:-1]) ** 2).mean() + ((d[:, 1:] - d[:, :-1]) ** 2).mean() + ((d[:, :, 1:] - d[:, :, :-1]) ** 2).mean()

    def mesh(self):
        return trimesh.Trimesh(self.verts().detach().cpu().numpy(), self.mesh0.faces, process=False)


class DeformModel(torch.nn.Module):
    """smooth surface: base vertices + normal offsets; Laplacian smoothness via uniform graph Laplacian."""

    def __init__(self, mesh: trimesh.Trimesh):
        super().__init__()
        self.base = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32, device=DEV)
        self.nrm = torch.tensor(np.asarray(mesh.vertex_normals), dtype=torch.float32, device=DEV)
        self.faces = torch.tensor(np.asarray(mesh.faces), dtype=torch.int64, device=DEV)
        self.off = torch.nn.Parameter(torch.zeros(len(self.base), device=DEV))
        E = np.asarray(mesh.edges_unique)
        self.E = torch.tensor(E, device=DEV)
        self.mesh0 = mesh

    def verts(self):
        return self.base + self.off[:, None] * self.nrm

    def smooth_loss(self):
        V = self.verts()
        n = len(V)
        acc = torch.zeros_like(V).index_add_(0, self.E[:, 0], V[self.E[:, 1]]).index_add_(0, self.E[:, 1], V[self.E[:, 0]])
        deg = torch.zeros(n, device=DEV).index_add_(0, self.E[:, 0], torch.ones(len(self.E), device=DEV)).index_add_(
            0, self.E[:, 1], torch.ones(len(self.E), device=DEV))
        L = V - acc / deg.clamp(min=1)[:, None]
        return (L ** 2).sum(1).mean()

    def mesh(self):
        m = trimesh.Trimesh(self.verts().detach().cpu().numpy(), self.mesh0.faces, process=False)
        return m


def smooth_remesh(mesh, voxel=0.004, sigma=1.5):
    """voxelise (filled) -> gaussian -> marching cubes: a smooth, coarse watertight surface."""
    from scipy import ndimage
    from skimage import measure
    vg = mesh.voxelized(voxel).fill()
    occ = np.pad(vg.matrix, 3).astype(np.float32)
    vol = ndimage.gaussian_filter(occ, sigma)
    V, F, _, _ = measure.marching_cubes(vol, 0.5)
    V = (V - 3) * voxel
    # voxel index -> object frame
    V = trimesh.transform_points(V / voxel, vg.transform)
    m = trimesh.Trimesh(V, F[:, ::-1], process=True)
    trimesh.smoothing.filter_taubin(m, iterations=20)
    if m.volume < 0:
        m.invert()
    return m


# ----------------------------------------------------------------------------------------------- fitting
class Frame:
    def __init__(self, mask, valid, depth, K, T_cam_obj, H, W):
        self.mask = torch.tensor(mask, dtype=torch.float32, device=DEV)[None]
        self.valid = torch.tensor(valid, dtype=torch.float32, device=DEV)[None]
        self.depth = torch.tensor(depth, dtype=torch.float32, device=DEV)            # metres (0 = none), same size
        self.T = torch.tensor(T_cam_obj, dtype=torch.float32, device=DEV)
        self.R = Renderer(K, H, W)


def fit(model, frames, iters=600, lr=3e-3, batch=12, sigma=2e-4, depth_w=0.5, pose_prior=(0.01, np.radians(3)),
        smooth_w=0.0, log=print):
    model.to(DEV)
    n = len(frames)
    dpose = torch.nn.Parameter(torch.zeros(n, 6, device=DEV))
    opt = torch.optim.Adam([{"params": model.parameters(), "lr": lr}, {"params": [dpose], "lr": lr * 0.5}])
    rng = np.random.default_rng(0)
    hist = []
    for it in range(iters):
        idx = rng.choice(n, min(batch, n), replace=False)
        V = model.verts()
        loss = 0.0
        for i in idx:
            f = frames[i]
            Tc = f.T @ se3_exp(dpose[i])
            Vc = V @ Tc[:3, :3].T + Tc[:3, 3]
            sil, z = f.R.raster(Vc[None], model.faces, sigma=sigma, fpp=8)
            l_sil = 1 - soft_iou(sil, f.mask, f.valid)[0]
            _, zh = f.R.raster(Vc[None], model.faces)
            m = (f.mask[0] > 0) & (zh[0] > 0) & (f.depth > 0.1)
            l_d = torch.tensor(0.0, device=DEV)
            if m.sum() > 30:
                e = (zh[0][m] - f.depth[m]).abs()
                l_d = torch.where(e < 0.01, 0.5 * e ** 2 / 0.01, e - 0.005).mean() / 0.01
            loss = loss + l_sil + depth_w * l_d
        loss = loss / len(idx)
        pr = (dpose[idx, 3:] / pose_prior[0]).pow(2).sum(1).mean() + (dpose[idx, :3] / pose_prior[1]).pow(2).sum(1).mean()
        loss = loss + 0.01 * pr
        if smooth_w > 0:
            loss = loss + smooth_w * model.smooth_loss()
        opt.zero_grad(); loss.backward(); opt.step()
        hist.append(float(loss))
        if it % 100 == 0 or it == iters - 1:
            log(f"  it {it} loss {np.mean(hist[-50:]):.4f}")
    return dpose.detach()


@torch.no_grad()
def eval_iou(model, frames, dpose=None):
    V = model.verts()
    out = []
    for i, f in enumerate(frames):
        Tc = f.T @ (se3_exp(dpose[i]) if dpose is not None else torch.eye(4, device=DEV))
        sil, _ = f.R.raster((V @ Tc[:3, :3].T + Tc[:3, 3])[None], model.faces)
        r = sil[0] > 0.5; g = f.mask[0] > 0; v = f.valid[0] > 0
        out.append(float(((r & g & v).sum()) / max(1, ((r | g) & v).sum())))
    return np.array(out)
