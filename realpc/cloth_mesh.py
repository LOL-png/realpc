"""Cloth as a tracked mesh (v1): fused top+side depth surface + causal ARAP / point-to-plane tracking.

Per frame (current frame only; causal):
  surface   top (colour-guided refined) + side depth -> pixels of the robot (FK mesh points) and of the table (plane +
            table colour) removed -> TSDF over a box around the cloth (voxel 5 mm, truncation 1.5 cm, top weight 1, side
            0.5) -> marching cubes -> faces kept only where a measured point is within 1 cm -> surface points + normals
  template  at the first frame: that surface, decimated by vertex clustering to ~2K vertices (largest connected part)
  tracking  per frame, 6 local/global iterations: E = w_d sum (n.(x_i - y))^2 (point-to-plane to the nearest surface
            point, within 3 cm) + w_a ARAP(x | template) + w_p |x_i - pinned target|^2 (vertices near closed fingertips
            follow the FK fingertips) + w_s |x - x_prev|^2 (small, keeps unobserved parts in place)
  points    fixed barycentric samples on the template triangles -> identity persists
v1 has no 2D-track term yet (next: projected vertices follow CoTracker3 tracks in both cameras).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes


def backproject(D, K, T_wc, stride=2):
    Ds = D[::stride, ::stride]
    v, u = np.nonzero(Ds > 0.25)
    z = Ds[v, u]; u = u * stride; v = v * stride
    P = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], 1) @ T_wc[:3, :3].T + T_wc[:3, 3]
    return P, u, v


def tsdf_surface(views, lo, hi, voxel=0.005, trunc=0.015, support=0.01):
    """views: list of (depth HxW with removed pixels = 0, K, T_wc, weight, P_measured world points).
    -> surface vertices (n,3), faces (m,3), vertex normals (n,3)."""
    shape = np.maximum(np.ceil((hi - lo) / voxel).astype(int), 2)
    g = np.stack(np.meshgrid(*[lo[i] + voxel * (np.arange(shape[i]) + 0.5) for i in range(3)], indexing="ij"), -1).reshape(-1, 3)
    num = np.zeros(len(g)); den = np.zeros(len(g))
    for D, K, T_wc, w, _ in views:
        Ti = np.linalg.inv(T_wc)
        pc = g @ Ti[:3, :3].T + Ti[:3, 3]
        z = pc[:, 2]
        uv = (pc @ K.T)[:, :2] / np.maximum(z, 1e-6)[:, None]
        H, W = D.shape
        ok = (z > 0.1) & (uv[:, 0] >= 0) & (uv[:, 0] < W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < H - 1)
        d = np.zeros(len(g)); d[ok] = D[uv[ok, 1].astype(int), uv[ok, 0].astype(int)]
        ok &= d > 0.25
        sdf = d - z
        ok &= sdf > -trunc * 3                     # behind the surface beyond 3x trunc: unknown
        val = np.clip(sdf / trunc, -1, 1)
        num[ok] += w * val[ok]; den[ok] += w
    T = np.where(den > 0, num / np.maximum(den, 1e-9), 1.0).reshape(shape)
    if T.min() >= 0 or T.max() <= 0:
        return np.zeros((0, 3)), np.zeros((0, 3), int), np.zeros((0, 3))
    V, F, N, _ = marching_cubes(T, 0.0, spacing=(voxel,) * 3)
    V = V + lo + voxel / 2
    meas = np.concatenate([v[4] for v in views])
    dv = cKDTree(meas).query(V, distance_upper_bound=support)[0]
    keepv = np.isfinite(dv)
    F = F[keepv[F].all(1)]
    used = np.unique(F)
    remap = -np.ones(len(V), int); remap[used] = np.arange(len(used))
    return V[used], remap[F], -N[used]                # normals point out of the solid (towards the cameras)


def decimate(V, F, cell):
    """vertex clustering on a grid of size `cell`; keeps the largest connected component."""
    key = np.floor(V / cell).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    n = inv.max() + 1
    Vn = np.zeros((n, 3)); np.add.at(Vn, inv, V); Vn /= np.bincount(inv, minlength=n)[:, None]
    Fn = inv[F]
    Fn = Fn[(Fn[:, 0] != Fn[:, 1]) & (Fn[:, 1] != Fn[:, 2]) & (Fn[:, 0] != Fn[:, 2])]
    Fn = np.unique(np.sort(Fn, 1), axis=0)
    from scipy.sparse.csgraph import connected_components
    E = np.concatenate([Fn[:, [0, 1]], Fn[:, [1, 2]], Fn[:, [0, 2]]])
    A = sp.coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n))
    nc, lab = connected_components(A, directed=False)
    big = np.bincount(lab).argmax()
    keep = lab == big
    remap = -np.ones(n, int); remap[keep] = np.arange(keep.sum())
    Fn = remap[Fn]; Fn = Fn[(Fn >= 0).all(1)]
    return Vn[keep], Fn


class ClothMesh:
    def __init__(self, V, F, n_samples=4000, seed=0, w_data=1.0, w_arap=4.0, w_pin=50.0, w_stay=0.05, corr_dist=0.03):
        self.V0, self.F = V.copy(), F
        self.X = V.copy()
        E = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [0, 2]]]), 1), axis=0)
        self.E = E
        n = len(V)
        self.nbrs = [[] for _ in range(n)]
        for a, b in E:
            self.nbrs[a].append(b); self.nbrs[b].append(a)
        # uniform-weight Laplacian
        I = np.concatenate([E[:, 0], E[:, 1], E[:, 0], E[:, 1]])
        J = np.concatenate([E[:, 1], E[:, 0], E[:, 0], E[:, 1]])
        W = np.concatenate([-np.ones(len(E)), -np.ones(len(E)), np.ones(len(E)), np.ones(len(E))])
        self.L = sp.coo_matrix((W, (I, J)), shape=(n, n)).tocsr()
        rng = np.random.default_rng(seed)
        area = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
        self.sf = rng.choice(len(F), n_samples, p=area / area.sum())
        r = rng.random((n_samples, 2)); m = r.sum(1) > 1; r[m] = 1 - r[m]
        self.sb = np.stack([1 - r.sum(1), r[:, 0], r[:, 1]], 1)
        self.w = dict(data=w_data, arap=w_arap, pin=w_pin, stay=w_stay)
        self.corr_dist = corr_dist

    def samples(self, X=None):
        X = self.X if X is None else X
        return np.einsum("nk,nkd->nd", self.sb, X[self.F[self.sf]])

    def edge_ratio(self):
        a = np.linalg.norm(self.X[self.E[:, 0]] - self.X[self.E[:, 1]], axis=1)
        b = np.linalg.norm(self.V0[self.E[:, 0]] - self.V0[self.E[:, 1]], axis=1)
        return a / np.maximum(b, 1e-6)

    def _rotations(self, X):
        R = np.zeros((len(X), 3, 3))
        for i, nb in enumerate(self.nbrs):
            if not nb:
                R[i] = np.eye(3); continue
            P = self.V0[nb] - self.V0[i]; Q = X[nb] - X[i]
            U, _, Vt = np.linalg.svd(P.T @ Q)
            d = np.sign(np.linalg.det(Vt.T @ U.T))
            R[i] = Vt.T @ np.diag([1, 1, d]) @ U.T
        return R

    def step(self, Y, NY, pins=None, iters=6):
        """Y (m,3) surface points, NY (m,3) normals, pins: (idx, targets) -> updates self.X."""
        n = len(self.X)
        Xp = self.X.copy()
        X = self.X.copy()
        if pins is not None and len(pins[0]):
            X[pins[0]] = pins[1]
        tree = cKDTree(Y) if len(Y) else None
        w = self.w
        for _ in range(iters):
            R = self._rotations(X)
            # ARAP rhs: b_i = sum_j 0.5 (R_i + R_j)(v_i - v_j)
            b_arap = np.zeros((n, 3))
            for a_, b_ in self.E:
                e = 0.5 * (R[a_] + R[b_]) @ (self.V0[a_] - self.V0[b_])
                b_arap[a_] += e; b_arap[b_] -= e
            # data (point-to-plane): per-vertex 3x3 block n n^T
            Ad = np.zeros((n, 3, 3)); bd = np.zeros((n, 3))
            if tree is not None:
                d, j = tree.query(X, distance_upper_bound=self.corr_dist)
                ok = np.isfinite(d)
                nn = NY[j[ok]]
                M = nn[:, :, None] * nn[:, None, :] + 0.1 * np.eye(3)          # + a little point-to-point
                Ad[ok] = w["data"] * M
                bd[ok] = w["data"] * np.einsum("nij,nj->ni", M, Y[j[ok]])
            Ap = np.zeros(n); bp = np.zeros((n, 3))
            if pins is not None and len(pins[0]):
                Ap[pins[0]] = w["pin"]; bp[pins[0]] = w["pin"] * pins[1]
            # assemble 3n system: (w_a L (x) I3 + blockdiag(Ad) + diag(Ap + w_s)) x = w_a b_arap + bd + bp + w_s Xp
            LL = sp.kron(self.L, sp.eye(3), format="csr") * w["arap"]
            rows = np.repeat(np.arange(n) * 3, 9) + np.tile(np.repeat(np.arange(3), 3), n)
            cols = np.repeat(np.arange(n) * 3, 9) + np.tile(np.tile(np.arange(3), 3), n)
            blocks = Ad + (Ap + w["stay"])[:, None, None] * np.eye(3)
            B = sp.coo_matrix((blocks.reshape(-1), (rows, cols)), shape=(3 * n, 3 * n)).tocsr()
            rhs = (w["arap"] * b_arap + bd + bp + w["stay"] * Xp).reshape(-1)
            X = spla.spsolve((LL + B).tocsc(), rhs).reshape(n, 3)
        self.X = X
        return X
