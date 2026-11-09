"""Object mesh library: one watertight mesh per rigid object, fused from masked depth of many episodes and cameras.

Pipeline (scripts/build_object_lib.py):
  1. scene cloud of an episode at a static tick: top + side depth -> world (calibrated cameras), table region above the
     table plane, robot (FK mesh samples) removed
  2. reference episode: per-object points from the image masks (SAM2 labels)
  3. other episodes: the reference partial scan is registered into the scene cloud (colour-gated FPFH RANSAC +
     coloured ICP); scene points near the registered reference become that episode's partial scan (canonical frame)
  4. fusion of all partial scans + a table-contact closure of the unseen underside -> voxel occupancy -> closing +
     hole fill -> marching cubes -> Taubin smoothing: always watertight
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np
from scipy.spatial import cKDTree



def o3d():
    import open3d
    return open3d


def depth_points(D_m, rgb, K, T_wc, stride=2, zmin=0.12, zmax=2.0, labels=None):
    """depth (m) + colour (BGR) -> world points, colours (RGB float 0-1), pixel labels (if given), pixel uv."""
    Ds = D_m[::stride, ::stride]
    v, u = np.nonzero((Ds > zmin) & (Ds < zmax))
    z = Ds[v, u]
    u = u * stride; v = v * stride
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    P = np.stack([x, y, z], 1) @ T_wc[:3, :3].T + T_wc[:3, 3]
    C = rgb[v, u][:, ::-1].astype(np.float32) / 255.0
    L = labels[v, u] if labels is not None else None
    return P, C, L, np.stack([u, v], 1)


def table_crop(P, region, n, d, above=0.004, below=0.0):
    T = np.asarray(region["T_lb_table"], float)
    hs = np.asarray(region["half_size"], float)
    Ti = np.linalg.inv(T)
    Q = P @ Ti[:3, :3].T + Ti[:3, 3]
    h = P @ n + d
    return (np.abs(Q[:, 0]) < hs[0] + 0.05) & (np.abs(Q[:, 1]) < hs[1] + 0.05) & (h > above) & (h < 0.45)


def robot_free(P, robot_pts, r=0.025):
    if len(robot_pts) == 0:
        return np.ones(len(P), bool)
    d, _ = cKDTree(robot_pts).query(P, distance_upper_bound=r)
    return ~np.isfinite(d)


def to_pcd(P, C=None):
    O = o3d()
    pc = O.geometry.PointCloud(O.utility.Vector3dVector(np.asarray(P, np.float64)))
    if C is not None:
        pc.colors = O.utility.Vector3dVector(np.asarray(C, np.float64))
    return pc


def fpfh(pc, voxel):
    O = o3d()
    pc.estimate_normals(O.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.5, max_nn=30))
    return O.pipelines.registration.compute_fpfh_feature(pc, O.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100))


def register(src_P, src_C, dst_P, dst_C, voxel=0.005, init=None, seed=0):
    """reference partial scan -> scene: global FPFH RANSAC (unless init) + coloured ICP. -> T, fitness, rmse"""
    O = o3d()
    reg = O.pipelines.registration
    s = to_pcd(src_P, src_C).voxel_down_sample(voxel)
    t = to_pcd(dst_P, dst_C).voxel_down_sample(voxel)
    if len(s.points) < 30 or len(t.points) < 30:
        return np.eye(4), 0.0, 1.0
    if init is None:
        fs, ft = fpfh(s, voxel), fpfh(t, voxel)
        O.utility.random.seed(seed)
        best = None
        for k in range(3):
            r = reg.registration_ransac_based_on_feature_matching(
                s, t, fs, ft, True, voxel * 2.0, reg.TransformationEstimationPointToPoint(False), 3,
                [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9), reg.CorrespondenceCheckerBasedOnDistance(voxel * 2.0)],
                reg.RANSACConvergenceCriteria(200000, 0.999))
            if best is None or r.fitness > best.fitness:
                best = r
        init = best.transformation
    for p in (s, t):
        p.estimate_normals(O.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.5, max_nn=30))
    T = np.asarray(init)
    for rad in (voxel * 3, voxel * 1.5, voxel):
        try:
            r = reg.registration_colored_icp(s, t, rad, T, reg.TransformationEstimationForColoredICP(lambda_geometric=0.9),
                                             reg.ICPConvergenceCriteria(1e-6, 1e-6, 60))
        except RuntimeError:
            r = reg.registration_icp(s, t, rad, T, reg.TransformationEstimationPointToPlane(), reg.ICPConvergenceCriteria(max_iteration=60))
        T = np.asarray(r.transformation)
    ev = reg.evaluate_registration(s, t, voxel * 1.5, T)
    return T, float(ev.fitness), float(ev.inlier_rmse)


def colour_model(C_obj, C_other, k=15):
    """kNN colour classifier in a perceptual space: p(object | colour)."""
    lab = lambda C: cv2.cvtColor((np.clip(C, 0, 1)[None] * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)[0].astype(np.float32)
    rng = np.random.default_rng(0)
    a = lab(C_obj[rng.choice(len(C_obj), min(4000, len(C_obj)), replace=False)])
    b = lab(C_other[rng.choice(len(C_other), min(8000, len(C_other)), replace=False)]) if len(C_other) else np.zeros((0, 3), np.float32)
    X = np.concatenate([a, b]); y = np.r_[np.ones(len(a)), np.zeros(len(b))]
    tr = cKDTree(X)

    def prob(C):
        _, j = tr.query(lab(C), k=k)
        return y[j].mean(1)
    return prob


def watertight_mesh(P, N=None, voxel=0.0025, close_iter=2, table=None, smooth=10):
    """points (object frame) [+ table plane (n,d) in that frame for the underside closure] -> watertight trimesh."""
    import trimesh
    from scipy import ndimage
    from skimage import measure
    P = np.asarray(P, float)
    pts = [P]
    if table is not None:                       # unseen underside: fill the footprint of near-table points at the table
        n, d = table
        h = P @ n + d
        low = P[h < 0.015]
        if len(low) > 20:
            # 2D grid in the table plane
            e1 = np.cross(n, [1, 0, 0]); e1 = e1 if np.linalg.norm(e1) > 0.1 else np.cross(n, [0, 1, 0]); e1 /= np.linalg.norm(e1)
            e2 = np.cross(n, e1)
            uv = np.stack([low @ e1, low @ e2], 1)
            g = np.floor(uv / voxel).astype(int)
            g0 = g.min(0)
            occ = np.zeros(tuple(g.max(0) - g0 + 1), bool)
            occ[tuple((g - g0).T)] = True
            occ = ndimage.binary_closing(occ, iterations=2)
            occ = ndimage.binary_fill_holes(occ)
            ii = np.argwhere(occ) + g0
            uv2 = (ii + 0.5) * voxel
            base = uv2[:, :1] * e1 + uv2[:, 1:] * e2
            base = base - (base @ n + d)[:, None] * n         # project onto the table plane
            pts.append(base)
    Q = np.concatenate(pts)
    lo = Q.min(0) - 4 * voxel
    idx = np.floor((Q - lo) / voxel).astype(int)
    shape = idx.max(0) + 5
    occ = np.zeros(shape, bool)
    occ[tuple(idx.T)] = True
    occ = ndimage.binary_dilation(occ, iterations=1)
    occ = ndimage.binary_closing(occ, iterations=close_iter)
    occ = ndimage.binary_fill_holes(occ)
    lab_, nl = ndimage.label(occ)
    if nl > 1:
        sizes = ndimage.sum(occ, lab_, range(1, nl + 1))
        occ = lab_ == (1 + int(np.argmax(sizes)))
    occ = ndimage.binary_erosion(occ, iterations=1)       # undo the dilation (keeps the shell thickness ~1 voxel)
    vol = ndimage.gaussian_filter(occ.astype(np.float32), 0.8)
    vol = np.pad(vol, 1)
    V, F, _, _ = measure.marching_cubes(vol, 0.5)
    V = (V - 1) * voxel + lo
    m = trimesh.Trimesh(V, F[:, ::-1], process=True)
    if smooth:
        trimesh.smoothing.filter_taubin(m, iterations=smooth)
    parts = m.split(only_watertight=False)
    if len(parts) > 1:
        m = max(parts, key=lambda x: x.area)
    if m.volume < 0:
        m.invert()
    return m


def save_object(mesh, meta, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    mesh.export(os.path.join(out_dir, f"{name}.ply"))
    json.dump(meta, open(os.path.join(out_dir, f"{name}.json"), "w"), indent=1)


def carve(views, lo, hi, voxel=0.003, tol=0.008, plane=None, min_free=2, free_frac=0.3):
    """Space carving on a voxel grid [lo, hi): a voxel is FREE if some depth view sees past it (measured depth > voxel
    depth + tol at its pixel). views: list of (depth_m HxW, K, T_frame_cam). plane (n, d): voxels below it are free.
    -> occupancy bool grid (occupied-or-unseen), grid origin lo, voxel; `seen` counts views that observed each voxel."""
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    shape = np.maximum(np.ceil((hi - lo) / voxel).astype(int), 1)
    ii = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), -1).reshape(-1, 3)
    X = lo + (ii + 0.5) * voxel
    fv = np.zeros(len(X), np.int32); obs = np.zeros(len(X), np.int32)
    seen = np.zeros(len(X), np.int16)
    for D, K, Twc in views:
        Ti = np.linalg.inv(Twc)
        Pc = X @ Ti[:3, :3].T + Ti[:3, 3]
        z = Pc[:, 2]
        H, W = D.shape
        u = np.round(K[0, 0] * Pc[:, 0] / np.maximum(z, 1e-6) + K[0, 2]).astype(int)
        v = np.round(K[1, 1] * Pc[:, 1] / np.maximum(z, 1e-6) + K[1, 2]).astype(int)
        inb = (z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        d = np.zeros(len(X)); d[inb] = D[v[inb], u[inb]]
        valid = inb & (d > 0.1)
        fv += valid & (d > z + tol)
        obs += valid & (d > z - tol)
        seen += (valid & (np.abs(d - z) < tol)).astype(np.int16)
    free = (fv >= min_free) & (fv >= free_frac * np.maximum(obs, 1))
    if plane is not None:
        n, dd = plane
        free |= (X @ n + dd) < -0.001
    occ = (~free).reshape(shape)
    return occ, lo, voxel, seen.reshape(shape)


def occupancy_mesh(occ, lo, voxel, seed_pts=None, smooth=4, sigma=0.5):
    """largest (or seed-containing) component of an occupancy grid -> watertight trimesh."""
    import trimesh
    from scipy import ndimage
    from skimage import measure
    lab, nl = ndimage.label(occ)
    if nl == 0:
        raise ValueError("empty occupancy")
    if seed_pts is not None and len(seed_pts):
        idx = np.floor((np.asarray(seed_pts) - lo) / voxel).astype(int)
        ok = np.all((idx >= 0) & (idx < np.array(occ.shape)), 1)
        ids = lab[tuple(idx[ok].T)]
        ids = ids[ids > 0]
        keep = np.bincount(ids, minlength=nl + 1)
        sizes = ndimage.sum(occ, lab, range(nl + 1))
        # components that contain >= 5% of the observed object points
        sel = np.flatnonzero(keep > max(5, 0.05 * keep.sum()))
        occ = np.isin(lab, sel) if len(sel) else lab == (1 + int(np.argmax(sizes[1:])))
    else:
        sizes = ndimage.sum(occ, lab, range(1, nl + 1))
        occ = lab == (1 + int(np.argmax(sizes)))
    occ = ndimage.binary_fill_holes(occ)
    vol = np.pad(ndimage.gaussian_filter(occ.astype(np.float32), sigma), 1)
    V, F, _, _ = measure.marching_cubes(vol, 0.5)
    V = (V - 1 + 0.5) * voxel + lo
    m = trimesh.Trimesh(V, F[:, ::-1], process=True)
    if smooth:
        trimesh.smoothing.filter_taubin(m, iterations=smooth)
    parts = m.split(only_watertight=False)
    if len(parts) > 1:
        m = max(parts, key=lambda x: x.volume if x.is_watertight else 0)
    if m.volume < 0:
        m.invert()
    return m


def footprint_fill(occ, lo, voxel, P, plane, max_gap=0.03):
    """objects resting on the table: fill each (table-plane) column between the table and the lowest observed object
    point when that point is within `max_gap` of the table (unseen underside of a resting object)."""
    n, d = plane
    shape = np.array(occ.shape)
    ii = np.floor((P - lo) / voxel).astype(int)
    ok = np.all((ii >= 0) & (ii < shape), 1)
    P, ii = P[ok], ii[ok]
    h = P @ n + d
    ax = int(np.argmax(np.abs(n)))                    # grid axis closest to the table normal
    other = [k for k in range(3) if k != ax]
    key = ii[:, other[0]] * shape[other[1]] + ii[:, other[1]]
    order = np.lexsort((h, key))
    key_s, first = np.unique(key[order], return_index=True)
    out = occ.copy()
    # 2D holes of the resting footprint (e.g. a dark tray with no depth) are closed by a table-level floor slab
    from scipy import ndimage
    fp = np.zeros((shape[other[0]], shape[other[1]]), bool)
    low = np.array([h[order[fi]] <= max_gap for fi in first])
    fp[key_s[low] // shape[other[1]], key_s[low] % shape[other[1]]] = True
    filled = ndimage.binary_fill_holes(ndimage.binary_closing(fp, iterations=2)) & ~fp
    for a_, b_ in np.argwhere(filled):
        for z in range(shape[ax]):
            sl = [0, 0, 0]; sl[other[0]] = a_; sl[other[1]] = b_; sl[ax] = z
            X = lo + (np.array(sl) + 0.5) * voxel
            hz = X @ n + d
            if -0.0005 <= hz <= 0.004:
                out[tuple(sl)] = True
    for kk, fi in zip(key_s, first):
        j = order[fi]
        if h[j] > max_gap:
            continue
        a_, b_ = kk // shape[other[1]], kk % shape[other[1]]
        top = ii[j, ax]
        sl = [0, 0, 0]; sl[other[0]] = a_; sl[other[1]] = b_
        for z in range(0, shape[ax]):
            sl[ax] = z
            X = lo + (np.array(sl) + 0.5) * voxel
            hz = X @ n + d
            if -0.0005 <= hz and (z <= top if n[ax] > 0 else z >= top):
                out[tuple(sl)] = True
    return out


def object_occupancy(free_or_occ, lo, voxel, P, near=0.015):
    """carved occupancy restricted to the neighbourhood of the observed object points (no unbounded shadow volumes);
    voxels holding observed points are always occupied."""
    from scipy import ndimage
    occ = free_or_occ
    shape = np.array(occ.shape)
    ii = np.floor((np.asarray(P) - lo) / voxel).astype(int)
    ok = np.all((ii >= 0) & (ii < shape), 1)
    occP = np.zeros(occ.shape, bool); occP[tuple(ii[ok].T)] = True
    occP = ndimage.binary_dilation(occP, iterations=1)
    dist = ndimage.distance_transform_edt(~occP) * voxel
    return (occ | occP) & (dist < near)


def surface_residual(mesh, P, n=60000):
    """distance (m) of points P to the mesh surface (dense surface samples)."""
    import trimesh
    S, _ = trimesh.sample.sample_surface_even(mesh, n, seed=0)
    d, _ = cKDTree(np.asarray(S)).query(P)
    return d
