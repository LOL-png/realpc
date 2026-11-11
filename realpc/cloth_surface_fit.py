"""Current-view point-to-triangle coverage, without snapping data to vertices."""
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree


def coverage_system(X, F, points, eligible, radius, weight, candidates=4):
    """Return A,b for .5*sum(w*||barycentric(X)-observed||^2).

    Correspondences are fixed within one local solve and recomputed afterward.
    Restrict to triangles with visible vertices; normalize observation density
    so a densely sampled camera does not receive a larger total force.
    """
    n = len(X)
    faces = F[np.asarray(eligible)[F].all(axis=1)]
    if not len(faces) or not len(points):
        return sp.csr_matrix((n, n)), np.zeros((n, 3))
    from trimesh.triangles import closest_point, points_to_barycentric
    triangles = X[faces]
    k = min(candidates, len(faces))
    _, near = cKDTree(triangles.mean(axis=1)).query(points, k=k)
    near = np.asarray(near).reshape(len(points), k)
    trial = triangles[near].reshape(-1, 3, 3)
    obs = np.repeat(points, k, axis=0)
    closest = closest_point(trial, obs).reshape(len(points), k, 3)
    squared = ((closest - points[:, None]) ** 2).sum(axis=2)
    best = squared.argmin(axis=1)
    rows = np.arange(len(points))
    good = squared[rows, best] < radius ** 2
    ids = near[rows, best][good]
    positions = closest[rows, best][good]
    bary = points_to_barycentric(triangles[ids], positions)
    bary = np.clip(bary, 0., 1.)
    bary /= np.maximum(bary.sum(axis=1, keepdims=True), 1e-12)
    indices = faces[ids]
    B = sp.coo_matrix((bary.ravel(),
        (np.repeat(np.arange(len(ids)), 3), indices.ravel())),
        shape=(len(ids), n)).tocsr()
    mass = np.asarray(B.sum(axis=0)).ravel()
    weights = weight / np.maximum(B @ mass, 1e-9)
    return B.T @ sp.diags(weights) @ B, B.T @ (weights[:, None] * points[good])
