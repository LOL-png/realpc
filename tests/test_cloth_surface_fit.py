import numpy as np

from realpc.cloth_surface_fit import coverage_system


def test_aligned_interior_observations_do_not_pull_vertices_inward():
    X = np.array([[0., 0., 0.], [.1, 0., 0.], [0., .1, 0.]])
    F = np.array([[0, 1, 2]])
    points = np.array([[.02, .03, 0.], [.04, .01, 0.], [.01, .07, 0.]])
    A, b = coverage_system(X, F, points, np.ones(3, bool), .1, 8.)
    np.testing.assert_allclose(A @ X, b, atol=1e-12)
    np.testing.assert_allclose(A.toarray(), A.toarray().T, atol=1e-12)
    assert np.linalg.eigvalsh(A.toarray()).min() > -1e-10


def test_duplicate_depth_samples_do_not_increase_force():
    X = np.array([[0., 0., 0.], [.1, 0., 0.], [0., .1, 0.]])
    F = np.array([[0, 1, 2]])
    points = np.array([[.02, .03, .01], [.04, .01, .01]])
    A, b = coverage_system(X, F, points, np.ones(3, bool), .1, 8.)
    C, d = coverage_system(X, F, np.repeat(points, 5, axis=0), np.ones(3, bool), .1, 8.)
    np.testing.assert_allclose(A.toarray(), C.toarray(), atol=1e-12)
    np.testing.assert_allclose(b, d, atol=1e-12)
    assert (b - A @ X)[:, 2].sum() > 0


def test_invisible_or_distant_triangles_supply_no_fit_force():
    X = np.array([[0., 0., 0.], [.1, 0., 0.], [0., .1, 0.]])
    F = np.array([[0, 1, 2]])
    for points, visible in [(np.ones((1, 3)), np.ones(3, bool)),
                            (np.zeros((1, 3)), np.zeros(3, bool))]:
        A, b = coverage_system(X, F, points, visible, .01, 8.)
        assert A.nnz == 0
        assert not b.any()
