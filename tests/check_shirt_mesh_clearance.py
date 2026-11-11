"""A flat shirt must not consume the tracker's self-clearance in slivers."""
import sys
from pathlib import Path

import ipctk
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from realpc.shirt_template import make_panels

# Fitted outline that previously admitted an interior grid vertex 0.183 mm
# from a boundary edge despite staying far from the edge's endpoints.
poly = np.array([
    [-.18871190, -.28122247], [.20312240, -.25577953],
    [.17292446, -.06542240], [.23492659, -.09250403],
    [.31329175, .01653243], [.17918705, .11362624],
    [.06538387, .14451682], [.04988485, .10663647],
    [.00402107, .08348113], [-.03807527, .10408693],
    [-.05409247, .14379758], [-.16598236, .12496356],
    [-.32657787, .01406615], [-.25714095, -.09577799],
    [-.19216122, -.06100090],
])
v, f, n, pair = make_panels(poly, cell=.018, layers=1)
assert np.array_equal(np.unique(f), np.arange(n))
assert np.array_equal(pair, np.arange(n))
edges, counts = np.unique(np.sort(np.vstack([
    f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1), axis=0, return_counts=True)
assert counts.max() == 2
ipctk.set_num_threads(1)
mesh = ipctk.CollisionMesh(v, edges.astype(np.int32), f.astype(np.int32))
collisions = ipctk.NormalCollisions()
collisions.build(mesh, v, .02)
gap = float(np.sqrt(collisions.compute_minimum_distance(mesh, v)))
assert not ipctk.has_intersections(mesh, v)
assert gap > .00024, gap  # tracker clearance plus its initialization margin
print('PASS flat sheet topology and self-clearance:', gap * 1000, 'mm')
