"""Report exported cloth strain separately from visible alignment.

This is an offline audit. Tracker intersection flags are not an independent
collision or finite-thickness certificate.
"""
import argparse
import json
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--root', required=True)
p.add_argument('--out', required=True)
a = p.parse_args()
result = {}
for n in (1, 2, 3, 10):
    stem = Path(a.root) / f'n{n}/data/greycloth/episode_0000'
    with np.load(str(stem) + '_cloth_mesh.npz') as z:
        x, rest, faces = (z[k].astype(np.float64 if k != 'F' else np.int64)
                          for k in ('X', 'V0', 'F'))
    edges = np.unique(np.sort(np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    length = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=-1)
    assert np.all(length > 1e-9)
    ratios = np.linalg.norm(x[:, edges[:, 0]] - x[:, edges[:, 1]], axis=-1) / length
    metrics = json.loads(Path(str(stem) + '_cloth_metrics.json').read_text())
    assert len(x) == len(metrics) == 453
    result[n] = dict(
        max_extension_percent=float(np.maximum(ratios - 1, 0).max() * 100),
        max_compression_percent=float(np.maximum(1 - ratios, 0).max() * 100),
        tracker_crossing_frames=sum(bool(r.get('ipc_intersections')) for r in metrics),
        max_grasp_mm=max(r['grasp_error_mm'] for r in metrics),
        per_frame_extension_percent=(np.maximum(ratios - 1, 0).max(axis=1) * 100).tolist(),
        per_frame_compression_percent=(np.maximum(1 - ratios, 0).max(axis=1) * 100).tolist(),
        per_frame_grasp_mm=[r['grasp_error_mm'] for r in metrics],
    )
    print(n, {k: v for k, v in result[n].items() if not k.startswith('per_frame')})
Path(a.out).write_text(json.dumps(dict(
    rates=result,
    note='Strain relative to exported rest mesh. Intersection flags come from the tracker, '
         'not an independent collision audit. These metrics do not validate hidden layers.'), indent=2))
