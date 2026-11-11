"""Dataset flags must preserve live quality, including late valid frames."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from realpc.cloth_quality import frame_quality

spec = importlib.util.spec_from_file_location(
    'verify_scene_dataset', Path(__file__).resolve().parents[1] / 'scripts/verify_scene_dataset.py')
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


def example():
    config = dict(segmentation={}, cloth=dict(max_stretch=.02))
    metric = dict(ipc_intersections=False, max_edge_strain=.01,
                  cameras={c: dict(surface_coverage_10mm=.9, surface_to_mesh_med_mm=3.)
                           for c in ('top', 'side')})
    # A supported late frame is valid, an unsupported early frame is not.
    metrics = []
    points = np.zeros((2, 152, 3), np.float32)
    for t in range(152):
        row = copy.deepcopy(metric)
        row['t'] = t
        if t == 1:
            row['cameras']['top']['surface_coverage_10mm'] = .1
        row['quality'] = frame_quality(points[:, t], row, config)
        metrics.append(row)
    data = dict(ticks=np.arange(152), cloth_points=points,
                cloth_valid=np.array([m['quality']['valid'] for m in metrics]),
                cloth_approximate=np.ones(152, bool), cloth_fold_approximate=np.ones(152, bool),
                cloth_geometry_observed_only=False)
    return data, metrics, config


def test_current_quality_instead_of_recording_cutoff():
    data, metrics, config = example()
    verify.verify_cloth_quality(data, metrics, config)
    assert data['cloth_valid'][151] and not data['cloth_valid'][1]


def test_reject_changed_validity_or_fabricated_quality():
    data, metrics, config = example()
    data['cloth_valid'][1] = True
    with pytest.raises(AssertionError):
        verify.verify_cloth_quality(data, metrics, config)
    metrics[1]['quality']['valid'] = True
    with pytest.raises(AssertionError):
        verify.verify_cloth_quality(data, metrics, config)


def test_approximate_flag_cannot_be_dropped():
    data, metrics, config = example()
    data['cloth_approximate'][0] = False
    with pytest.raises(AssertionError):
        verify.verify_cloth_quality(data, metrics, config)
