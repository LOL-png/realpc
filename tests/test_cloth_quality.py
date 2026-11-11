import copy
import numpy as np

from realpc.cloth_quality import frame_quality


def inputs():
    config = dict(segmentation=dict(cloth_fit_min_coverage=.5, cloth_fit_max_median_mm=25.),
                  cloth=dict(max_stretch=.02))
    metrics = dict(ipc_intersections=False, max_edge_strain=.019,
                   cameras={c: dict(surface_coverage_10mm=.85, surface_to_mesh_med_mm=5.)
                            for c in ('top', 'side')})
    return np.zeros((8, 3)), metrics, config


def test_quality_has_no_recording_phase_cutoff():
    points, metrics, config = inputs()
    before = frame_quality(points, dict(metrics, tick=0), config)
    after = frame_quality(points, dict(metrics, tick=300), config)
    assert before == after
    assert before['valid'] and before['approximate']
    assert before['strain_bound_enforced']


def test_failed_observation_or_physical_check_is_invalid():
    points, base, config = inputs()
    failures = [dict(ipc_intersections=True), dict(ipc_intersections=None), dict(max_edge_strain=.05)]
    for change in failures:
        assert not frame_quality(points, base | change, config)['valid']
    for value in (.2, float('nan')):
        metrics = copy.deepcopy(base)
        metrics['cameras']['side']['surface_coverage_10mm'] = value
        assert not frame_quality(points, metrics, config)['valid']
    assert not frame_quality(points * float('nan'), base, config)['valid']
    assert not frame_quality(points, base | dict(cameras={}), config)['valid']


def test_unbounded_mode_is_explicit_and_not_a_physical_certificate():
    points, metrics, config = inputs()
    config['cloth']['max_stretch'] = 0.
    quality = frame_quality(points, metrics, config)
    assert quality['valid'] and quality['approximate']
    assert not quality['strain_bound_enforced']


def test_contact_lag_gate_uses_current_hold_and_preserves_old_configs():
    points, metrics, config = inputs()
    failed = metrics | dict(pins=3, hard_grasp_error_mm=36.6)
    assert frame_quality(points, failed, config)['valid']  # historical policy
    config['segmentation']['cloth_fit_max_hard_grasp_mm'] = 5.
    assert not frame_quality(points, failed, config)['valid']
    assert frame_quality(points, failed | dict(hard_grasp_error_mm=1.), config)['valid']
    assert frame_quality(points, failed | dict(pins=0), config)['valid']
    assert not frame_quality(points, metrics, config)['valid']
    assert not frame_quality(points, failed | dict(hard_grasp_error_mm=float('nan')), config)['valid']
