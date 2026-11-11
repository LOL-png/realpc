"""Current-frame cloth quality flags shared by live and recording callers."""
import numpy as np


def frame_quality(points, metrics, config):
    limits = config['segmentation']
    bound = float(config['cloth'].get('max_stretch', 0.))
    finite = bool(np.isfinite(points).all())
    crossing = metrics.get('ipc_intersections')
    crossing_checked = crossing is not None
    no_crossing = crossing_checked and not bool(crossing)
    strain = metrics.get('max_edge_strain', float('inf'))
    within_bound = bool(bound <= 0 or (np.isfinite(strain) and strain <= bound + 1e-6))
    views = {}
    for camera in ('top', 'side'):
        row = metrics.get('cameras', {}).get(camera, {})
        coverage = row.get('surface_coverage_10mm', float('nan'))
        depth = row.get('surface_to_mesh_med_mm', float('nan'))
        views[camera] = bool(np.isfinite(coverage) and np.isfinite(depth)
            and coverage >= limits.get('cloth_fit_min_coverage', .5)
            and depth <= limits.get('cloth_fit_max_median_mm', 25.))
    valid = bool(finite and no_crossing and within_bound and all(views.values()))
    return dict(valid=valid, approximate=True, finite=finite,
                crossing_checked=crossing_checked, no_detected_crossing=no_crossing,
                strain_bound_enforced=bound > 0, within_configured_strain=within_bound,
                supported_views=views)
