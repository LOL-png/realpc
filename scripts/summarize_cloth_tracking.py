"""Report held-out image-track and grasp errors, separately from depth fit."""
import argparse
import json
from pathlib import Path

import numpy as np


def summary(series):
    count = min(map(len, series.values()))
    result = {}
    for name, rows in series.items():
        phases = {}
        for phase, lo, hi in [('before_lift', 0, 160), ('first_fold', 160, 250),
                              ('between_folds', 250, 335), ('second_fold', 335, count)]:
            part = rows[lo:min(hi, count)]
            if not part:
                continue
            stats = dict(frames=len(part), held_frames=sum(r.get('pins', 0) > 0 for r in part))
            for key in ('hard_grasp_error_mm', 'grasp_error_mm'):
                values = [r[key] for r in part if r.get('pins', 0) > 0 and r.get(key) is not None]
                stats[key] = dict(maximum=float(max(values)), median=float(np.median(values))) if values else None
            for kind, value_key, count_key in [('heldout_tracks', 'long_median_px', 'long_n'),
                                               ('cotracker', 'held_median_px', 'held_n')]:
                stats[kind] = {}
                for camera in ('top', 'side'):
                    values = [r.get(kind, {}).get(camera, {}) for r in part]
                    values = [r for r in values if r.get(count_key, 0) >= 5 and r.get(value_key) is not None]
                    stats[kind][camera] = dict(
                        evaluated_frames=len(values),
                        median_of_frame_medians_px=float(np.median([r[value_key] for r in values])) if values else None,
                        median_track_count=float(np.median([r[count_key] for r in values])) if values else None)
            phases[phase] = stats
        result[name] = phases
    return dict(matched_frames=count, models=result,
        note='These 2D track constraints are excluded from temporal fitting; camera depth still contributes to surface fitting. LK long tracks have age >=30; frames require >=5 tracks. Feature populations can differ across fits, so these are diagnostics, not matched ground-truth material trajectories. Grasp statistics use held frames only; hard centres and the assumed soft neighbourhood are separate errors. Depth/silhouette audits include all frames independently.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--metrics', action='append', required=True, help='label=metrics.json')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    series = {}
    for item in a.metrics:
        name, path = item.split('=', 1)
        series[name] = json.loads(Path(path).read_text())
    report = summary(series)
    Path(a.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
