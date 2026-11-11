"""Recorded contact-association regression; expected IDs are test annotations.

The runtime rule consumes only current RGB-D and its past material tracks.
Requires an explicitly supplied trusted local pre-acquisition checkpoint.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
p = argparse.ArgumentParser()
p.add_argument('--state', required=True)
p.add_argument('--inputs', required=True)
p.add_argument('--expected-id', type=int, required=True)
p.add_argument('--arm', default='left', choices=['left', 'right'])
a = p.parse_args()

for held_only in (False, False, True):
    with open(a.state, 'rb') as f:
        tracker, history, metrics = pickle.load(f)
    t = len(history)
    packet = dict(np.load(Path(a.inputs) / 'frames' / f'{t:04d}.npz'))
    tracker.track_grasp = True
    tracker.tick = t
    if held_only:
        for track in tracker.tracks.values():
            track['held'][:] = True
    snapshots = {c: {k: v.copy() for k, v in track.items()}
                 for c, track in tracker.tracks.items()}
    rest = tracker.cm.V0.copy()
    tracker.update_pins(packet, tracker.cm.X, tracker.views(packet))
    event = next(e for e in reversed(tracker.events)
                 if e['t'] == t and e['arm'] == a.arm)
    association = event['contact_association']
    if held_only:
        assert association['track_count'] == 0
        assert not association['applied']
    else:
        assert event['seed'] == a.expected_id, event
        assert association['source'] == 'persistent_material_tracks'
        assert association['applied']
    for camera, before in snapshots.items():
        for key, value in before.items():
            np.testing.assert_array_equal(value, tracker.tracks[camera][key])
    np.testing.assert_array_equal(rest, tracker.cm.V0)
    json.dumps(event)  # Saved diagnostics must be serializable.
    print('PASS', 'evaluation tracks excluded' if held_only else 'material vote', event)
