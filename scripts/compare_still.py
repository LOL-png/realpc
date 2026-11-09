#!/usr/bin/env python
"""Side-by-side still: converted tracks vs the harness's offline refined depth (final.npy, 0.05 mm units).

  python scripts/compare_still.py --episode <dir> --tracks t.npz --harness_final <cache>/final.npy --step K --out x.png
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from realpc.io import Episode  # noqa: E402
from realpc.pipeline import Tracks  # noqa: E402
from realpc.viz import compare_still  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--episode", required=True); ap.add_argument("--tracks", required=True)
ap.add_argument("--cam", default="top"); ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "yam_bimanual.json"))
ap.add_argument("--harness_final", required=True); ap.add_argument("--step", type=int, required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
ep = Episode(a.episode); tr = Tracks.load(a.tracks)
F = np.load(a.harness_final, mmap_mode="r")
from realpc.config import build_arms, load_config  # noqa: E402
arms = build_arms(load_config(a.config), ep.calib)
compare_still(ep, tr, np.asarray(F[int(tr.ticks[a.step])], np.float32) / 20000.0, a.step, a.out, cam=a.cam, arms=arms)
print("saved", a.out)
