#!/usr/bin/env python
"""Verification video of converted tracks over the episode's camera images (H.264 mp4).

  python scripts/verify_video.py --episode <dir> --tracks tracks.npz --out verify.mp4 [--stride 1]

Panels: top photo + tracked objects (one colour per moving object, static objects faint, robot = FK mesh silhouette)
        | side photo (synced) + the same / fixed 3D view over the measured cloud | per-object pose residual vs time
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from realpc.io import Episode  # noqa: E402
from realpc.pipeline import Tracks  # noqa: E402
from realpc.viz import render_tracks  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--episode", required=True)
ap.add_argument("--tracks", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--stride", type=int, default=1)
ap.add_argument("--title", default=None)
ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "yam_bimanual.json"))
ap.add_argument("--sync_cache", default=None, help="cache dir for the side-camera sync map (realpc.sync); omit = no sync")
a = ap.parse_args()
ep = Episode(a.episode)
tr = Tracks.load(a.tracks)
from realpc.config import build_arms, load_config  # noqa: E402
from realpc.sync import side_frame_map  # noqa: E402
arms = build_arms(load_config(a.config), ep.calib)
sm = side_frame_map(ep, a.sync_cache) if a.sync_cache else None
render_tracks(ep, tr, a.out, title=a.title or f"{ep.task}/{ep.name} [{tr.variant}]", stride=a.stride, arms=arms, side_map=sm)
print("saved", a.out)
