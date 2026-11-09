#!/usr/bin/env python
"""Batch conversion: recorded episode -> time-consistent 4D point tracks (.npz).

  python scripts/convert.py --episode <episode_dir> --out tracks.npz [--start S --stop E] [--offline]

Uses exactly the streaming tracker (realpc.tracker.Tracker.step) frame by frame; --start lets it begin mid-episode with
no earlier information (what a model sees at inference). --offline additionally runs the whole-sequence refinement
(back-fill + pose smoothing) and writes <out> with variant "offline".
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from realpc.config import build_arms, load_config, table_plane, tracker_config  # noqa: E402
from realpc.io import Episode  # noqa: E402
from realpc.pipeline import convert, offline_refine  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "yam_bimanual.json"))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=None)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="tracker overrides key=value, e.g. allow_deform=true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    ep = Episode(a.episode)
    arms = build_arms(cfg, ep.calib)
    over = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        over[k] = {"true": True, "false": False}.get(v.lower(), None)
        if over[k] is None:
            over[k] = float(v) if "." in v else int(v)
    tr = convert(ep, arms, a.start, a.stop, cfg=tracker_config(cfg, **over))
    if a.offline:
        tr = offline_refine(tr)
    tr.save(a.out)
    print("saved", a.out, "points", tr.pos.shape[0], "steps", tr.pos.shape[1], "variant", tr.variant)


if __name__ == "__main__":
    main()
