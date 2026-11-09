#!/usr/bin/env python
"""v3 conversion (refined depth + CoTracker3 online tracks lifted to 3D -> per-object SE(3)), causal.

  python scripts/convert_v3.py --episode <dir> --out tracks.npz [--start S --stop E] [--set allow_deform=true]
Model weights are fetched once into <repo>/.model_cache (TORCH_HOME), not committed.
"""
import argparse
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from realpc import depth_refine  # noqa: E402
from realpc.config import build_arms, load_config, tracker_config  # noqa: E402
from realpc.io import DepthSource, Episode  # noqa: E402
from realpc.v3 import V3Converter, load_cotracker  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "yam_bimanual.json"))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=None)
    ap.add_argument("--depth_cache", default=os.path.join(ROOT, ".cache"))
    ap.add_argument("--raw_depth", action="store_true", help="skip the colour-guided refinement")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config)
    over = {"motion": "tracks"}
    for kv in a.set:
        k, v = kv.split("=", 1)
        over[k] = {"true": True, "false": False}.get(v.lower(), v)
        if isinstance(over[k], str):
            try:
                over[k] = float(v) if "." in v else int(v)
            except ValueError:
                pass
    ep = Episode(a.episode)
    arms = build_arms(cfg, ep.calib)
    src = DepthSource(ep, "top", a.depth_cache)

    def refine(rgb, tick):
        raw = np.asarray(src.D[tick])
        if a.raw_depth:
            return raw.astype(np.float32) * src.scale
        return depth_refine.refine_frame(rgb, raw, src.scale).float().cpu().numpy()

    model = load_cotracker(os.path.join(ROOT, ".model_cache"))
    conv = V3Converter(ep, arms, tracker_config(cfg, **over), model, refine)
    tr = conv.convert(a.start, a.stop)
    tr.save(a.out)
    print("saved", a.out, "points", tr.pos.shape[0], "steps", tr.pos.shape[1], "variant", tr.variant)


if __name__ == "__main__":
    main()
