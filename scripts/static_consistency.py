#!/usr/bin/env python
"""Top-vs-side static scene consistency of a calibration: side depth points (table region, robot removed) -> nearest
top depth point, median / p90 distance (mm), on the first tick of each episode (robot at rest, nothing moving).

  python scripts/static_consistency.py --calib <json|recorder> --episodes <ep> ...
"""
import argparse, json, os, sys
import numpy as np
from scipy.spatial import cKDTree
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from realpc.calib_rig import RigCalib  # noqa: E402
from realpc.config import build_arms, load_config  # noqa: E402
from realpc.io import Episode, read_depth_frames  # noqa: E402
from realpc import objlib  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--calib", required=True); ap.add_argument("--episodes", nargs="+", required=True)
ap.add_argument("--tick", type=int, default=0)
a = ap.parse_args()
cfg = load_config(os.path.join(ROOT, "configs", "yam_bimanual.json"))
out = []
for e in a.episodes:
    ep = Episode(e)
    cal = RigCalib.recorder(ep) if a.calib == "recorder" else RigCalib.load(a.calib)
    region = json.load(open(os.path.join(ep.path, "table_region.json")))
    arms = build_arms(cfg, ep.calib); arms.T_base["right"] = cal.T_world_rb()
    Ld, Bd = arms.dense_local(800)
    LP = arms.link_poses({s: ep.low[f"joint_pos_{s}"][a.tick] for s in ("left", "right")})
    Rb = np.einsum("nij,nj->ni", LP[Bd, :3, :3], Ld) + LP[Bd, :3, 3]
    P = {}
    for c in ("top", "side"):
        D = read_depth_frames(os.path.join(ep.path, f"{c}_depth.mkv"), [a.tick])[a.tick].astype(np.float32) * float(ep.meta["depth_scale"][c])
        X, _, _, _ = objlib.depth_points(D, np.zeros(D.shape + (3,), np.uint8), ep.intrinsics(c), cal.T_world_cam(c), stride=2)
        k = objlib.table_crop(X, region, *ep.table_plane(), above=-0.05) & objlib.robot_free(X, Rb, 0.05)
        P[c] = X[k]
    d, _ = cKDTree(P["top"]).query(P["side"], distance_upper_bound=0.08)
    d = d[np.isfinite(d)] * 1000
    out.append((np.median(d), np.percentile(d, 90)))
    print(f"{e.split('hf/')[-1]}: side->top static median {np.median(d):.1f} mm, p90 {np.percentile(d, 90):.1f} mm (n {len(d)})")
o = np.array(out); print(f"ALL {a.calib}: median {np.median(o[:,0]):.1f} mm, p90 {np.median(o[:,1]):.1f} mm")
