#!/usr/bin/env python
"""Dataset manifest of a v4 conversion directory: versions, schema, per-episode QA, aggregate stats, sha256 per file.

  python scripts/make_manifest.py --out_dir ~/data_ext/realpc_out/v1 [--align <align json> ...]
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from realpc.v4 import SCHEMA  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out_dir", required=True); ap.add_argument("--align", nargs="*", default=[])
ap.add_argument("--calib", default=os.path.join(ROOT, "configs", "calib_yam_rig_v3.json"))
a = ap.parse_args()


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


cal = json.load(open(a.calib))
lib = {}
for f in sorted(glob.glob(os.path.join(ROOT, "assets", "objects", "*", "*.json"))):
    m = json.load(open(f))
    lib[f"{m['task']}/{m['name']}"] = {"version": m.get("version", "v1"), "n_scans": m.get("n_scans", len(m.get("scans", []))),
                                      "extents_m": m["extents_m"], "watertight": m["watertight"],
                                      "ply_sha256": sha(f[:-5] + ".ply")}
eps = []
for p in sorted(glob.glob(os.path.join(a.out_dir, "*", "episode_*.npz"))):
    q = p[:-4] + "_qa.json"
    qa = json.load(open(q)) if os.path.exists(q) else {}
    eps.append({"file": os.path.relpath(p, a.out_dir), "sha256": sha(p), "bytes": os.path.getsize(p), "qa": qa})
commit = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
res = [o["res_med_mm"] for e in eps for o in e["qa"].get("objects", {}).values() if o.get("res_med_mm") is not None]
man = {
    "dataset": "agentwm_v1 (ChongkaiGao/agentworldmodel, 52 episodes: sweep 50, hang 1, greycloth 1)",
    "created": datetime.datetime.now().isoformat(timespec="seconds"), "code_commit": commit,
    "converter": "realpc.v4.StreamConverter (causal: every tick uses frames <= t; the same class serves live inference)",
    "causal": True,
    "calibration": {"file": "configs/calib_yam_rig_v3.json", "version": cal.get("version"), "const_lag_ticks": cal.get("const_lag"),
                    "corrections": cal.get("corrections"), "joint_offsets_rad": cal.get("joint_offsets_rad")},
    "object_masks": "SAM2.1 hiera-small video propagation, forward only from the first tick, seeded with renders of the library "
                    "meshes at their initial poses + the FK arms (scripts/segment_objects.py); no hand labels",
    "library": lib,
    "groups": {"0": "robot_left (incl. fingers)", "1": "robot_right (incl. fingers)", "2": "static scene (table region, first tick)",
               "3": "library object (obj_id)", "4": "cloth (greycloth; valid before the fold only)"},
    "points_per_episode": {"robot": 9600, "object": "1500 per library object", "static": "~8-9K (8 mm voxels)", "cloth": 4000},
    "schema": SCHEMA, "loader": "realpc.dataset.load(npz) -> points [T,N,3], group, obj_id, is_finger, color, valid",
    "alignment_checks": [json.load(open(x))["summary"] | {"episode": json.load(open(x))["episode"]} for x in a.align if os.path.exists(x)],
    "aggregate": {"episodes": len(eps), "total_bytes": int(sum(e["bytes"] for e in eps)),
                  "object_depth_residual_median_mm": float(np.median(res)) if res else None},
    "episodes": eps,
}
json.dump(man, open(os.path.join(a.out_dir, "manifest.json"), "w"), indent=1)
print("episodes", len(eps), "bytes", man["aggregate"]["total_bytes"], "-> manifest.json")
