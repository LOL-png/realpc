"""Load a rig config (configs/*.json) and build the robot model + tracker config for an episode."""
from __future__ import annotations

import json
import os

import numpy as np

from .robot import BimanualArms
from .tracker import TrackerConfig


def load_config(path: str) -> dict:
    cfg = json.load(open(path))
    cfg["_dir"] = os.path.dirname(os.path.abspath(path))
    return cfg


def build_arms(cfg: dict, calib: dict, seed: int = 0) -> BimanualArms:
    r = cfg["robot"]
    c = cfg.get("calibration", {})
    T_r = np.asarray(c["T_world_right_base"], float) if c.get("T_world_right_base") is not None else np.asarray(calib["T_lb_rb"], float)
    return BimanualArms(os.path.join(cfg["_dir"], r["urdf"]), {"left": np.eye(4), "right": T_r}, r["arm_joints"], r["finger_joints"],
                        r["points_per_link"], seed=seed)


def table_plane(cfg: dict, calib: dict):
    tp = cfg.get("calibration", {}).get("table_plane")
    if tp is not None:
        return np.asarray(tp["n"], float), float(tp["d"])
    return np.asarray(calib["table_n"], float), float(calib["table_d"])


def tracker_config(cfg: dict, **over) -> TrackerConfig:
    kw = dict(cfg.get("tracker", {}))
    kw.update(over)
    return TrackerConfig(**kw)
