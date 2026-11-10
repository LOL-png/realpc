"""Adapt converted realpc datasets to the encoder window interface."""
import numpy as np

class TracksView:
    """adapter: v4 dataset -> the Tracks fields make_window uses (robot first)."""

    def __init__(self, d):
        g = d["group"]
        rob = g <= 1
        order = np.r_[np.flatnonzero(rob), np.flatnonzero(~rob)]
        self.pos = d["points"].transpose(1, 0, 2)[order].copy()
        self.pos[~d["valid"].T[order]] = np.nan
        gg = np.where(rob, np.where(d["is_finger"], 0, 1), np.where(g == 2, 2, np.where(g == 3, 3, 4)))[order]
        self.group = gg
        # robot: link body ids; scene: segment id (objects 1000+obj, static 999, cloth 998)
        nb = len(d["robot_bodies"])
        rb = np.r_[d["robot_body"].astype(int), np.zeros(int((~rob).sum()), int)]
        seg = np.where(g == 3, 1000 + d["obj_id"].astype(int), np.where(g == 2, 999, 998))
        self.body = np.where(np.r_[np.ones(int(rob.sum()), bool), np.zeros(int((~rob).sum()), bool)], rb, seg[order])
        self.n_robot = int(rob.sum())


