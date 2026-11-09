#!/usr/bin/env python
"""SE(3)-patch encoder round trip on v4 converted episodes, with scene patches seeded WITHIN object segments.

The v4 points carry their segment (each library object / static / cloth), so the scene-token budget is split across
segments (>= 2 tokens per object, the rest by sqrt(point count)) and FPS seeding + assignment happen inside each segment:
no token mixes an object with the static scene. `--tok default` keeps the encoder's own tokenisation for comparison.

  PLWM_SE3_SRC=<dir with se3patch.py, se3latent.py> python scripts/roundtrip_v4.py --converted <npz> ... --out <json> \
      [--fb_ckpt fb.pt] [--rc_ckpt rc.pt]
"""
import argparse
import json
import os
import sys
import types

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from realpc.dataset import load  # noqa: E402
from realpc.se3_roundtrip import _load_plwm, build_analytic, build_learned, errors, make_window, roundtrip  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--converted", nargs="+", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--fb_ckpt", default=None); ap.add_argument("--rc_ckpt", default=None)
ap.add_argument("--n_hist", type=int, default=16); ap.add_argument("--n_fut", type=int, default=32); ap.add_argument("--stride", type=int, default=48)
a = ap.parse_args()
sp, sl = _load_plwm()


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


def segment_tokenise(self, a_, g, body, valid):
    """scene tokens seeded within segments (body id >= 998 marks the scene segment)."""
    cfg = self.cfg
    seg, kind = self._orig_tokenise(a_, g, body, valid)            # robot part as the encoder does it
    sc = valid & (g >= 2)
    if not sc.any():
        return seg, kind
    seg[sc] = -1
    kind[cfg.n_robot:] = -1
    ids = torch.unique(body[sc])
    cnt = torch.stack([(body[sc] == i).sum() for i in ids]).float()
    budget = cfg.n_scene
    isobj = ids >= 1000
    alloc = torch.where(isobj, torch.full_like(cnt, 2.0), torch.ones_like(cnt))
    rest = budget - alloc.sum()
    share = cnt.sqrt() / cnt.sqrt().sum() * rest
    alloc = (alloc + share.floor()).long()
    while alloc.sum() < budget:
        alloc[torch.argmax(cnt / alloc.float())] += 1
    nxt = cfg.n_robot
    for i, k in zip(ids.tolist(), alloc.tolist()):
        m = sc & (body == i)
        P = a_[m]
        k = int(min(k, len(P)))
        if k <= 0:
            continue
        si = self._wfps(P, torch.ones(len(P), device=P.device), k)
        lab = torch.cdist(P, P[si]).argmin(1)
        seg[m] = nxt + lab
        kind[nxt: nxt + k] = 2
        nxt += k
    return seg, kind


def patch(ae, mode):
    if mode == "segments":
        base = ae.base if hasattr(ae, "base") else ae
        for m in {ae, base}:
            if hasattr(m, "tokenise"):
                m._orig_tokenise = m.tokenise
                m._wfps = sp.wfps
                m.tokenise = types.MethodType(segment_tokenise, m)
    return ae


aes = [("fb4096_analytic", lambda: build_analytic(sp, 26, 70), 4096), ("rc16k_analytic", lambda: build_analytic(sp, 24, 120), 16384)]
if a.fb_ckpt:
    aes.append(("fb4096_learned", lambda: build_learned(sp, sl, a.fb_ckpt), 4096))
if a.rc_ckpt:
    aes.append(("rc16k_learned", lambda: build_learned(sp, sl, a.rc_ckpt), 16384))
out = {}
for cp in a.converted:
    d = load(cp)
    tr = TracksView(d)
    T = tr.pos.shape[1]
    k0s = list(range(a.n_hist, T - a.n_fut - 1, a.stride))
    key = f"{d['task']}/{d['episode']}"
    out[key] = {}
    for nm, mk, npts in aes:
        for tok in ("default", "segments"):
            ae = patch(mk(), tok)
            res = []
            for k0 in k0s:
                b, info = make_window(tr, k0, a.n_hist, a.n_fut, n_points=npts, seed=k0)
                # body ids reach the encoder: scene segment ids >= 998
                dec, seg, _ = roundtrip(ae, b)
                e, _ = errors(dec, info, seg, a.n_hist)
                res.append(e)
            agg = {}
            for gname in ("gripper", "robot", "static", "moving", "cloth", "static_in_moving_tokens"):
                v = [r[gname]["mean_mm"] for r in res if gname in r]
                p = [r[gname].get("p90_mm") for r in res if gname in r and "p90_mm" in r[gname]]
                if v:
                    agg[gname] = {"mean_mm": float(np.mean(v)), "p90_mm": float(np.mean(p)) if p else None, "windows": len(v)}
            out[key][f"{nm}/{tok}"] = agg
            print(key, nm, tok, {k: round(v["mean_mm"], 2) for k, v in agg.items()}, flush=True)
json.dump(out, open(a.out, "w"), indent=1)
