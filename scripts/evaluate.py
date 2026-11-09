#!/usr/bin/env python
"""Evaluate one episode: tracking quality of the causal (warm, from the episode start), causal COLD-START (tracker started
`n_hist` steps before each window's t0 with no earlier information) and offline variants, and the SE(3)-patch encoder
round trip on world-model windows (n_hist past + t0 + n_fut future steps).

  PLWM_SE3_SRC=<folder with se3patch.py, se3latent.py> python scripts/evaluate.py --episode <dir> --tracks <causal.npz> \
      --out <metrics.json> [--fb_ckpt ae.pt] [--rc_ckpt ae16k.pt] [--depth_cache <dir>]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from realpc.config import build_arms, load_config, tracker_config  # noqa: E402
from realpc.io import DepthSource, Episode  # noqa: E402
from realpc.metrics import measured_to_model, model_to_measured, point_groups, recompute_mdist, robot_vs_depth  # noqa: E402
from realpc.pipeline import Tracks, convert, offline_refine  # noqa: E402


def track_metrics(ep, arms, tr, k0=0, k1=None, mm_every=5):
    k1 = tr.pos.shape[1] if k1 is None else k1
    g = point_groups(tr, k0, k1)
    out = {"model_to_measured": model_to_measured(tr, g, k0, k1)}
    m2 = measured_to_model(ep, arms, tr, g, range(k0, k1, mm_every))
    m2.pop("unexplained_per_step", None)
    out["measured_to_model"] = m2
    alive = np.isfinite(tr.pos[:, k0:k1, 0])
    out["points_alive_mean"] = float(alive.sum(0).mean())
    out["points_born_in_range"] = int(((tr.birth > k0) & (tr.birth < k1)).sum())
    return out


def window_starts(T, n_hist, n_fut, stride):
    return list(range(n_hist, T - n_fut - 1, stride))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--tracks", required=True, help="causal tracks of the whole episode (scripts/convert.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "yam_bimanual.json"))
    ap.add_argument("--fb_ckpt", default=None)
    ap.add_argument("--rc_ckpt", default=None)
    ap.add_argument("--depth_cache", default=None, help="cache dir for decoded depth (robot check vs raw depth)")
    ap.add_argument("--n_hist", type=int, default=16)
    ap.add_argument("--n_fut", type=int, default=32)
    ap.add_argument("--stride", type=int, default=48)
    ap.add_argument("--save_offline", default=None)
    ap.add_argument("--cold_dir", default=None, help="precomputed cold-start windows <cold_dir>/<task>_<t0>.npz (t0 = window t0 step);"
                    " otherwise the ICP tracker is run per window")
    ap.add_argument("--task", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    ep = Episode(a.episode)
    arms = build_arms(cfg, ep.calib)
    t0 = time.time()
    warm = Tracks.load(a.tracks)
    off = offline_refine(warm)
    for tr in (warm, off):
        tr.mdist = recompute_mdist(ep, arms, tr)
    if a.save_offline:
        off.save(a.save_offline)
    print("tracks ready", round(time.time() - t0), "s", flush=True)
    res = {"episode": ep.path.split(os.sep)[-2:], "n_steps": int(warm.pos.shape[1]), "tracking": {}, "windows": [], "roundtrip": {}}
    res["tracking"]["causal_warm"] = track_metrics(ep, arms, warm)
    res["tracking"]["offline"] = track_metrics(ep, arms, off)
    if a.depth_cache:
        srcs = [DepthSource(ep, c, a.depth_cache) for c in ("top", "side")]
        steps = np.linspace(0, warm.pos.shape[1] - 1, 25).astype(int)
        res["tracking"]["robot_vs_raw_depth"] = robot_vs_depth(ep, arms, warm, steps, srcs)
    print("tracking done", round(time.time() - t0), "s", flush=True)
    # cold-start windows
    T = warm.pos.shape[1]
    ks = window_starts(T, a.n_hist, a.n_fut, a.stride)
    if a.cold_dir:
        ks = [k for k in ks if os.path.exists(os.path.join(a.cold_dir, f"{a.task}_{k}.npz"))]
    colds = {}
    agg = {"causal_warm": [], "causal_cold": [], "offline": []}
    for k0 in ks:
        if a.cold_dir:
            cold = Tracks.load(os.path.join(a.cold_dir, f"{a.task}_{k0}.npz"))
        else:
            cold = convert(ep, arms, k0 - a.n_hist, k0 + a.n_fut + 1, cfg=tracker_config(cfg), verbose=False)
        cold.mdist = recompute_mdist(ep, arms, cold)
        colds[k0] = cold
        w = {"t0_step": int(k0), "t0_frame": int(warm.frame_index[k0])}
        for nm, tr, lo, hi in (("causal_warm", warm, k0 - a.n_hist, k0 + a.n_fut + 1), ("offline", off, k0 - a.n_hist, k0 + a.n_fut + 1),
                               ("causal_cold", cold, 0, a.n_hist + a.n_fut + 1)):
            g = point_groups(tr, lo, hi)
            w[nm] = {"model_to_measured": model_to_measured(tr, g, lo, hi)}
            m2 = measured_to_model(ep, arms, tr, g, range(lo, hi, 8))
            w[nm]["unexplained_gt2cm_frac"] = m2["unexplained_gt2cm_frac"]
        res["windows"].append(w)
        print("window", k0, "done", round(time.time() - t0), "s", flush=True)
    # SE(3) round trips
    try:
        from realpc.se3_roundtrip import _load_plwm, build_analytic, build_learned, errors, make_window, roundtrip
        sp, sl = _load_plwm()
    except ImportError as e:
        print("SE(3) round trip skipped:", e)
        json.dump(res, open(a.out, "w"), indent=1)
        return
    aes = [("fb4096_analytic", lambda: build_analytic(sp, 26, 70), 4096, None),
           ("fb4096_analytic_oracle_res", lambda: build_analytic(sp, 26, 70), 4096, 0.02)]
    if a.fb_ckpt:
        aes += [("fb4096_learned", lambda: build_learned(sp, sl, a.fb_ckpt), 4096, None)]
    aes += [("rc16k_analytic", lambda: build_analytic(sp, 24, 120), 16384, None)]
    if a.rc_ckpt:
        aes += [("rc16k_learned", lambda: build_learned(sp, sl, a.rc_ckpt), 16384, None)]
    built = {nm: f() for nm, f, _, _ in aes}
    rt = {}
    for vname, getter in (("causal_warm", lambda k0: (warm, k0)), ("causal_cold", lambda k0: (colds[k0], a.n_hist)),
                          ("offline", lambda k0: (off, k0))):
        for nm, _, npts, orc in aes:
            per = []
            for k0 in ks:
                tr, kk = getter(k0)
                b, info = make_window(tr, kk, a.n_hist, a.n_fut, n_points=npts, seed=k0)
                dec, seg, _ = roundtrip(built[nm], b, oracle_res=orc)
                e, _ = errors(dec, info, seg, a.n_hist)
                per.append(e)
            summ = {}
            for gname in ("gripper", "robot", "static", "moving", "cloth", "static_in_moving_tokens"):
                vals = [p[gname] for p in per if gname in p]
                if vals:
                    summ[gname] = {"mean_mm": float(np.mean([v["mean_mm"] for v in vals])),
                                   "p90_mm": float(np.mean([v["p90_mm"] for v in vals])),
                                   "n_windows": len(vals)}
                    if "per_step_mean_mm" in vals[0]:
                        summ[gname]["per_step_mean_mm"] = [float(x) for x in np.nanmean([v["per_step_mean_mm"] for v in vals], 0)]
            rt.setdefault(vname, {})[nm] = summ
        print("roundtrip", vname, "done", round(time.time() - t0), "s", flush=True)
    res["roundtrip"] = rt
    res["window_t0_steps"] = ks
    json.dump(res, open(a.out, "w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
