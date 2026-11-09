#!/usr/bin/env python
"""Rig calibration v3: track bundle adjustment through the arm FK (realpc/handeye_ba.py).

Per camera and per arm: T_cam<-base (left and right separately), per (episode, camera) piecewise lag, optional joint
zero offsets. Held out: whole episodes (--holdout) and every 4th track segment of the fit episodes.

  python scripts/calib_v3.py --tracks_dir <dir> --out configs/calib_yam_rig_v3.json [--holdout greycloth/episode_0000 ...]
"""
import argparse
import datetime
import glob
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from realpc.config import load_config  # noqa: E402
from realpc.diffcalib import TorchArm  # noqa: E402
from realpc.handeye_ba import HandEyeBA, Seq  # noqa: E402
from realpc.io import Episode  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tracks_dir", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--data", default=os.path.expanduser("~/data_ext/agentwm_probe/hf"))
ap.add_argument("--holdout", nargs="*", default=[]); ap.add_argument("--iters", type=int, default=1500)
ap.add_argument("--offsets", action="store_true"); ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--life", type=int, default=80)
ap.add_argument("--rig", action="store_true", help="rigid camera rig (recorder top<->side kept), one correction per arm")
ap.add_argument("--const_lag", type=float, default=None, help="one constant lag (ticks) for every camera: no lag search / fit")
ap.add_argument("--masks_root", default=os.path.expanduser("~/data_ext/masks"))
ap.add_argument("--sil_w", type=float, default=2.0); ap.add_argument("--sil_every", type=int, default=4)
ap.add_argument("--depth_w", type=float, default=0.3); ap.add_argument("--depth_sigma", type=float, default=0.01)
a = ap.parse_args()

cfg = load_config(os.path.join(ROOT, "configs", "yam_bimanual.json")); r = cfg["robot"]
arm = TorchArm(os.path.join(cfg["_dir"], r["urdf"]), r["arm_joints"], r["finger_joints"])
seqs, eps = [], {}
for f in sorted(glob.glob(os.path.join(a.tracks_dir, "*.npz"))):
    b = os.path.basename(f)[:-4]
    task, rest = b.split("_", 1)
    name, cam = rest.rsplit("_", 1)
    key = f"{task}/{name}"
    if key not in eps:
        eps[key] = Episode(os.path.join(a.data, key))
    seqs.append(Seq(eps[key], cam, f, eps[key].intrinsics(cam)))
ep0 = next(iter(eps.values()))
T0 = {}
for c in ("top", "side"):
    Ti = np.linalg.inv(np.asarray(ep0.low[f"cam_pose_{c}"][0]))
    T0[(c, "left")] = Ti
    T0[(c, "right")] = Ti @ np.asarray(ep0.calib["T_lb_rb"])
ba = HandEyeBA(arm, T0, rig=a.rig)
nl = len(arm.links)
print("sequences:", [(s.key, s.cam, len(s.starts)) for s in seqs], flush=True)


def is_fit(s, j):
    return s.key not in a.holdout and j % 4 != 3


def is_ho(s, j):
    return not is_fit(s, j)


def gated(s, j, res, max_rms):
    """association gates: small residual, residual small vs the track's own motion, seed depth consistent."""
    tr = s.tracks[j]; vi = s.vis[j]
    rng = np.array([np.ptp(tr[vi[:, i], i], 0).max() if vi[:, i].any() else 0 for i in range(tr.shape[1])])
    ok = (res["rms"] < max_rms) & (res["rms"] < 0.25 * rng)
    # seed depth: triangulated camera-frame depth of the point at the seed frame vs the measured depth
    d0 = s.depth0[j]
    h = res["h"]
    with torch.no_grad():
        P = ba.proj_mats(s, torch.tensor([float(s.starts[j]) - res["lag"]], dtype=torch.float64, device="cuda"))
        Kinv = np.linalg.inv(s.K)
        P = P.reshape(-1, 3, 4).cpu().numpy()
    X = np.c_[res["X"], np.ones(len(h))]
    pc = np.einsum("nij,nj->ni", P[h], X) @ Kinv.T
    z = pc[:, 2]
    dok = (d0 < 0.15) | (np.abs(z - d0) < 0.08)
    return ok & dok


def scan_and_associate(max_rms, lag_center=None, half=None):
    import time as _t
    t0 = _t.time()
    assoc, lagseg = {}, {}
    if a.const_lag is not None:
        for s in seqs:
            for j in range(len(s.starts)):
                res = ba.associate(s, j, float(a.const_lag))
                res["lag"] = float(a.const_lag)
                res["ok"] = gated(s, j, res, max_rms)
                assoc[(s.key, s.cam, j)] = res
                lagseg[(s.key, s.cam, j)] = (np.array([a.const_lag]), np.array([0.0]))
        print(f"  associated at constant lag {a.const_lag} ({_t.time() - t0:.0f} s)", flush=True)
        return assoc, lagseg
    for s in seqs:
        for j in range(len(s.starts)):
            if lag_center is None:
                # coarse-to-fine: the side stream stalls / jumps by up to ~70 ticks either way (validated on masks)
                coarse = np.arange(-72, 73, 6) if s.cam == "side" else np.arange(-24, 25, 4)
                sc_c = ba.lag_scan(s, j, coarse)
                c0 = coarse[sc_c.argmin()]
                fine = np.arange(c0 - 4, c0 + 5, 2)
                sc_f = ba.lag_scan(s, j, fine)
                allL = np.r_[coarse, fine]; allS = np.r_[sc_c, sc_f]
                lags, ui = np.unique(allL, return_index=True)
                sc = allS[ui]
            else:
                c0 = lag_center[(s.key, s.cam, j)]
                lags = np.arange(round(c0) - half, round(c0) + half + 1, 1)
                sc = ba.lag_scan(s, j, lags)
            lagseg[(s.key, s.cam, j)] = (lags, sc)
    # smooth the per-segment lag (median of 3 neighbours) per sequence
    for s in seqs:
        L = np.array([lagseg[(s.key, s.cam, j)][0][lagseg[(s.key, s.cam, j)][1].argmin()] for j in range(len(s.starts))], float)
        # flat score curves (robot still in the segment) carry no timing information -> the sequence median
        flat = np.array([np.ptp(lagseg[(s.key, s.cam, j)][1]) < 0.4 for j in range(len(s.starts))])
        med = np.median(L[~flat]) if (~flat).any() else 0.0
        L[flat] = med
        Ls = np.array([np.median(L[max(0, j - 1): j + 2]) for j in range(len(L))])
        for j in range(len(s.starts)):
            res = ba.associate(s, j, float(Ls[j]))
            res["lag"] = float(Ls[j])
            res["ok"] = gated(s, j, res, max_rms)
            assoc[(s.key, s.cam, j)] = res
        print(f"  scan {s.key} {s.cam}: lags {np.round(Ls, 1).tolist()} ({_t.time() - t0:.0f} s)", flush=True)
    return assoc, lagseg


def knots_from(assoc):
    kn = {}
    for s in seqs:
        t = np.array([s.starts[j] + a.life / 2 for j in range(len(s.starts))], float)
        v = np.array([assoc[(s.key, s.cam, j)]["lag"] for j in range(len(s.starts))], float)
        kn[(s.key, s.cam)] = (t, v)
    return kn


def make_obs(assoc, use):
    obs = []
    for s in seqs:
        for j in range(len(s.starts)):
            if not use(s, j):
                continue
            res = assoc[(s.key, s.cam, j)]
            for i in np.flatnonzero(res["ok"]):
                vk = np.flatnonzero(s.vis[j][:, i])
                obs.append(dict(seq=s, j=j, h=int(res["h"][i]), X=res["X"][i], k=vk, uv=s.tracks[j][vk, i], d0=float(s.depth0[j][i])))
    return obs


def heldout_report(assoc):
    """held-out segments: tracks re-triangulated under the current calibration (lag scanned); RMS stats."""
    out = {}
    for cam in ("top", "side"):
        rms, n_ok, n_all = [], 0, 0
        for s in seqs:
            if s.cam != cam:
                continue
            for j in range(len(s.starts)):
                if not is_ho(s, j):
                    continue
                res = assoc[(s.key, s.cam, j)]
                rms.append(np.minimum(res["rms"], 20)); n_ok += int(res["ok"].sum()); n_all += len(res["rms"])
        rr = np.concatenate(rms) if rms else np.zeros(0)
        out[cam] = {"tracks": int(n_all), "inliers": n_ok, "inlier_frac": n_ok / max(1, n_all),
                    "rms_med_capped": float(np.median(rr)) if len(rr) else None,
                    "frac_rms_lt3": float((rr < 3).mean()) if len(rr) else None}
    return out


def corrections():
    out = {}
    for (c, arm_), xi in ba.xi.items():
        T = (ba.T_cb(c, arm_) @ torch.linalg.inv(ba.T0[(c, arm_)])).detach().cpu().numpy()     # camera-frame correction
        ang = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))
        # displacement of the arm base origin seen from the camera, mm
        Tb = ba.T_cb(c, arm_).detach().cpu().numpy(); Tb0 = ba.T0[(c, arm_)].cpu().numpy()
        out[f"{c}/{arm_}"] = {"rot_deg": float(ang), "base_origin_shift_mm": float(np.linalg.norm(Tb[:3, 3] - Tb0[:3, 3]) * 1000)}
    return out


# ------------------------------------------------------------------ silhouette term on SAM robot masks (differentiable render)
import cv2  # noqa: E402
from realpc.diffcalib import Renderer, soft_iou, interp_q  # noqa: E402
arm_lo = arm
SIL = []                                            # (key, cam, tick, mask_left, mask_right, valid_left, valid_right)
QT = {}
REND = {}
for key, ep in eps.items():
    for cam in ("top", "side"):
        p = os.path.join(a.masks_root, key, f"{cam}_masks.npz")
        if not os.path.exists(p):
            continue
        M = np.load(p)["masks"]
        hh, ww = (180, 320) if cam == "top" else (240, 320)
        if (cam, hh) not in REND:
            K = ep.intrinsics(cam).copy(); K[:2] *= ww / M.shape[2]
            REND[cam] = Renderer(K, hh, ww)
        for t in range(0, len(M), a.sil_every):
            m = cv2.resize(M[t], (ww, hh), interpolation=cv2.INTER_NEAREST)
            ml, mr = m == 2, m == 3
            if ml.sum() + mr.sum() < 150:
                continue
            obj = m >= 4
            SIL.append((key, cam, t, ml, mr, ~(obj | mr), ~(obj | ml)))
print("silhouette frames:", len(SIL), flush=True)
life = a.life


def sil_is_fit(item):
    key, cam, t = item[:3]
    j = int(t // 40)                                # segment index of the track files (every=40)
    return key not in a.holdout and j % 4 != 3


SIL_FIT = [x for x in SIL if sil_is_fit(x)]
SIL_HO = [x for x in SIL if not sil_is_fit(x)]
knots_ref = {}


def sil_loss_items(items, hard=False):
    losses, ious = [], []
    for key, cam, t, ml, mr, vl, vr in items:
        ep = eps[key]
        kn = knots_ref.get((key, cam))
        lag = float(np.interp(t, *kn)) if kn is not None else 0.0
        R = REND[cam]
        for arm_, m, v in (("left", ml, vl), ("right", mr, vr)):
            if (key, arm_) not in QT:
                QT[(key, arm_)] = torch.tensor(ep.low[f"joint_pos_{arm_}"], dtype=torch.float64, device="cuda")
            q = interp_q(QT[(key, arm_)], torch.tensor([t - lag], dtype=torch.float64, device="cuda"))
            V = arm_lo.verts(q, ba.joff[arm_] if a.offsets else None)
            M4 = ba.T_cb(cam, arm_)
            Vc = (V @ M4[:3, :3].T + M4[:3, 3]).float()
            sil, _ = R.raster(Vc, arm_lo.faces, sigma=None if hard else 3e-4, fpp=1 if hard else 8)
            mt = torch.tensor(m, dtype=torch.float32, device="cuda")[None]
            vt = torch.tensor(v, dtype=torch.float32, device="cuda")[None]
            if mt.sum() < 60 and sil.sum() < 60:
                continue
            iou = soft_iou(sil, mt, vt)[0]
            losses.append(1 - iou); ious.append(float(iou))
    return losses, ious


rng_sil = np.random.default_rng(0)


def extra(it):
    if a.sil_w <= 0 or not SIL_FIT or it % 2:
        return 0.0
    idx = rng_sil.choice(len(SIL_FIT), min(4, len(SIL_FIT)), replace=False)
    ls, _ = sil_loss_items([SIL_FIT[i] for i in idx])
    return a.sil_w * torch.stack(ls).mean() if ls else 0.0


@torch.no_grad()
def sil_report(items):
    out = {}
    for cam in ("top", "side"):
        it_ = [x for x in items if x[1] == cam]
        _, ious = sil_loss_items(it_, hard=True)
        out[cam] = {"frames": len(it_), "iou_med": float(np.median(ious)) if ious else None}
    return out


log = {}
assoc, lagseg = scan_and_associate(max_rms=6.0)
lagseg0 = lagseg
log["heldout_recorder"] = heldout_report(assoc)
print("held-out (recorder):", json.dumps(log["heldout_recorder"]), flush=True)
knots = knots_from(assoc)
knots_ref.update(knots)
log["sil_heldout_recorder_lag"] = sil_report(SIL_HO)
print("held-out silhouette IoU (recorder calib, scanned lag):", log["sil_heldout_recorder_lag"], flush=True)
for rd in range(a.rounds):
    obs = make_obs(assoc, is_fit)
    print(f"round {rd}: {len(obs)} fit tracks, {sum(len(o['k']) for o in obs)} observations", flush=True)
    X, knots, hist = ba.refine(seqs, obs, knots, iters=a.iters, fit_offsets=(a.offsets and rd == a.rounds - 1),
                               depth_w=a.depth_w, depth_sigma=a.depth_sigma, extra_loss=extra,
                               fit_lag=a.const_lag is None)
    knots_ref.update(knots)
    print("corrections:", json.dumps(corrections()), flush=True)
    lag_c = {(s.key, s.cam, j): float(np.interp(s.starts[j] + a.life / 2, *knots[(s.key, s.cam)])) for s in seqs for j in range(len(s.starts))}
    assoc, lagseg = scan_and_associate(max_rms=max(3.0, 6.0 - 1.5 * (rd + 1)), lag_center=lag_c, half=6)
    knots = knots_from(assoc)
    knots_ref.update(knots)
    log[f"heldout_round{rd}"] = heldout_report(assoc)
    log[f"sil_heldout_round{rd}"] = sil_report(SIL_HO)
    print(f"held-out silhouette IoU (round {rd}):", log[f"sil_heldout_round{rd}"], flush=True)
    print(f"held-out (round {rd}):", json.dumps(log[f"heldout_round{rd}"]), flush=True)

# --------------------------------------------------------------------------------------------------- save
Tcb = {k: ba.T_cb(*k).detach().cpu().numpy() for k in ba.xi}
T_lb_cam = {c: np.linalg.inv(Tcb[(c, "left")]) for c in ("top", "side")}
T_lb_rb_per_cam = {c: T_lb_cam[c] @ Tcb[(c, "right")] for c in ("top", "side")}
lags_out = {f"{k[0]}:{k[1]}": {"knot_ticks": v[0].tolist(), "lag_ticks": v[1].tolist()} for k, v in knots.items()}
lag_curves = {f"{k[0]}:{k[1]}:{k[2]}": {"lags": v[0].tolist(), "score_px": np.round(v[1], 3).tolist()} for k, v in lagseg0.items()}
out = {"version": "v3-tracks", "created": datetime.datetime.now().isoformat(timespec="seconds"),
       "method": "bundle adjustment of image-seeded CoTracker3 tracks through the YAM FK; per camera AND per arm "
                 "T_cam<-base, per (episode,camera) piecewise lag, " + ("joint zero offsets, " if a.offsets else "") +
                 "Huber reprojection; association by DLT triangulation over all (arm, link) hypotheses with depth gates",
       "corrections": corrections(), "log": log,
       "T_cam_base": {f"{c}/{arm_}": Tcb[(c, arm_)].tolist() for (c, arm_) in Tcb},
       "T_lb_top": T_lb_cam["top"].tolist(), "T_lb_side": T_lb_cam["side"].tolist(),
       "T_lb_rb_from_top": T_lb_rb_per_cam["top"].tolist(), "T_lb_rb_from_side": T_lb_rb_per_cam["side"].tolist(),
       "joint_offsets_rad": {k: v.detach().cpu().numpy().tolist() for k, v in ba.joff.items()},
       "const_lag": a.const_lag, "lags": lags_out, "lag_scan_recorder_calib": lag_curves, "holdout_episodes": a.holdout,
       "recorder": {"T_lb_top": np.linalg.inv(T0[("top", "left")]).tolist(), "T_lb_side": np.linalg.inv(T0[("side", "left")]).tolist(),
                    "T_lb_rb": np.asarray(ep0.calib["T_lb_rb"]).tolist()}}
json.dump(out, open(a.out, "w"), indent=1)
print("saved", a.out)
