#!/usr/bin/env python
"""Rig calibration v2 on stationary-robot frames (realpc/calib_v2.py). Writes configs/calib_yam_rig_v2.json with the
held-out report, and before/after FK overlay stills.

  python scripts/calibrate_v2.py --episodes <ep> ... --out configs/calib_yam_rig_v2.json --overlay_dir <dir>
"""
import argparse
import datetime
import json
import os
import sys

import cv2
import numpy as np
from scipy.spatial import cKDTree

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from realpc.calib_v2 import RigCalibrator, backproject, stationary_frames  # noqa: E402
from realpc.config import build_arms, load_config  # noqa: E402
from realpc.io import DepthSource, Episode  # noqa: E402
from realpc.robot_overlay import MeshProjector, depth_residual, draw, edge_distance  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", nargs="+", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--overlay_dir", required=True); ap.add_argument("--depth_cache", default=os.path.join(ROOT, ".cache"))
ap.add_argument("--config", default=os.path.join(ROOT, "configs", "yam_bimanual.json"))
ap.add_argument("--still", type=int, default=10); ap.add_argument("--max_per_ep", type=int, default=14)
ap.add_argument("--thr", type=float, default=0.01); ap.add_argument("--min_sep", type=float, default=0.12)
a = ap.parse_args()
cfg = load_config(a.config)
eps = [Episode(p) for p in a.episodes]
arms = build_arms(cfg, eps[0].calib)
proj = MeshProjector(arms.robot, max_faces=3000)
cams = ("top", "side")
fit_frames, ho_frames, fit_static, ho_static, used = [], [], [], [], []
Ld, Bd = arms.dense_local()
for ep in eps:
    picked, cand = stationary_frames(ep.low, still=a.still, thr=a.thr, min_sep=a.min_sep)
    picked = picked[: a.max_per_ep] if len(picked) <= a.max_per_ep else [picked[i] for i in np.linspace(0, len(picked) - 1, a.max_per_ep).astype(int)]
    srcs = {c: DepthSource(ep, c, a.depth_cache) for c in cams}
    caps = {c: cv2.VideoCapture(ep.video(c)) for c in cams}
    print(ep.task, "stationary ticks", len(cand), "picked", len(picked), picked, flush=True)
    for j, t in enumerate(picked):
        hold = (j % 5 == 4)
        q = {arm: ep.low[f"joint_pos_{arm}"][t] for arm in arms.T_base}
        LP = arms.link_poses(q)
        Rw = np.einsum("nij,nj->ni", LP[Bd, :3, :3], Ld) + LP[Bd, :3, 3]
        st = {"robot_w": Rw}
        for c in cams:
            caps[c].set(cv2.CAP_PROP_POS_FRAMES, int(t)); ok, img = caps[c].read()
            D = srcs[c].depth(int(t))
            fr = dict(cam=c, img=img, depth=D, K=ep.intrinsics(c), q=q, ep=f"{ep.task}/{ep.name}", tick=int(t))
            (ho_frames if hold else fit_frames).append(fr)
            st[c] = (D, ep.intrinsics(c))
        (ho_static if hold else fit_static).append(st)
        used.append(f"{ep.task}/{ep.name}:{t}{' (held-out)' if hold else ''}")
T_wc0 = {c: np.asarray(eps[0].low[f"cam_pose_{c}"][0], float) for c in cams}
T_rb0 = np.asarray(eps[0].calib["T_lb_rb"], float)
cal = RigCalibrator(arms, proj, cams, T_wc0, T_rb0, *eps[0].table_plane())
x, log = cal.solve(fit_frames, fit_static)


def evaluate(x, frames, statics):
    Twc, Tb = cal.poses(x)
    res = {c: {"edge_px": [], "arm_dz_mm": [], "arm_within2cm": []} for c in cams}
    for f in frames:
        c = f["cam"]; H, W = f["img"].shape[:2]
        rd = proj.render({arm: arms.qmap(f["q"][arm]) for arm in Tb}, Tb, Twc[c], f["K"], H, W)
        for arm, (m, dep) in rd.items():
            if m.sum() < 300:
                continue
            res[c]["edge_px"].append(edge_distance(f["img"], m)[0])
            dm, dw, nd = depth_residual(f["depth"], dep, m)
            if np.isfinite(dm):
                res[c]["arm_dz_mm"].append(dm * 1000); res[c]["arm_within2cm"].append(dw)
    cross = []
    for s in statics:
        Pt = backproject(*s["top"]); Ps = backproject(*s["side"])
        Ptw = Pt @ Twc["top"][:3, :3].T + Twc["top"][:3, 3]; Psw = Ps @ Twc["side"][:3, :3].T + Twc["side"][:3, 3]
        rt = cKDTree(s["robot_w"])
        Ptw = Ptw[~np.isfinite(rt.query(Ptw, distance_upper_bound=0.06)[0])]
        Psw = Psw[~np.isfinite(rt.query(Psw, distance_upper_bound=0.06)[0])]
        d = cKDTree(Ptw).query(Psw, distance_upper_bound=0.05)[0]
        cross.append(d[np.isfinite(d)])
    cr = np.concatenate(cross) * 1000 if cross else np.zeros(0)
    out = {c: {k: (float(np.nanmedian(v)) if len(v) else None) for k, v in d_.items()} for c, d_ in res.items()}
    out["top_vs_side_static_mm"] = {"median": float(np.median(cr)) if len(cr) else None, "p90": float(np.percentile(cr, 90)) if len(cr) else None}
    return out


x0 = np.zeros(18)
before, after = evaluate(x0, ho_frames, ho_static), evaluate(x, ho_frames, ho_static)
corr = {}
for i, name in enumerate(("top", "side", "right_base")):
    v = x[6 * i: 6 * i + 6]
    corr[name] = {"rot_deg": float(np.degrees(np.linalg.norm(v[:3]))), "trans_mm": float(np.linalg.norm(v[3:]) * 1000)}
plausible = all(c["rot_deg"] < 2.5 and c["trans_mm"] < 25 for c in corr.values())
better = []
for c in cams:
    b, a_ = before[c], after[c]
    better.append((a_["edge_px"] or 99) <= (b["edge_px"] or 99) + 0.3)
    if b["arm_dz_mm"] is not None and a_["arm_dz_mm"] is not None:
        better.append(a_["arm_dz_mm"] <= b["arm_dz_mm"] + 1)
better.append((after["top_vs_side_static_mm"]["median"] or 99) <= (before["top_vs_side_static_mm"]["median"] or 99))
adopt = bool(plausible and all(better))
Twc, Tb = cal.poses(x)
out = {"version": "v2", "created": datetime.datetime.now().isoformat(timespec="seconds"),
       "method": "joint robust LM on stationary-robot frames (joint speed < thr rad/tick for >= %d ticks before/after): FK silhouette vs Canny edges, "
                 "FK surface vs arm depth, top-vs-side static point-to-plane, table plane; weak prior to the recorder calib" % a.still,
       "adopted": adopt, "plausible": plausible, "heldout_improved": bool(all(better)),
       "corrections": corr, "heldout_before": before, "heldout_after": after, "fit_log": log, "frames": used,
       "T_lb_top": Twc["top"].tolist(), "T_lb_side": Twc["side"].tolist(), "T_lb_rb": Tb["right"].tolist(),
       "recorder": {"T_lb_top": T_wc0["top"].tolist(), "T_lb_side": T_wc0["side"].tolist(), "T_lb_rb": T_rb0.tolist()}}
json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps({k: out[k] for k in ("adopted", "plausible", "heldout_improved", "corrections", "heldout_before", "heldout_after")}, indent=1))
# overlay stills: held-out frames, before | after, per camera
os.makedirs(a.overlay_dir, exist_ok=True)
for c in cams:
    rows = []
    for f in [f for f in ho_frames if f["cam"] == c][:6]:
        H, W = f["img"].shape[:2]
        tiles = []
        for lab, xx in (("recorder", x0), ("v2", x)):
            Twc_, Tb_ = cal.poses(xx)
            rd = proj.render({arm: arms.qmap(f["q"][arm]) for arm in Tb_}, Tb_, Twc_[c], f["K"], H, W)
            im = draw(f["img"], rd)
            cv2.putText(im, f"{lab}  {f['ep']}:{f['tick']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            tiles.append(cv2.resize(im, (960, 540)))
        rows.append(np.concatenate(tiles, 1))
    if rows:
        cv2.imwrite(os.path.join(a.overlay_dir, f"calib_v2_{c}_before_after.png"), np.concatenate(rows, 0))
print("saved", a.out)
