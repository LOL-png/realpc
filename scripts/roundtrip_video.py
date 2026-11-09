#!/usr/bin/env python
"""SE(3)-patch encoder round-trip video for one world-model window: GT tracks | analytic decode | learned decode |
per-point error (learned), oblique virtual view, coloured by group (gripper / robot / static / moving).

  PLWM_SE3_SRC=<dir> python scripts/roundtrip_video.py --tracks tracks.npz --t0 <step> --out rt.mp4 [--ckpt ae.pt]
      [--n_points 4096 --n_robot 26 --n_scene 70]
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from realpc.pipeline import Tracks  # noqa: E402
from realpc.se3_roundtrip import _load_plwm, build_analytic, build_learned, errors, make_window, roundtrip  # noqa: E402
from realpc.viz import FFWriter, splat, virtual_cam  # noqa: E402


def draw_points(img, P, cols, T, K, r=1):
    return splat(img, P, cols, T, K, r=r)

COL = {0: (0, 220, 255), 1: (0, 140, 255), 2: (150, 150, 150), 3: (255, 200, 0), 4: (255, 0, 255)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--t0", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n_points", type=int, default=4096)
    ap.add_argument("--n_robot", type=int, default=26)
    ap.add_argument("--n_scene", type=int, default=70)
    ap.add_argument("--title", default="")
    ap.add_argument("--fps", type=int, default=8)
    a = ap.parse_args()
    sp, sl = _load_plwm()
    tr = Tracks.load(a.tracks)
    b, info = make_window(tr, a.t0, n_points=a.n_points, seed=a.t0)
    decs = [("analytic (Kabsch, no residual)", build_analytic(sp, a.n_robot, a.n_scene))]
    if a.ckpt:
        decs.append(("learned (zero-shot)", build_learned(sp, sl, a.ckpt)))
    outs = []
    for nm, ae in decs:
        dec, seg, _ = roundtrip(ae, b)
        e, E = errors(dec, info, seg)
        outs.append((nm, info["A"][:, None] + dec, E, e))
    X = np.concatenate([info["X"][:, :16], info["A"][:, None], info["X"][:, 16:]], 1)        # GT incl. t0
    eg = info["eval_group"]
    cols = np.array([COL[g] for g in eg], np.uint8)
    focus = np.nanmedian(X[eg >= 2, 16], 0) if (eg >= 3).sum() < 20 else np.nanmedian(X[eg >= 3, 16], 0)
    Tv, Kv = virtual_cam(focus, dist=0.9, elev=45, azim=200, f=650, W=480, H=400)
    wr = None
    for s in range(X.shape[1]):
        panels = []
        gt = np.full((400, 480, 3), 20, np.uint8)
        ok = np.isfinite(X[:, s]).all(1)
        draw_points(gt, X[ok, s], cols[ok], Tv, Kv, r=1)
        cv2.putText(gt, "GT tracks", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(gt)
        for nm, D, E, e in outs:
            Dk = np.concatenate([D[:, :16], info["A"][:, None], D[:, 16:]], 1)
            im = np.full((400, 480, 3), 20, np.uint8)
            draw_points(im, Dk[ok, s], cols[ok], Tv, Kv, r=1)
            Ek = np.concatenate([E[:, :16], np.zeros((len(E), 1)), E[:, 16:]], 1)[:, s]
            txt = f"{nm}  moving {np.nanmean(Ek[eg == 3]) * 1000:.1f} mm  static {np.nanmean(Ek[eg == 2]) * 1000:.1f} mm"
            cv2.putText(im, txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            panels.append(im)
        nm, D, E, e = outs[-1]
        Ek = np.concatenate([E[:, :16], np.zeros((len(E), 1)), E[:, 16:]], 1)[:, s]
        em = np.full((400, 480, 3), 20, np.uint8)
        ec = cv2.applyColorMap(np.clip(np.nan_to_num(Ek) / 0.02 * 255, 0, 255).astype(np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0]
        draw_points(em, X[ok, s], ec[ok], Tv, Kv, r=1)
        cv2.putText(em, f"error of '{nm}' on GT (0..20 mm)", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(em)
        frame = np.concatenate(panels, 1)
        bar = np.zeros((30, frame.shape[1], 3), np.uint8)
        cv2.putText(bar, f"{a.title}  step {s - 16:+d} (t0 = 0)   colours: yellow gripper, orange robot, grey static, blue moving",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        frame = np.concatenate([bar, frame], 0)
        if wr is None:
            wr = FFWriter(a.out, frame.shape[1], frame.shape[0], a.fps)
        wr.write(frame)
    wr.close()
    print("saved", a.out)


if __name__ == "__main__":
    main()
