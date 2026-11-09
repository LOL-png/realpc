"""Camera stream synchronisation by content. The recorder writes one frame per tick for every camera, but the side
stream's content drifts against the top stream by up to ~1-2 s within an episode (seen on greycloth). We estimate a
monotonic map side_frame(top_tick) by dynamic time warping of the two cameras' motion-energy signals (Sakoe-Chiba band).
The top camera defines time (the fused clouds and q are stamped with its capture time)."""
from __future__ import annotations

import os

import cv2
import numpy as np


def motion_energy(path, scale=0.125):
    c = cv2.VideoCapture(path); prev = None; m = []
    while True:
        ok, im = c.read()
        if not ok:
            break
        g = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), None, fx=scale, fy=scale).astype(np.float32)
        m.append(0.0 if prev is None else float(np.abs(g - prev).mean())); prev = g
    m = np.array(m)
    rep = m < 0.05
    if rep.any() and (~rep).sum() > 2:
        m[rep] = np.interp(np.flatnonzero(rep), np.flatnonzero(~rep), m[~rep])
    m = np.convolve(m, np.ones(5) / 5, mode="same")
    return (m - m.mean()) / (m.std() + 1e-9)


def dtw_map(a, b, band=90, slope_pen=0.2):
    """map index of a (reference) -> index of b, monotonic, |i - j| <= band. Cost |a_i - b_j| + slope penalty."""
    n, m = len(a), len(b)
    INF = 1e18
    D = np.full((n + 1, m + 1), INF); D[0, 0] = 0
    P = np.zeros((n + 1, m + 1), np.int8)
    for i in range(1, n + 1):
        lo, hi = max(1, i - band), min(m, i + band)
        for j in range(lo, hi + 1):
            c = abs(a[i - 1] - b[j - 1])
            opts = (D[i - 1, j - 1], D[i - 1, j] + slope_pen, D[i, j - 1] + slope_pen)
            k = int(np.argmin(opts)); D[i, j] = c + opts[k]; P[i, j] = k
    i, j = n, m
    mp = np.zeros(n, int)
    while i > 0 and j > 0:
        mp[i - 1] = j - 1
        k = P[i, j]
        if k == 0:
            i, j = i - 1, j - 1
        elif k == 1:
            i -= 1
        else:
            j -= 1
    return np.maximum.accumulate(mp)


def side_frame_map(ep, cache_dir=None, band=90):
    """top tick -> side frame index (cached)."""
    p = os.path.join(cache_dir, f"{ep.task}_{ep.name}_side_sync.npy") if cache_dir else None
    if p and os.path.exists(p):
        return np.load(p)
    a = motion_energy(ep.video("top")); b = motion_energy(ep.video("side"))
    mp = dtw_map(a, b, band)
    mp = np.round(np.convolve(np.pad(mp.astype(float), 7, mode="edge"), np.ones(15) / 15, mode="valid")).astype(int)
    if p:
        np.save(p, mp)
    return mp
