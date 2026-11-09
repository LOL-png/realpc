"""SE(3)-patch encoder round trip on converted tracks (optional dependency: the plwm SE(3)-patch AE).

The encoder (plwm/ae/se3patch.py, se3latent.py) is found via `import plwm.ae.se3patch` or, if plwm is not installed,
from the folder in the environment variable PLWM_SE3_SRC holding se3patch.py and se3latent.py (copied from a plwm tag).

Windows follow the world model: n_hist past steps, the t0 step, n_fut future steps (default 16 + 1 + 32). Points = those
alive at t0; a step is valid where the track exists. Variants:
  analytic          FPS patches + per-patch Kabsch (rotation + translation per step), residual OFF (no learned part)
  analytic_oracle   analytic + the true per-point residual clipped to the residual cap (+-res_scale per axis):
                    the best a residual channel of that size could do
  learned           a trained checkpoint (SE3 base AE or SE3-latent adapter + base), zero-shot, encode -> decode
Per group: gripper / robot / static scene / moving scene (cloth on cloth scenes); blending = static points whose token
also holds moving points.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import torch


def _load_plwm():
    try:
        from plwm.ae import se3latent, se3patch  # type: ignore
        return se3patch, se3latent
    except ImportError:
        src = os.environ.get("PLWM_SE3_SRC")
        if not src:
            raise ImportError("plwm not installed and PLWM_SE3_SRC not set (folder with se3patch.py / se3latent.py)")
        pkg = "_plwm_se3"
        if pkg not in sys.modules:
            spec = importlib.util.spec_from_file_location(pkg, os.path.join(src, "__init__.py"), submodule_search_locations=[src])
            mod = importlib.util.module_from_spec(spec)
            sys.modules[pkg] = mod
            spec.loader.exec_module(mod)
        import importlib as il
        return il.import_module(pkg + ".se3patch"), il.import_module(pkg + ".se3latent")


def make_window(tracks, k0, n_hist=16, n_fut=32, n_points=4096, robot_frac=0.25, seed=0, moved_thresh=0.01):
    """k0 = t0 step index in the tracks. Returns (batch dict for the AE, info dict)."""
    rng = np.random.default_rng(seed)
    T = tracks.pos.shape[1]
    assert k0 - n_hist >= 0 and k0 + n_fut < T, "window outside the tracks"
    steps = np.r_[np.arange(k0 - n_hist, k0), np.arange(k0 + 1, k0 + n_fut + 1)]
    a_all = tracks.pos[:, k0]
    alive = np.isfinite(a_all).all(1)
    rob = np.flatnonzero(alive[: tracks.n_robot])
    sce = tracks.n_robot + np.flatnonzero(alive[tracks.n_robot:])
    nr = min(len(rob), int(round(n_points * robot_frac)))
    ns = min(len(sce), n_points - nr)
    idx = np.r_[rng.choice(rob, nr, replace=False), rng.choice(sce, ns, replace=False)]
    A = a_all[idx].astype(np.float64)
    X = tracks.pos[idx][:, steps].astype(np.float64)                       # [N, 48, 3]
    sv = np.isfinite(X).all(-1)
    D = np.where(sv[..., None], X - A[:, None], 0.0)
    grp = tracks.group[idx].astype(int)
    bg = np.where(grp == 0, 0, np.where(grp == 1, 1, 2))
    body = tracks.body[idx].astype(int)
    # eval groups (GT motion within the window)
    full = np.concatenate([X[:, :n_hist], A[:, None], X[:, n_hist:]], 1)
    disp = np.nanmax(np.linalg.norm(full - A[:, None], axis=-1), axis=1)
    eg = np.where(bg == 0, 0, np.where(bg == 1, 1, np.where(np.nan_to_num(disp) > moved_thresh, 3, 2)))
    eg = np.where(grp == 4, np.where(np.nan_to_num(disp) > moved_thresh, 4, 2), eg)          # cloth (moving) as its own group
    b = {"anchors": torch.tensor(A, dtype=torch.float32)[None], "delta": torch.tensor(D, dtype=torch.float32)[None],
         "step_valid": torch.tensor(sv)[None], "valid": torch.ones(1, len(idx), dtype=torch.bool),
         "body_group": torch.tensor(bg)[None], "body_id": torch.tensor(body)[None]}
    return b, {"idx": idx, "eval_group": eg, "steps": steps, "X": X, "A": A, "sv": sv}


def build_analytic(se3patch, n_robot, n_scene, n_fut=32):
    cfg = se3patch.SE3PatchConfig(n_robot=n_robot, n_scene=n_scene, res_off=True, n_future_steps=n_fut)
    return se3patch.SE3PatchAE(cfg).eval()


def build_learned(se3patch, se3latent, ckpt_path, base_only=False):
    p = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if p["ae_config"].get("kind") == "se3latent":
        ae = se3latent.se3latent_from_payload(p)
        return ae.base if base_only else ae
    return se3patch.se3_from_payload(p)


@torch.no_grad()
def roundtrip(ae, b, device="cuda", oracle_res=None):
    ae = ae.to(device)
    bb = {k: v.to(device) for k, v in b.items()}
    enc = ae.encode(bb)
    out = ae.decode_batch(enc["mu"], enc, bb)
    if oracle_res is not None:                                            # analytic + true residual clipped to the cap
        gt = bb["delta"]
        r = (gt - out).clamp(-oracle_res, oracle_res)
        out = out + r * bb["step_valid"][..., None].float()
    seg = enc["seg"][0].cpu().numpy()
    return out[0].cpu().numpy(), seg, enc


def errors(dec, info, seg, n_hist=16):
    """per-point per-step error (m) and summaries per eval group."""
    X, A, sv = info["X"], info["A"], info["sv"]
    E = np.linalg.norm(A[:, None] + dec - X, axis=-1)                     # [N, 48]
    E[~sv] = np.nan
    eg = info["eval_group"]
    names = {0: "gripper", 1: "robot", 2: "static", 3: "moving", 4: "cloth"}
    out = {}
    # blending: static points whose token holds >= 1 moving point
    mov_tok = set(seg[(eg >= 3) & (seg >= 0)].tolist())
    mixed = np.array([s in mov_tok for s in seg])
    for gi, nm in names.items():
        m = eg == gi
        if not m.any():
            continue
        e = E[m] * 1000
        per_step_mean = np.nanmean(e, axis=0)
        out[nm] = {"n": int(m.sum()), "mean_mm": float(np.nanmean(e)), "p90_mm": float(np.nanpercentile(e, 90)),
                   "max_mm": float(np.nanmax(e)), "future_mean_mm": float(np.nanmean(e[:, n_hist:])),
                   "per_step_mean_mm": [float(x) for x in per_step_mean]}
    m = (eg == 2) & mixed
    if m.any():
        out["static_in_moving_tokens"] = {"n": int(m.sum()), "mean_mm": float(np.nanmean(E[m]) * 1000), "p90_mm": float(np.nanpercentile(E[m] * 1000, 90))}
    m = (eg == 2) & ~mixed
    if m.any():
        out["static_pure_tokens"] = {"n": int(m.sum()), "mean_mm": float(np.nanmean(E[m]) * 1000)}
    return out, E
