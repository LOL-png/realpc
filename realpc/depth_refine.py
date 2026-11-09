"""Per-frame colour-guided depth refinement (causal: current frame only).

Vendored from the user's reconstruction harness (scripts/refine_lib.py, stage 1 "per-frame refinement"), unchanged
except for the device selection and this header. Solves a confidence-weighted, RGB-edge-aware least-squares problem in
inverse depth (coarse-to-fine PCG); RealSense readings near depth discontinuities get low confidence.
Entry point: refine_frame(rgb_u8 HxWx3, raw_u16 HxW, depth_scale) -> depth (torch HxW, metres).
"""
import math
import numpy as np
import torch
import torch.nn.functional as F

DEV = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')


def to_t(a, dtype=torch.float32):
    return torch.from_numpy(np.ascontiguousarray(a)).to(DEV, dtype)


def gauss_blur(x, sigma):
    """x: (C,H,W) tensor."""
    if sigma <= 0:
        return x
    r = int(math.ceil(3 * sigma))
    k = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (k / sigma) ** 2); k = k / k.sum()
    C = x.shape[0]
    xx = x[None]
    xx = F.conv2d(F.pad(xx, (r, r, 0, 0), mode='replicate'), k.view(1, 1, 1, -1).repeat(C, 1, 1, 1), groups=C)
    xx = F.conv2d(F.pad(xx, (0, 0, r, r), mode='replicate'), k.view(1, 1, -1, 1).repeat(C, 1, 1, 1), groups=C)
    return xx[0]


def color_affinity(guide, sigma_c, floor=1e-3):
    """guide: (3,H,W) float in 0..255. Returns wx (H,W-1), wy (H-1,W), wd1, wd2 (H-1,W-1) diagonals."""
    dx = ((guide[:, :, 1:] - guide[:, :, :-1]) ** 2).sum(0)
    dy = ((guide[:, 1:, :] - guide[:, :-1, :]) ** 2).sum(0)
    d1 = ((guide[:, 1:, 1:] - guide[:, :-1, :-1]) ** 2).sum(0)
    d2 = ((guide[:, 1:, :-1] - guide[:, :-1, 1:]) ** 2).sum(0)
    s2 = 2 * sigma_c ** 2
    f = lambda d: torch.exp(-d / s2) + floor
    return f(dx), f(dy), f(d1) * 0.5, f(d2) * 0.5


def field_affinity(x, sigma):
    """Affinity from a scalar field (e.g. a previous inverse-depth solution)."""
    s2 = 2 * sigma ** 2
    ex = torch.exp(-((x[:, 1:] - x[:, :-1]) ** 2) / s2)
    ey = torch.exp(-((x[1:, :] - x[:-1, :]) ** 2) / s2)
    e1 = torch.exp(-((x[1:, 1:] - x[:-1, :-1]) ** 2) / s2)
    e2 = torch.exp(-((x[1:, :-1] - x[:-1, 1:]) ** 2) / s2)
    return ex, ey, e1, e2


def lap_apply(x, W):
    """Weighted graph Laplacian L x (8-neighbourhood)."""
    wx, wy, w1, w2 = W
    out = torch.zeros_like(x)
    t = wx * (x[:, 1:] - x[:, :-1]); out[:, 1:] += t; out[:, :-1] -= t
    t = wy * (x[1:, :] - x[:-1, :]); out[1:, :] += t; out[:-1, :] -= t
    t = w1 * (x[1:, 1:] - x[:-1, :-1]); out[1:, 1:] += t; out[:-1, :-1] -= t
    t = w2 * (x[1:, :-1] - x[:-1, 1:]); out[1:, :-1] += t; out[:-1, 1:] -= t
    return out


def lap_diag(W, shape):
    wx, wy, w1, w2 = W
    d = torch.zeros(shape, device=wx.device, dtype=wx.dtype)
    d[:, 1:] += wx; d[:, :-1] += wx
    d[1:, :] += wy; d[:-1, :] += wy
    d[1:, 1:] += w1; d[:-1, :-1] += w1
    d[1:, :-1] += w2; d[:-1, 1:] += w2
    return d


def cg_solve(c, b, W, lam, x0, iters):
    """Solve (diag(c) + lam L) x = b with Jacobi-PCG."""
    Minv = 1.0 / (c + lam * lap_diag(W, c.shape) + 1e-12)
    A = lambda v: c * v + lam * lap_apply(v, W)
    x = x0.clone()
    r = b - A(x)
    z = Minv * r
    p = z.clone()
    rz = (r * z).sum()
    for _ in range(iters):
        Ap = A(p)
        alpha = rz / ((p * Ap).sum() + 1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        z = Minv * r
        rz_new = (r * z).sum()
        p = z + (rz_new / (rz + 1e-30)) * p
        rz = rz_new
    return x


def down2(x):
    """area downsample by 2 of (C,H,W) or (H,W), handling odd sizes by padding."""
    squeeze = x.dim() == 2
    if squeeze:
        x = x[None]
    H, W = x.shape[-2:]
    x = F.pad(x[None], (0, W % 2, 0, H % 2), mode='replicate')[0]
    x = F.avg_pool2d(x[None], 2)[0]
    return x[0] if squeeze else x


def up_to(x, shape):
    return F.interpolate(x[None, None], size=shape, mode='bilinear', align_corners=False)[0, 0]


def solve_pyramid(c, y, guide, lam, sigma_c, extra_aff=None, levels=4, iters=(60, 60, 80, 120), x_init=None,
                  floor=1e-3):
    """Coarse-to-fine confidence-weighted WLS.  c,y: (H,W); guide (3,H,W).
    extra_aff: optional (H,W) field whose affinity multiplies the colour affinity."""
    cs, bs, gs, es = [c], [c * y], [guide], [extra_aff]
    for _ in range(levels - 1):
        cs.append(down2(cs[-1])); bs.append(down2(bs[-1])); gs.append(down2(gs[-1]))
        es.append(down2(es[-1]) if extra_aff is not None else None)
    x = None
    for lv in range(levels - 1, -1, -1):
        cl, bl, gl, el = cs[lv], bs[lv], gs[lv], es[lv]
        W = color_affinity(gl, sigma_c, floor)
        if el is not None:
            E = field_affinity(el, extra_sigma(el))
            W = tuple(a * e + floor * 0.1 for a, e in zip(W, E))
        lam_l = lam / (4 ** lv)
        if x is None:
            if x_init is not None:
                x = F.interpolate(x_init[None, None], size=cl.shape, mode='area')[0, 0]
            else:
                yy = bl / cl.clamp_min(1e-6)
                x = torch.where(cl > 1e-6, yy, (bl.sum() / cl.sum().clamp_min(1e-6)).expand_as(yy))
        else:
            x = up_to(x, cl.shape)
        x = cg_solve(cl, bl, W, lam_l, x, iters[min(lv, len(iters) - 1)] if lv < len(iters) else iters[-1])
    return x


_EXTRA_SIGMA = [0.02]


def extra_sigma(_):
    return _EXTRA_SIGMA[0]


# ----------------------------------------------------------------- confidence

def box_norm(x, w, r):
    """Normalised box filter of x with weights w (radius r)."""
    k = 2 * r + 1
    num = F.avg_pool2d((x * w)[None, None], k, 1, r, count_include_pad=False)[0, 0]
    den = F.avg_pool2d(w[None, None], k, 1, r, count_include_pad=False)[0, 0]
    return num / den.clamp_min(1e-6), den


def raw_confidence(inv, valid, band_px=8, grad_rel=0.0045, hole_px=5, weight=None):
    """inv: (H,W) inverse depth (1/m) (0 where invalid); valid: bool (H,W).
    Confidence = 1 (or `weight`) on clean measured pixels; strongly reduced
    inside a band around measured depth discontinuities (RealSense smears
    depth across ~10 px there, so these readings mix foreground and
    background) and near missing measurements; 0 where there is no reading.

    Discontinuities are detected from the relative inverse-depth gradient over
    a 4 px baseline after a 7x7 normalised box filter, which separates real
    steps (>= ~0.5 %/px) from the slanted table (<= ~0.2 %/px) and noise."""
    vf = valid.float()
    s, den = box_norm(inv, vf, 3)
    s = torch.where(den > 0.3, s, torch.zeros_like(s))
    ok = den > 0.3
    gx = torch.zeros_like(s); gy = torch.zeros_like(s)
    gx[:, 2:-2] = (s[:, 4:] - s[:, :-4]).abs() / 4
    gy[2:-2, :] = (s[4:, :] - s[:-4, :]).abs() / 4
    okx = torch.zeros_like(ok); okx[:, 2:-2] = ok[:, 4:] & ok[:, :-4]
    oky = torch.zeros_like(ok); oky[2:-2, :] = ok[4:, :] & ok[:-4, :]
    g = torch.maximum(gx * okx, gy * oky) / s.clamp_min(1e-3)
    edge = (g > grad_rel) & ok
    kb = 2 * band_px + 1
    band = (F.max_pool2d(edge.float()[None, None], kb, 1, band_px)[0, 0] > 0) & valid
    kh = 2 * hole_px + 1
    near_hole = (F.max_pool2d((1 - vf)[None, None], kh, 1, hole_px)[0, 0] > 0) & valid
    c = vf.clone() if weight is None else (vf * weight)
    c = torch.where(near_hole, c * 0.1, c)
    c = torch.where(band, c * 0.02, c)
    return c, band, near_hole


def refine_frame(rgb_u8, raw_u16, depth_scale, return_debug=False, **kw):
    """Single-frame refinement of a raw z16 frame. Returns depth (H,W) metres (torch)."""
    z = to_t(raw_u16) * depth_scale
    return refine(rgb_u8, z, None, return_debug=return_debug, **kw)


def refine(rgb_u8, z, weight=None, lam1=30.0, lam2=300.0, sigma_c=7.0, sigma_rel2=0.006,
           band_px=8, grad_rel=0.0045, levels=4, band_c=0.02, snap_iters=0, want_support=False, min_component=0,
           return_debug=False):
    """z: (H,W) torch metres (0 = no reading); weight: optional (H,W) torch
    per-pixel data weight (e.g. number of fused readings).  Inside the
    discontinuity band the confidence is capped at `band_c` regardless of the
    weight, because the readings there are systematically wrong."""
    guide = to_t(rgb_u8).permute(2, 0, 1)
    guide = gauss_blur(guide, 0.8)
    valid = z > 0
    if min_component:
        valid = drop_specks(valid, min_component)
        z = torch.where(valid, z, torch.zeros_like(z))
    inv = torch.where(valid, 1.0 / z.clamp_min(1e-6), torch.zeros_like(z))
    c, edge, near_hole = raw_confidence(inv, valid, band_px, grad_rel)
    if weight is not None:
        base = torch.where(valid, weight.clamp_min(1.0), torch.zeros_like(weight))
        c = torch.where(edge, torch.clamp(base, max=1.0) * band_c,
                        torch.where(near_hole, torch.clamp(base * 0.1, max=1.0), base))
    # pass 1: colour-guided, moderate smoothing -> edges move onto RGB edges
    x1 = solve_pyramid(c, inv, guide, lam1, sigma_c, levels=levels)
    # robust re-weighting: discard measurements that disagree with pass 1
    res = (inv - x1).abs() / x1.clamp_min(1e-6)
    c2 = c * torch.exp(-(res / 0.02) ** 2)
    # pass 2: strong smoothing, but cut along depth edges of pass 1 too
    _EXTRA_SIGMA[0] = sigma_rel2
    x2 = solve_pyramid(c2, inv, guide, lam2, sigma_c, extra_aff=torch.log(x1.clamp_min(1e-3)), levels=levels,
                       x_init=x1)
    depth = 1.0 / x2.clamp_min(1e-3)
    if snap_iters:
        depth, snapped = snap_edges(depth, guide, iters=snap_iters, mask=(c < 0.05))
    else:
        snapped = torch.zeros_like(valid)
    sup = support_map(valid, guide, sigma_c) if want_support else None
    if return_debug:
        return depth, dict(c=c, c2=c2, edge=edge, x1=x1, x2=x2, inv=inv, valid=valid, snapped=snapped, support=sup)
    return depth


def drop_specks(valid, min_area):
    """Remove isolated speckles of readings (8-connected components smaller
    than min_area px).  On surfaces outside the sensor range (the arm link
    closer than the minimum Z, black anodised parts) RealSense returns only
    sparse spurious readings; propagating them would invent surfaces."""
    import cv2
    m = valid.to(torch.uint8).cpu().numpy()
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    return torch.from_numpy(keep[lab]).to(valid.device)


def support_map(valid, guide, sigma_c, lam=1.0, eps=0.01, iters=60):
    """How strongly each pixel is tied to a real reading through
    colour-coherent paths (solved at half resolution):
        (c + eps + lam L) s = c,   c = 1 on pixels with a reading.
    s ~ 1 on/near readings, decays over ~sqrt(lam/eps) = 10 half-res px
    inside uniform colour and immediately across colour edges."""
    c = down2(valid.float())
    g = down2(guide)
    W = color_affinity(g, sigma_c, 1e-4)
    Minv = 1.0 / (c + eps + lam * lap_diag(W, c.shape))
    A = lambda v: (c + eps) * v + lam * lap_apply(v, W)
    x = c.clone(); r = c - A(x); zz = Minv * r; p = zz.clone(); rz = (r * zz).sum()
    for _ in range(iters):
        Ap = A(p); al = rz / ((p * Ap).sum() + 1e-30)
        x = x + al * p; r = r - al * Ap; zz = Minv * r
        rzn = (r * zz).sum(); p = zz + (rzn / (rz + 1e-30)) * p; rz = rzn
    return up_to(x, valid.shape).clamp(0, 1)


def snap_edges(depth, guide, iters=3, k=7, rel=0.02, mask=None):
    """Remove 'flying' samples on depth ramps that the smooth solve leaves at
    weak colour edges: a pixel whose kxk neighbourhood spans a relative depth
    range > rel and whose own depth lies strictly between the near and far
    clusters is reassigned to the cluster whose mean colour is closer to its
    own colour (ties -> closer depth).  Only pixels in `mask` (no confident
    reading: inside the discontinuity band or a hole) are touched, so measured
    slanted surfaces are left as they are."""
    H, W = depth.shape
    r = k // 2
    snapped = torch.zeros((H, W), dtype=torch.bool, device=depth.device)
    oy, ox = torch.meshgrid(torch.arange(-r, r + 1, device=depth.device), torch.arange(-r, r + 1, device=depth.device),
                            indexing='ij')
    oy = oy.reshape(-1, 1); ox = ox.reshape(-1, 1)
    gflat = guide.reshape(3, -1)
    for _ in range(iters):
        mx = F.max_pool2d(depth[None, None], k, 1, r)[0, 0]
        mn = -F.max_pool2d(-depth[None, None], k, 1, r)[0, 0]
        cand = (mx - mn) / depth.clamp_min(1e-3) > rel
        if mask is not None:
            cand &= mask
        idx = torch.nonzero(cand.reshape(-1)).squeeze(1)
        if idx.numel() == 0:
            break
        y = idx // W; x = idx % W
        ny = (y[None] + oy).clamp(0, H - 1); nx = (x[None] + ox).clamp(0, W - 1)
        nidx = ny * W + nx                      # (k*k, n)
        d = depth.reshape(-1)
        Pc = d[nidx]
        lo_, hi_ = mn.reshape(-1)[idx], mx.reshape(-1)[idx]
        mid = 0.5 * (lo_ + hi_)
        near = Pc < mid[None]
        dd = d[idx]
        inside = (dd > lo_ + 0.2 * (hi_ - lo_)) & (dd < hi_ - 0.2 * (hi_ - lo_))
        G = gflat[:, nidx]                      # (3, k*k, n)
        own = gflat[:, idx]
        nf = near.float(); ff = 1 - nf
        cn = (G * nf[None]).sum(1) / nf.sum(0).clamp_min(1)[None]
        cf = (G * ff[None]).sum(1) / ff.sum(0).clamp_min(1)[None]
        zn = (Pc * nf).sum(0) / nf.sum(0).clamp_min(1)
        zf = (Pc * ff).sum(0) / ff.sum(0).clamp_min(1)
        dn = ((own - cn) ** 2).sum(0); df = ((own - cf) ** 2).sum(0)
        go_near = torch.where((dn - df).abs() > 1.0, dn < df, (dd - zn).abs() < (zf - dd).abs())
        new = torch.where(go_near, zn, zf)
        upd = inside & (nf.sum(0) >= 3) & (ff.sum(0) >= 3)
        if not bool(upd.any()):
            break
        d = d.clone()
        d[idx[upd]] = new[upd]
        depth = d.view(H, W)
        snapped.view(-1)[idx[upd]] = True
    return depth, snapped
