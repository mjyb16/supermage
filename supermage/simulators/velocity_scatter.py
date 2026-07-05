"""Velocity-axis scatter kernel shared by the analytic renderers.

Deposits per-pixel quantile sub-channels into a position-position-velocity
cube with linear two-bin weighting, dropping (not folding) out-of-band flux.
Used by :class:`supermage.simulators.analytic_cube.AnalyticInverse` and
:class:`supermage.simulators.lensed_cube.AnalyticLens`.
"""
import torch


def scatter_quantiles_along_v(cube_hi, v_sub, I_map, *, vel0_hi, dv_hi, Nv_hi,
                              Y_flat, X_flat, N_pix_hi, hw):
    """
    Linear two-bin scatter of K equal-flux velocity subsamples per pixel into
    the (high-res) velocity axis, DROPPING out-of-band flux.

    The legacy implementation clamped the bin indices into [0, Nv_hi-1], which
    folded any out-of-band line-of-sight velocity into the first/last channel —
    near a black hole this deposits a spurious, mass-dependent point source in
    the edge channels. Here the indices are still clamped (scatter_add requires
    in-range indices) but the weights of out-of-range target bins are zeroed,
    so each subsample contributes exactly its in-band fraction (a subsample
    straddling the band edge keeps its in-band half; fully out-of-band flux is
    dropped, not folded). The validity masks are autograd constants — gradients
    flow through the fractional weights as before — and the op stays
    torch.func.vmap-compatible.

    Parameters
    ----------
    cube_hi : (V*H*W-viewable) tensor, pre-zeroed output cube (V, H, W)
    v_sub   : (K, H, W) subsample velocities (v_los + quantile offsets)
    I_map   : (H, W) per-pixel intensity, split equally across the K subsamples
    vel0_hi, dv_hi, Nv_hi : velocity-axis origin/step/length (high-res)
    Y_flat, X_flat : (1, H*W) precomputed spatial indices
    N_pix_hi, hw   : spatial side and H*W stride
    """
    K = v_sub.shape[0]
    iv_f = (v_sub - vel0_hi) / dv_hi
    iv0f = torch.floor(iv_f)
    fv   = iv_f - iv0f                             # in [0, 1) by construction
    iv0  = iv0f.to(torch.long)

    val0 = (iv0 >= 0) & (iv0 <= Nv_hi - 1)         # target bin iv0   in range
    val1 = (iv0 >= -1) & (iv0 <= Nv_hi - 2)        # target bin iv0+1 in range

    iv0c = iv0.clamp(0, Nv_hi - 1)                 # scatter-safe indices ...
    iv1c = (iv0 + 1).clamp(0, Nv_hi - 1)
    w0 = (1 - fv) * val0.to(fv.dtype)              # ... with zeroed weights,
    w1 = fv * val1.to(fv.dtype)                    #     not clamped flux

    fsub = (I_map / float(K)).view(1, -1)          # (1, H*W)
    iv0c = iv0c.view(K, -1)
    iv1c = iv1c.view(K, -1)
    w0 = w0.view(K, -1)
    w1 = w1.view(K, -1)

    baseY = Y_flat.expand(K, -1)                   # (K, H*W)
    baseX = X_flat.expand(K, -1)
    idx0 = iv0c * hw + baseY * N_pix_hi + baseX
    idx1 = iv1c * hw + baseY * N_pix_hi + baseX

    flat = cube_hi.view(-1)
    flat = flat.scatter_add(0, idx0.reshape(-1), (fsub * w0).reshape(-1))
    flat = flat.scatter_add(0, idx1.reshape(-1), (fsub * w1).reshape(-1))
    return flat.view(cube_hi.shape)
