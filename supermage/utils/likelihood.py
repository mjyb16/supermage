"""Shared visibility-likelihood infrastructure (dataset-agnostic).

Turns gridded interferometric data + a SuperMAGE forward model into a Gaussian
visibility log-likelihood, single-GPU or multi-GPU (vectorised across N GPUs for
nested sampling).

Precision note (important)
--------------------------
The forward model and data tensors are kept in **float32** (memory/speed), but the
chi^2 **reduction is done in float64**.  At the |logL| magnitudes of real ALMA fits
(~1e5-1e6) the float32 ULP (~0.016-0.25) quantises distinct parameter points onto
identical logL values -> plateaus -> degenerate live points -> UltraNest's MLFriends
whitening goes singular -> ``LinAlgError: Distances are not positive``.  Doing only the
*reduction* in float64 removes all ties and reproduces full-float64 logL ordering
exactly (Spearman = 1.0), at negligible cost; full-float64 forward models are
unnecessary.  See ``claude_notes/ngc4697_notes.md`` ("UltraNest NS Failure").
"""
import concurrent.futures

import numpy as np
import torch

__all__ = [
    "load_raw_data",
    "build_data_tensors",
    "gaussian_loglike",
    "build_multi_gpu_log_likelihood",
]


# ──────────────────────────────────────────────────────────────────────────────
# Data I/O
# ──────────────────────────────────────────────────────────────────────────────
def load_raw_data(data_file, max_freq_index=-1):
    """Load a gridded interferometric ``.npz`` (VisCube output) into a dict of raw arrays.

    Supports both conventions:
      - legacy FULL-plane files (hermitian-redundant grids; every uv cell and its
        conjugate mirror both stored/fit),
      - HALF-plane files (``half_plane=True`` in the npz): only the non-redundant
        Hermitian slab, shape ``(F, (npix+1)//2, npix)``, produced by
        ``viscube.half_plane_slab`` — this removes the double-counting that
        understated posterior widths by up to ~2x.

    Half-plane files carry the image-plane ``taper_map`` (the analytic FT of the
    gridding kernel) which the forward model must MULTIPLY in; it is exposed as
    ``d["taper_map"]``. The legacy ``apo_map`` key (divide convention) is
    deliberately NOT read.
    """
    raw = np.load(data_file)
    half_plane = bool(raw["half_plane"]) if "half_plane" in raw.files else False
    d = {
        "npix_uv":      int(raw["npix"]),
        "fov_arcsec":   float(raw["fov_arcsec"]),
        "delta_u":      float(raw["delta_u"]),
        "vis_bin_re":   raw["vis_bin_re"]  [:max_freq_index],
        "vis_bin_imag": raw["vis_bin_imag"][:max_freq_index],
        "std_bin_re":   raw["std_bin_re"]  [:max_freq_index],
        "std_bin_imag": raw["std_bin_imag"][:max_freq_index],
        "data_mask":    raw["mask"]        [:max_freq_index],
        "chan_freq_hz": raw["chan_freq_hz"][:max_freq_index],
        "half_plane":   half_plane,
    }
    assert np.all(np.diff(d["chan_freq_hz"]) > 0), \
        "Channels must be in ascending frequency order"

    d["taper_map"] = (np.asarray(raw["taper_map"], dtype=float)
                      if "taper_map" in raw.files else None)
    for meta_key in ("window", "window_m", "window_beta"):
        if meta_key in raw.files:
            d[meta_key] = raw[meta_key][()]

    npix = d["npix_uv"]
    if half_plane:
        slab_shape = ((npix + 1) // 2, npix)
        for k in ("vis_bin_re", "vis_bin_imag", "std_bin_re", "std_bin_imag", "data_mask"):
            if d[k].shape[-2:] != slab_shape:
                raise ValueError(
                    f"half_plane npz: {k} last axes {d[k].shape[-2:]} != expected "
                    f"slab {slab_shape} (npix={npix} is the FULL grid side)."
                )
        # Masked cells MUST hold zero data: the DC row's conjugate-duplicate
        # columns are masked out but were gridded — if their data survived,
        # (model=0 - data) would add a spurious constant chi^2 offset.
        off = ~d["data_mask"].astype(bool)
        if not (np.all(d["vis_bin_re"][off] == 0.0) and np.all(d["vis_bin_imag"][off] == 0.0)):
            raise ValueError(
                "half_plane npz: masked cells contain NONZERO visibilities. "
                "Zero vis (and set std=1) at ~mask before saving "
                "(see the publication_run_nautilus gridding notebooks)."
            )
        if d["taper_map"] is None:
            raise ValueError("half_plane npz is missing the 'taper_map' key.")

    d["freq_ghz_fit"]   = d["chan_freq_hz"] / 1e9
    d["arcsec_per_pix"] = d["fov_arcsec"] / d["npix_uv"]
    d["n_chi"]          = int(np.sum(d["data_mask"]))
    # data-scale helpers used by some samplers (e.g. PT temperature ladders).
    # n_chi_half = number of INDEPENDENT complex cells (FULL mask, before channel
    # slicing): half the mask sum for hermitian-redundant full-plane files, the
    # mask sum itself for half-plane files.
    if half_plane:
        n_chi_half = int(np.sum(raw["mask"]))
    else:
        n_chi_half = int(np.sum(raw["mask"]) / 2)
    d["sigma_max"]      = (2.0 * n_chi_half) ** 0.25
    d["sigma_hot"]      = 2.0 * d["sigma_max"]
    return d


def build_data_tensors(raw, sigma_broad, device, dtype=torch.float32):
    """Push gridded visibilities to ``device``; return ``(vis_flat, sqrt_Cinv)``.

    ``vis_flat``  : real then imag, flattened, shape ``(2 * N_uv_cells,)``.
    ``sqrt_Cinv`` : ``1/std`` weights (with ``std`` inflated by ``sigma_broad``); zeros/NaNs
                    in the std map are replaced by 1 so they contribute finite (down-weighted)
                    terms.  Multiply a residual by ``sqrt_Cinv`` to get the whitened residual.
    """
    vis_re_t = torch.tensor(raw["vis_bin_re"],   device=device, dtype=dtype)
    vis_im_t = torch.tensor(raw["vis_bin_imag"], device=device, dtype=dtype)
    std_re_t = torch.nan_to_num(torch.tensor(raw["std_bin_re"],  device=device, dtype=dtype),
                                nan=1., posinf=1., neginf=1.)
    std_im_t = torch.nan_to_num(torch.tensor(raw["std_bin_imag"], device=device, dtype=dtype),
                                nan=1., posinf=1., neginf=1.)
    std_re_t[std_re_t == 0] = 1.
    std_im_t[std_im_t == 0] = 1.
    vis_flat  = torch.cat([vis_re_t.flatten(), vis_im_t.flatten()])
    std_flat  = torch.cat([std_re_t.flatten(), std_im_t.flatten()]) * sigma_broad
    sqrt_Cinv = torch.sqrt(1.0 / (std_flat ** 2))
    return vis_flat, sqrt_Cinv


# ──────────────────────────────────────────────────────────────────────────────
# Single-sample Gaussian log-likelihood (float64 reduction)
# ──────────────────────────────────────────────────────────────────────────────
def gaussian_loglike(yhat, vis_flat, sqrt_Cinv):
    """``-0.5 * || (yhat - vis_flat) * sqrt_Cinv ||^2`` with a **float64 reduction**.

    ``yhat`` may be float32 (e.g. a float32 forward model); the square+sum is performed in
    float64 to avoid logL quantization plateaus.  Differentiable (the float64 cast preserves
    the autograd graph), so it is also a valid MALA/HMC target.
    """
    resid = (yhat - vis_flat) * sqrt_Cinv
    return -0.5 * (resid.double() ** 2).sum()


# ──────────────────────────────────────────────────────────────────────────────
# Multi-GPU vectorised log-likelihood (for UltraNest / vectorised nested sampling)
# ──────────────────────────────────────────────────────────────────────────────
def build_multi_gpu_log_likelihood(gpu_resources, vmap_batch_size, logl_invalid):
    """Return ``(log_likelihood_vec, executor)`` for multi-GPU vectorised evaluation.

    Distributes each ``log_likelihood_vec(X)`` call (shape ``(N, D)`` -> ``(N,)``) across N GPUs
    using one thread per GPU (CUDA releases the GIL during kernels -> genuine parallelism).

    Each element of ``gpu_resources`` is a dict with keys:
      ``dev`` (torch.device), ``vis_flat`` (Tensor (2*Npix,) on dev),
      ``sqrt_Cinv`` (Tensor (2*Npix,) on dev), ``fwd_single`` ((x,)->yhat on dev),
      ``batched_fwd`` (vmap(fwd_single) or None).

    ``executor`` is a ThreadPoolExecutor owned by the caller — call
    ``executor.shutdown(wait=True)`` when sampling finishes.  The chi^2 reduction is float64
    (see module docstring) in both the vmap and sequential paths.
    """
    n_gpus = len(gpu_resources)

    # Each GPU gets its own mutable vmap flag so OOM on one GPU doesn't disable the others.
    for res in gpu_resources:
        res["_use_vmap"] = [res.get("batched_fwd") is not None]

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=n_gpus)

    def _eval_on_gpu(res, X_sub):
        n_sub     = len(X_sub)
        out       = np.full(n_sub, logl_invalid, dtype=np.float64)
        dev       = res["dev"]
        vis_flat  = res["vis_flat"]
        sqrt_Cinv = res["sqrt_Cinv"]
        use_vmap  = res["_use_vmap"]          # mutable list — shared by reference

        for start in range(0, n_sub, vmap_batch_size):
            end     = min(start + vmap_batch_size, n_sub)
            chunk   = end - start
            X_chunk = torch.tensor(X_sub[start:end], device=dev, dtype=vis_flat.dtype)

            if use_vmap[0] and chunk > 1:
                try:
                    with torch.no_grad():
                        Yhat = res["batched_fwd"](X_chunk)
                    valid = ~(Yhat.isnan().any(dim=1) | Yhat.isinf().any(dim=1))
                    resid = (Yhat - vis_flat.unsqueeze(0)) * sqrt_Cinv.unsqueeze(0)
                    logl  = -0.5 * (resid.double() ** 2).sum(dim=1)     # float64 reduction
                    logl  = torch.where(valid, logl, torch.full_like(logl, logl_invalid))
                    out[start:end] = logl.detach().cpu().numpy().astype(np.float64)
                    continue
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        print(f"[multi_gpu] CUDA OOM on {dev} — switching to sequential loop "
                              "for this GPU.")
                        use_vmap[0] = False
                        torch.cuda.empty_cache()
                    else:
                        raise

            # Sequential fallback (also used when chunk == 1)
            for i, x in enumerate(X_chunk):
                with torch.no_grad():
                    yhat = res["fwd_single"](x)
                if not (yhat.isnan().any() or yhat.isinf().any()):
                    resid = (yhat - vis_flat) * sqrt_Cinv
                    out[start + i] = float(-0.5 * (resid.double() ** 2).sum())   # float64 reduction

        return out

    def log_likelihood_vec(X_np):
        """Vectorised Gaussian log-likelihood dispatched across all GPUs (round-robin split)."""
        N       = len(X_np)
        splits  = [X_np[g::n_gpus] for g in range(n_gpus)]
        indices = [np.arange(g, N, n_gpus) for g in range(n_gpus)]
        futures = [executor.submit(_eval_on_gpu, gpu_resources[g], splits[g])
                   for g in range(n_gpus)]
        out = np.full(N, logl_invalid, dtype=np.float64)
        for g, fut in enumerate(futures):
            out[indices[g]] = fut.result()
        return out

    return log_likelihood_vec, executor
