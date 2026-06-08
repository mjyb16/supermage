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
    """Load a gridded interferometric ``.npz`` (VisCube output) into a dict of raw arrays."""
    raw = np.load(data_file)
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
    }
    assert np.all(np.diff(d["chan_freq_hz"]) > 0), \
        "Channels must be in ascending frequency order"
    d["freq_ghz_fit"]   = d["chan_freq_hz"] / 1e9
    d["arcsec_per_pix"] = d["fov_arcsec"] / d["npix_uv"]
    d["n_chi"]          = int(np.sum(d["data_mask"]))
    # data-scale helpers used by some samplers (e.g. PT temperature ladders)
    n_chi_half          = int(np.sum(raw["mask"]) / 2)   # FULL mask, before channel slicing
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
