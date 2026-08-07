"""Tests for supermage/utils/likelihood.py — visibility likelihood infrastructure.

Covers:
- gaussian_loglike (analytic value, float64 reduction, autograd gradient),
- build_data_tensors (flat layout, weight formula, bad-std sanitization, device/dtype),
- load_raw_data for legacy FULL-plane and HALF-plane npz files (channel slicing
  gotcha, derived quantities, validation errors, window metadata pass-through),
- build_multi_gpu_log_likelihood on CPU (vmap + sequential paths, round-robin
  ordering, NaN -> logl_invalid, consistency with gaussian_loglike).
"""
import contextlib

import numpy as np
import pytest
import torch

from supermage.utils.likelihood import (
    build_data_tensors,
    build_multi_gpu_log_likelihood,
    gaussian_loglike,
    load_raw_data,
)

F = 4          # channels in the synthetic gridded files
NPIX = 9       # full uv-grid side
HALF = (NPIX + 1) // 2  # 5 — hermitian half-plane slab rows


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic npz builders
# ──────────────────────────────────────────────────────────────────────────────
def _legacy_arrays(n_chan=F, npix=NPIX, seed=42, descending=False):
    """Arrays for a legacy FULL-plane gridded npz."""
    rng = np.random.default_rng(seed)
    mask = rng.random((n_chan, npix, npix)) < 0.5
    if mask.sum() % 2:  # keep the full mask sum EVEN so sigma_max is formula-exact
        mask[np.unravel_index(int(np.argmin(mask)), mask.shape)] = True
    chan = np.linspace(230.0e9, 230.3e9, n_chan)
    if descending:
        chan = chan[::-1].copy()
    return dict(
        npix=npix,
        fov_arcsec=3.6,
        delta_u=1234.5,
        vis_bin_re=rng.normal(size=(n_chan, npix, npix)),
        vis_bin_imag=rng.normal(size=(n_chan, npix, npix)),
        std_bin_re=rng.uniform(0.5, 2.0, size=(n_chan, npix, npix)),
        std_bin_imag=rng.uniform(0.5, 2.0, size=(n_chan, npix, npix)),
        mask=mask,
        chan_freq_hz=chan,
    )


def _half_arrays(n_chan=F, npix=NPIX, seed=7):
    """Arrays for a HALF-plane gridded npz (slab shape ((npix+1)//2, npix))."""
    rng = np.random.default_rng(seed)
    h = (npix + 1) // 2
    mask = rng.random((n_chan, h, npix)) < 0.6
    vis_re = rng.normal(size=(n_chan, h, npix))
    vis_im = rng.normal(size=(n_chan, h, npix))
    std_re = rng.uniform(0.5, 2.0, size=(n_chan, h, npix))
    std_im = rng.uniform(0.5, 2.0, size=(n_chan, h, npix))
    off = ~mask
    assert off.any()  # need masked cells for the zeroing checks below
    vis_re[off] = 0.0
    vis_im[off] = 0.0
    std_re[off] = 1.0
    std_im[off] = 1.0
    return dict(
        npix=npix,
        fov_arcsec=3.6,
        delta_u=1234.5,
        vis_bin_re=vis_re,
        vis_bin_imag=vis_im,
        std_bin_re=std_re,
        std_bin_imag=std_im,
        mask=mask,
        chan_freq_hz=np.linspace(230.0e9, 230.3e9, n_chan),
        half_plane=True,
        taper_map=rng.uniform(0.5, 1.0, size=(npix, npix)),
    )


def _save(tmp_path, name, arrays):
    path = tmp_path / name
    np.savez(path, **arrays)
    return str(path)


# ──────────────────────────────────────────────────────────────────────────────
# gaussian_loglike
# ──────────────────────────────────────────────────────────────────────────────
class TestGaussianLoglike:
    def test_matches_analytic_formula_float64(self):
        rng = np.random.default_rng(1)
        yhat = rng.normal(size=50)
        vis = rng.normal(size=50)
        w = rng.uniform(0.5, 3.0, size=50)
        got = gaussian_loglike(
            torch.tensor(yhat), torch.tensor(vis), torch.tensor(w)
        )
        want = -0.5 * np.sum(((yhat - vis) * w) ** 2)  # independent numpy float64
        assert got.dtype == torch.float64
        assert float(got) == pytest.approx(want, rel=1e-14)

    def test_float32_inputs_reduce_in_float64(self):
        rng = np.random.default_rng(2)
        yhat32 = torch.tensor(rng.normal(size=64), dtype=torch.float32)
        vis32 = torch.tensor(rng.normal(size=64), dtype=torch.float32)
        w32 = torch.tensor(rng.uniform(0.5, 2.0, size=64), dtype=torch.float32)
        got = gaussian_loglike(yhat32, vis32, w32)
        assert got.dtype == torch.float64  # reduction promoted despite fp32 inputs
        # Reference: residual formed in float32 (as the code does), summed in float64.
        resid32 = ((yhat32 - vis32) * w32).numpy().astype(np.float64)
        assert float(got) == pytest.approx(-0.5 * np.sum(resid32**2), rel=1e-14)

    def test_gradient_matches_analytic(self):
        rng = np.random.default_rng(3)
        yhat = torch.tensor(rng.normal(size=30), dtype=torch.float64,
                            requires_grad=True)
        vis = torch.tensor(rng.normal(size=30), dtype=torch.float64)
        w = torch.tensor(rng.uniform(0.5, 2.0, size=30), dtype=torch.float64)
        logl = gaussian_loglike(yhat, vis, w)
        logl.backward()
        analytic = -(yhat.detach() - vis) * w**2  # d/dyhat of -0.5*((yhat-v)*w)^2
        assert torch.allclose(yhat.grad, analytic, rtol=1e-12, atol=0)


# ──────────────────────────────────────────────────────────────────────────────
# build_data_tensors
# ──────────────────────────────────────────────────────────────────────────────
class TestBuildDataTensors:
    def _raw(self, seed=4):
        rng = np.random.default_rng(seed)
        shape = (2, 3, 3)
        return dict(
            vis_bin_re=rng.normal(size=shape),
            vis_bin_imag=rng.normal(size=shape),
            std_bin_re=rng.uniform(0.5, 2.0, size=shape),
            std_bin_imag=rng.uniform(0.5, 2.0, size=shape),
        )

    def test_vis_flat_layout_re_then_im(self):
        raw = self._raw()
        vis_flat, _ = build_data_tensors(raw, 1.0, "cpu", dtype=torch.float64)
        n = raw["vis_bin_re"].size
        assert vis_flat.shape == (2 * n,)
        np.testing.assert_array_equal(vis_flat[:n].numpy(),
                                      raw["vis_bin_re"].flatten())
        np.testing.assert_array_equal(vis_flat[n:].numpy(),
                                      raw["vis_bin_imag"].flatten())

    def test_weights_are_inverse_broadened_std(self):
        raw = self._raw(seed=5)
        sigma_broad = 3.0
        _, sqrt_Cinv = build_data_tensors(raw, sigma_broad, "cpu",
                                          dtype=torch.float64)
        std_flat = np.concatenate([raw["std_bin_re"].flatten(),
                                   raw["std_bin_imag"].flatten()])
        want = 1.0 / (std_flat * sigma_broad)
        np.testing.assert_allclose(sqrt_Cinv.numpy(), want, rtol=1e-13)

    def test_bad_std_entries_replaced_by_one_before_broadening(self):
        raw = self._raw(seed=6)
        # Plant 0 / NaN / +inf / -inf at known flat positions in BOTH std maps.
        bad_vals = [0.0, np.nan, np.inf, -np.inf]
        re_flat = raw["std_bin_re"].reshape(-1)
        im_flat = raw["std_bin_imag"].reshape(-1)
        re_pos, im_pos = [0, 3, 7, 12], [1, 4, 9, 15]
        for p, v in zip(re_pos, bad_vals):
            re_flat[p] = v
        for p, v in zip(im_pos, bad_vals):
            im_flat[p] = v
        sigma_broad = 2.0  # power of two -> weight 1/sigma_broad is fp-exact
        _, sqrt_Cinv = build_data_tensors(raw, sigma_broad, "cpu",
                                          dtype=torch.float64)
        n = re_flat.size
        w = sqrt_Cinv.numpy()
        # bad std -> 1 BEFORE broadening -> weight == 1/sigma_broad EXACTLY
        for p in re_pos:
            assert w[p] == 1.0 / sigma_broad
        for p in im_pos:
            assert w[n + p] == 1.0 / sigma_broad
        # untouched entries keep 1/(std*sigma_broad)
        good = np.setdiff1d(np.arange(n), re_pos)
        np.testing.assert_allclose(
            w[good], 1.0 / (re_flat[good] * sigma_broad), rtol=1e-13)
        assert np.all(np.isfinite(w)) and np.all(w > 0)

    def test_device_and_dtype_respected(self):
        raw = self._raw(seed=8)
        vis_flat, sqrt_Cinv = build_data_tensors(
            raw, 1.5, torch.device("cpu"), dtype=torch.float32)
        assert vis_flat.dtype == torch.float32
        assert sqrt_Cinv.dtype == torch.float32
        assert vis_flat.device.type == "cpu"
        assert sqrt_Cinv.device.type == "cpu"
        std_flat = np.concatenate([raw["std_bin_re"].flatten(),
                                   raw["std_bin_imag"].flatten()])
        np.testing.assert_allclose(sqrt_Cinv.numpy(),
                                   1.0 / (std_flat * 1.5), rtol=1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# load_raw_data — legacy FULL-plane files
# ──────────────────────────────────────────────────────────────────────────────
class TestLoadRawDataFullPlane:
    def test_default_max_freq_index_drops_last_channel(self, tmp_path):
        # The historical 79/80-channel gotcha: default -1 slices [:-1].
        arrays = _legacy_arrays()
        d = load_raw_data(_save(tmp_path, "full.npz", arrays))
        for k in ("vis_bin_re", "vis_bin_imag", "std_bin_re", "std_bin_imag",
                  "data_mask", "chan_freq_hz"):
            assert d[k].shape[0] == F - 1, k
        np.testing.assert_array_equal(d["chan_freq_hz"],
                                      arrays["chan_freq_hz"][:-1])
        np.testing.assert_array_equal(d["vis_bin_re"],
                                      arrays["vis_bin_re"][:-1])

    def test_max_freq_index_none_keeps_all_channels(self, tmp_path):
        arrays = _legacy_arrays()
        d = load_raw_data(_save(tmp_path, "full.npz", arrays),
                          max_freq_index=None)
        for k in ("vis_bin_re", "vis_bin_imag", "std_bin_re", "std_bin_imag",
                  "data_mask", "chan_freq_hz"):
            assert d[k].shape[0] == F, k
        np.testing.assert_array_equal(d["chan_freq_hz"], arrays["chan_freq_hz"])

    def test_derived_quantities(self, tmp_path):
        arrays = _legacy_arrays()
        path = _save(tmp_path, "full.npz", arrays)
        d = load_raw_data(path)  # default: last channel dropped
        assert d["half_plane"] is False
        assert d["taper_map"] is None
        assert d["npix_uv"] == NPIX
        assert d["arcsec_per_pix"] == pytest.approx(
            arrays["fov_arcsec"] / NPIX, rel=1e-15)
        # n_chi counts the SLICED mask
        assert d["n_chi"] == int(arrays["mask"][:-1].sum())
        np.testing.assert_allclose(d["freq_ghz_fit"],
                                   arrays["chan_freq_hz"][:-1] / 1e9,
                                   rtol=1e-15)
        # sigma_max uses the FULL (unsliced) mask, halved for hermitian files
        full_sum = int(arrays["mask"].sum())
        assert full_sum % 2 == 0  # by construction
        assert d["sigma_max"] == pytest.approx(
            (2.0 * (full_sum / 2)) ** 0.25, rel=1e-15)
        assert d["sigma_hot"] == pytest.approx(2.0 * d["sigma_max"], rel=1e-15)
        # unchanged when channels are sliced differently
        d_all = load_raw_data(path, max_freq_index=None)
        assert d_all["sigma_max"] == d["sigma_max"]
        assert d_all["n_chi"] == int(arrays["mask"].sum())

    def test_descending_freq_axis_raises(self, tmp_path):
        arrays = _legacy_arrays(descending=True)
        path = _save(tmp_path, "desc.npz", arrays)
        with pytest.raises(AssertionError, match="ascending"):
            load_raw_data(path)


# ──────────────────────────────────────────────────────────────────────────────
# load_raw_data — HALF-plane files
# ──────────────────────────────────────────────────────────────────────────────
class TestLoadRawDataHalfPlane:
    def test_valid_half_plane_file(self, tmp_path):
        arrays = _half_arrays()
        d = load_raw_data(_save(tmp_path, "half.npz", arrays),
                          max_freq_index=None)
        assert d["half_plane"] is True
        for k in ("vis_bin_re", "vis_bin_imag", "std_bin_re", "std_bin_imag",
                  "data_mask"):
            assert d[k].shape == (F, HALF, NPIX), k
        assert d["taper_map"] is not None
        np.testing.assert_allclose(d["taper_map"], arrays["taper_map"],
                                   rtol=1e-15)
        # n_chi_half == FULL mask sum (NOT halved) for half-plane files;
        # observable through sigma_max.
        full_sum = int(arrays["mask"].sum())
        assert d["sigma_max"] == pytest.approx((2.0 * full_sum) ** 0.25,
                                               rel=1e-15)
        assert d["n_chi"] == full_sum  # max_freq_index=None: no slicing
        # masked cells stayed zero through the loader
        off = ~d["data_mask"].astype(bool)
        assert np.all(d["vis_bin_re"][off] == 0.0)
        assert np.all(d["vis_bin_imag"][off] == 0.0)

    def test_slab_shape_mismatch_raises(self, tmp_path):
        # full-plane-shaped arrays but half_plane=True -> ValueError
        arrays = _legacy_arrays()
        arrays["half_plane"] = True
        arrays["taper_map"] = np.ones((NPIX, NPIX))
        path = _save(tmp_path, "badshape.npz", arrays)
        with pytest.raises(ValueError, match="slab"):
            load_raw_data(path, max_freq_index=None)

    def test_nonzero_vis_in_masked_cell_raises(self, tmp_path):
        # REGRESSION: leftover data at ~mask would add a constant chi2 offset.
        arrays = _half_arrays()
        off_idx = np.argwhere(~arrays["mask"])[0]
        arrays["vis_bin_re"][tuple(off_idx)] = 0.3
        path = _save(tmp_path, "dirty.npz", arrays)
        with pytest.raises(ValueError, match="NONZERO"):
            load_raw_data(path, max_freq_index=None)

    def test_missing_taper_map_raises(self, tmp_path):
        arrays = _half_arrays()
        del arrays["taper_map"]
        path = _save(tmp_path, "notaper.npz", arrays)
        with pytest.raises(ValueError, match="taper_map"):
            load_raw_data(path, max_freq_index=None)


# ──────────────────────────────────────────────────────────────────────────────
# window metadata pass-through
# ──────────────────────────────────────────────────────────────────────────────
class TestWindowMetadata:
    def test_window_keys_pass_through_when_present(self, tmp_path):
        arrays = _legacy_arrays()
        arrays["window"] = "kaiser_bessel"
        arrays["window_m"] = 1
        arrays["window_beta"] = 2.34
        d = load_raw_data(_save(tmp_path, "win.npz", arrays))
        assert str(d["window"]) == "kaiser_bessel"
        assert int(d["window_m"]) == 1
        assert float(d["window_beta"]) == pytest.approx(2.34, rel=1e-15)

    def test_window_keys_absent_when_missing(self, tmp_path):
        d = load_raw_data(_save(tmp_path, "nowin.npz", _legacy_arrays()))
        for k in ("window", "window_m", "window_beta"):
            assert k not in d


# ──────────────────────────────────────────────────────────────────────────────
# build_multi_gpu_log_likelihood — CPU "GPUs"
# ──────────────────────────────────────────────────────────────────────────────
D_PAR = 3       # parameter dimension of the toy linear model
D_OUT = 10      # flattened visibility length
LOGL_INVALID = -1.0e12


def _toy_problem(seed=11):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(D_OUT, D_PAR))
    vis = rng.normal(size=D_OUT)
    w = rng.uniform(0.5, 2.0, size=D_OUT)
    return A, vis, w


def _cpu_resources(A, vis, w, with_vmap):
    """Two CPU 'GPU' resources sharing a toy linear forward model.

    The forward NaN-poisons its output when x[0] < 0 (sqrt of a negative,
    vmap-safe) so invalid-point handling can be exercised.
    """
    def make_one():
        A_t = torch.tensor(A, dtype=torch.float64)
        vis_t = torch.tensor(vis, dtype=torch.float64)
        w_t = torch.tensor(w, dtype=torch.float64)

        def fwd_single(x):
            return A_t @ x + 0.0 * torch.sqrt(x[0])

        return {
            "dev": torch.device("cpu"),
            "vis_flat": vis_t,
            "sqrt_Cinv": w_t,
            "fwd_single": fwd_single,
            "batched_fwd": torch.func.vmap(fwd_single) if with_vmap else None,
        }

    return [make_one(), make_one()]


def _ref_logl(X, A, vis, w):
    """Independent per-row float64 gaussian log-likelihood."""
    Y = X @ A.T
    r = (Y - vis[None, :]) * w[None, :]
    return -0.5 * np.sum(r**2, axis=1)


@contextlib.contextmanager
def _loglike(resources, vmap_batch_size=3, logl_invalid=LOGL_INVALID):
    fn, executor = build_multi_gpu_log_likelihood(
        resources, vmap_batch_size, logl_invalid)
    try:
        yield fn
    finally:
        executor.shutdown(wait=True)


class TestMultiGpuLogLikelihood:
    @pytest.mark.parametrize("with_vmap", [True, False],
                             ids=["vmap", "sequential"])
    def test_matches_reference_and_preserves_ordering(self, with_vmap):
        A, vis, w = _toy_problem()
        rng = np.random.default_rng(12)
        # N=7 with 2 workers -> unequal round-robin shares (4 and 3);
        # x[0] > 0 everywhere so all rows are valid.
        X = rng.uniform(0.1, 1.0, size=(7, D_PAR))
        ref = _ref_logl(X, A, vis, w)
        assert len(np.unique(ref)) == 7  # distinct rows -> ordering is testable
        with _loglike(_cpu_resources(A, vis, w, with_vmap)) as fn:
            out = fn(X)
        assert out.shape == (7,)
        assert out.dtype == np.float64
        np.testing.assert_allclose(out, ref, rtol=1e-10)

    @pytest.mark.parametrize("with_vmap", [True, False],
                             ids=["vmap", "sequential"])
    def test_nan_forward_row_gets_logl_invalid_exactly(self, with_vmap):
        A, vis, w = _toy_problem()
        rng = np.random.default_rng(13)
        X = rng.uniform(0.1, 1.0, size=(7, D_PAR))
        X[4, 0] = -0.5  # sqrt(-0.5) -> NaN-poisoned forward output
        ref = _ref_logl(X, A, vis, w)
        with _loglike(_cpu_resources(A, vis, w, with_vmap)) as fn:
            out = fn(X)
        assert out[4] == LOGL_INVALID  # exact sentinel, no arithmetic on it
        good = [i for i in range(7) if i != 4]
        np.testing.assert_allclose(out[good], ref[good], rtol=1e-10)

    def test_vmap_and_sequential_paths_agree(self):
        A, vis, w = _toy_problem(seed=14)
        rng = np.random.default_rng(15)
        X = rng.uniform(0.1, 1.0, size=(6, D_PAR))
        with _loglike(_cpu_resources(A, vis, w, True)) as fn_v:
            out_v = fn_v(X)
        with _loglike(_cpu_resources(A, vis, w, False)) as fn_s:
            out_s = fn_s(X)
        np.testing.assert_allclose(out_v, out_s, rtol=1e-12)

    def test_consistent_with_gaussian_loglike_single_row(self):
        A, vis, w = _toy_problem(seed=16)
        rng = np.random.default_rng(17)
        x = rng.uniform(0.1, 1.0, size=(1, D_PAR))
        resources = _cpu_resources(A, vis, w, True)
        # gaussian_loglike on the exact same forward output
        x_t = torch.tensor(x[0], dtype=torch.float64)
        yhat = resources[0]["fwd_single"](x_t)
        direct = float(gaussian_loglike(yhat,
                                        resources[0]["vis_flat"],
                                        resources[0]["sqrt_Cinv"]))
        with _loglike(resources) as fn:
            out = fn(x)
        assert out[0] == pytest.approx(direct, rel=1e-15)
