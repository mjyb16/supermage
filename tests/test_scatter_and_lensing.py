"""Tests for the velocity scatter kernel and the lensed cube renderers.

Covers
------
- ``supermage.simulators.velocity_scatter.scatter_quantiles_along_v``:
  hand-checked two-bin splits, out-of-band flux dropping (regression for the
  legacy edge-folding bug), edge straddling, flux conservation, analytic
  autograd gradients and ``torch.func.vmap`` compatibility.
- ``supermage.simulators.lensed_cube.AnalyticLens``: identity-lens
  equivalence with ``AnalyticInverse``, the lens-center NaN guard (the ID141
  fix), source-plane offset geometry and the ``chunk_v`` tiling path.
- ``supermage.simulators.lensed_cube.CubeLens`` with a zero-deflection
  caustics SIE, and ``gaussian_quantile_offsets`` vs the analytic-cube
  ``gaussian_quantile_offsets_flex`` (checked against ``scipy`` ``ndtri``).
"""
import math

import numpy as np
import pytest
import torch

from conftest import LINE_HZ, flat_theta, make_freq_axis
from supermage.simulators.velocity_scatter import scatter_quantiles_along_v


# ---------------------------------------------------------------------------
# Helpers: tiny hand-checkable scatter setup (built exactly like
# AnalyticInverse.__init__ builds its Y_flat/X_flat/hw).
# ---------------------------------------------------------------------------
NV, NPIX = 8, 2          # 8 velocity bins, 2x2 spatial grid
VEL0, DV = 0.0, 10.0     # bin centers at 0, 10, ..., 70 km/s


def scatter_kwargs(n_pix=NPIX, nv=NV, vel0=VEL0, dv=DV):
    yy = torch.arange(n_pix)
    xx = torch.arange(n_pix)
    Y, X = torch.meshgrid(yy, xx, indexing="ij")
    return dict(
        vel0_hi=vel0, dv_hi=dv, Nv_hi=nv,
        Y_flat=Y.reshape(1, -1), X_flat=X.reshape(1, -1),
        N_pix_hi=n_pix, hw=n_pix * n_pix,
    )


def run_scatter(v_sub, I_map, dtype=torch.float64, **overrides):
    kw = scatter_kwargs()
    kw.update(overrides)
    cube = torch.zeros(kw["Nv_hi"], kw["N_pix_hi"], kw["N_pix_hi"], dtype=dtype)
    return scatter_quantiles_along_v(cube, v_sub, I_map, **kw)


# ---------------------------------------------------------------------------
# scatter_quantiles_along_v
# ---------------------------------------------------------------------------
class TestScatterQuantiles:
    def test_two_bin_linear_split_exact(self):
        # One pixel with I=1, K=1, v exactly halfway between bin centers 3 and 4.
        I_map = torch.zeros(NPIX, NPIX, dtype=torch.float64)
        I_map[0, 0] = 1.0
        v = torch.full((1, NPIX, NPIX), 35.0, dtype=torch.float64)  # iv_f = 3.5
        out = run_scatter(v, I_map)
        assert out[3, 0, 0].item() == 0.5
        assert out[4, 0, 0].item() == 0.5
        # Nothing anywhere else (other channels of that pixel, other pixels).
        rest = out.clone()
        rest[3, 0, 0] = rest[4, 0, 0] = 0.0
        assert torch.all(rest == 0)

    def test_on_bin_center_all_flux_in_one_bin(self):
        I_map = torch.zeros(NPIX, NPIX, dtype=torch.float64)
        I_map[1, 0] = 2.0
        v = torch.full((1, NPIX, NPIX), 30.0, dtype=torch.float64)  # exactly bin 3
        out = run_scatter(v, I_map)
        assert out[3, 1, 0].item() == 2.0
        assert out.sum().item() == 2.0

    def test_per_pixel_channel_routing(self):
        # Distinct on-center velocity per pixel: flux must land at the right
        # (channel, y, x) voxel — guards the flat-index arithmetic.
        I_map = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        v = torch.tensor([[[10.0, 20.0], [40.0, 60.0]]], dtype=torch.float64)
        out = run_scatter(v, I_map)
        expect = torch.zeros_like(out)
        expect[1, 0, 0] = 1.0
        expect[2, 0, 1] = 2.0
        expect[4, 1, 0] = 3.0
        expect[6, 1, 1] = 4.0
        assert torch.equal(out, expect)

    @pytest.mark.parametrize("v_off", [-1.0e6, -5.0 * DV, +1.0e6, VEL0 + NV * DV])
    def test_out_of_band_flux_dropped_not_folded(self, v_off):
        # REGRESSION: legacy clamping folded out-of-band flux into the edge
        # channels. Fully out-of-band velocities must contribute exactly 0,
        # in particular to channels 0 and Nv-1.
        I_map = torch.ones(NPIX, NPIX, dtype=torch.float64)
        v = torch.full((1, NPIX, NPIX), v_off, dtype=torch.float64)
        out = run_scatter(v, I_map)
        assert torch.all(out[0] == 0)
        assert torch.all(out[NV - 1] == 0)
        assert out.abs().sum().item() == 0.0

    def test_bottom_edge_straddle_keeps_only_inband_fraction(self):
        # v = -2.5: iv_f = -0.25 -> iv0 = -1, fv = 0.75. Only the fv fraction
        # lands in bin 0; the (1-fv)=0.25 share is dropped (bin -1 is outside).
        I_map = torch.zeros(NPIX, NPIX, dtype=torch.float64)
        I_map[0, 1] = 1.0
        v = torch.full((1, NPIX, NPIX), -2.5, dtype=torch.float64)
        out = run_scatter(v, I_map)
        assert out[0, 0, 1].item() == 0.75
        assert out.sum().item() == 0.75
        assert torch.all(out[1:] == 0)

    def test_top_edge_straddle_keeps_only_inband_fraction(self):
        # v = 72.5: iv_f = 7.25 -> iv0 = 7 (last bin), fv = 0.25. The (1-fv)
        # share stays in bin 7; the fv share targets bin 8 and is dropped.
        I_map = torch.zeros(NPIX, NPIX, dtype=torch.float64)
        I_map[1, 1] = 1.0
        v = torch.full((1, NPIX, NPIX), 72.5, dtype=torch.float64)
        out = run_scatter(v, I_map)
        assert out[NV - 1, 1, 1].item() == 0.75
        assert out.sum().item() == 0.75
        assert torch.all(out[: NV - 1] == 0)

    @pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-6), (torch.float64, 1e-12)])
    def test_flux_conservation_inband(self, dtype, tol):
        g = torch.Generator().manual_seed(11)
        K = 4
        # Strictly in-band for both target bins: iv_f in [0, Nv-1).
        v_sub = VEL0 + 0.5 + torch.rand(K, NPIX, NPIX, generator=g, dtype=dtype) \
            * (DV * (NV - 1) - 1.0)
        I_map = torch.rand(NPIX, NPIX, generator=g, dtype=dtype) + 0.1
        out = run_scatter(v_sub, I_map, dtype=dtype)
        assert out.sum().item() == pytest.approx(I_map.sum().item(), rel=tol)

    def test_autograd_gradients_analytic(self):
        # Channel-index weighted loss: L = sum_c c * out[c] = I * iv_f for a
        # single in-band subsample, so dL/dv = I/dv and dL/dI = (v-vel0)/dv.
        I_map = torch.zeros(NPIX, NPIX, dtype=torch.float64)
        I_map[0, 0] = 2.0
        I_map = I_map.requires_grad_(True)
        v = torch.tensor([[[47.3, 33.0], [12.5, 61.0]]], dtype=torch.float64,
                         requires_grad=True)
        out = run_scatter(v, I_map)
        w = torch.arange(NV, dtype=torch.float64).view(NV, 1, 1)
        loss = (out * w).sum()
        gv, gI = torch.autograd.grad(loss, (v, I_map))
        assert torch.isfinite(gv).all() and torch.isfinite(gI).all()
        assert gv[0, 0, 0].item() == pytest.approx(2.0 / DV, rel=1e-12)
        assert gI[0, 0].item() == pytest.approx(47.3 / DV, rel=1e-12)
        # Plain total-flux loss: in-band flux is conserved, so d(sum)/dI = 1
        # everywhere and d(sum)/dv = 0.
        I2 = torch.ones(NPIX, NPIX, dtype=torch.float64, requires_grad=True)
        out2 = run_scatter(v.detach(), I2)
        (gI2,) = torch.autograd.grad(out2.sum(), (I2,))
        assert torch.allclose(gI2, torch.ones_like(gI2), atol=1e-14)

    def test_vmap_matches_loop(self):
        g = torch.Generator().manual_seed(3)
        B, K = 3, 4
        vb = VEL0 + 5.0 + torch.rand(B, K, NPIX, NPIX, generator=g,
                                     dtype=torch.float64) * 60.0
        Ib = torch.rand(B, NPIX, NPIX, generator=g, dtype=torch.float64)

        def fn(v_sub, I_map):
            return run_scatter(v_sub, I_map)

        out_vmap = torch.func.vmap(fn)(vb, Ib)
        out_loop = torch.stack([fn(vb[i], Ib[i]) for i in range(B)])
        assert torch.allclose(out_vmap, out_loop, atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
# Lensed renderers (lensed_cube imports caustics at module scope)
# ---------------------------------------------------------------------------
def _lensed_cube_module():
    pytest.importorskip("caustics", reason="lensed_cube requires caustics")
    from supermage.simulators import lensed_cube
    return lensed_cube


class NullLens:
    """Identity lens: beta = theta."""

    def raytrace(self, x, y):
        return x, y


class BadCenterLens:
    """Identity lens with a few central pixels raytraced to inf/nan."""

    def raytrace(self, x, y):
        bx, by = x.clone(), y.clone()
        c = x.shape[0] // 2
        bx[c, c] = float("inf")
        by[c, c] = float("inf")
        bx[c, c - 1] = float("inf")
        by[c - 1, c] = float("nan")
        return bx, by


class AllNaNLens:
    def raytrace(self, x, y):
        nan = float("nan")
        return torch.full_like(x, nan), torch.full_like(y, nan)


FREQS = make_freq_axis(n_chan=6, span_kms=400.0)
BASE_VALS = dict(
    scale=10.0, m_gas=8.0, inclination=1.0, sky_rot=0.5,
    line_broadening=20.0, velocity_shift=5.0, x0=0.02, y0=-0.03,
    distance_pc=1.0e7,
)
SIM_KW = dict(K_vel=4, oversamp_xy=2, oversamp_v=2, device="cpu",
              dtype=torch.float64, line=LINE_HZ)


def _make_disk_models(dtype=torch.float64):
    """Fresh intensity+velocity models (each renderer re-points .inc)."""
    from supermage.simulators.intensity_models import ExponentialDisk2D
    from supermage.simulators.velocity_models import GasSelfGrav

    intensity = ExponentialDisk2D()
    velocity = GasSelfGrav(intensity, "cpu", dtype)
    return intensity, velocity


def _build_analytic_inverse(n_pix=8, **overrides):
    from supermage.simulators.analytic_cube import AnalyticInverse

    kw = {**SIM_KW, **overrides}
    intensity, velocity = _make_disk_models(kw["dtype"])
    return AnalyticInverse(intensity, velocity, FREQS, 0.1, n_pix, **kw)


def _build_analytic_lens(lens, n_pix=8, **overrides):
    lc = _lensed_cube_module()
    kw = {**SIM_KW, **overrides}
    intensity, velocity = _make_disk_models(kw["dtype"])
    return lc.AnalyticLens(lens, intensity, velocity, FREQS, 0.1, n_pix, **kw)


def _theta(sim, **overrides):
    vals = {**BASE_VALS, **overrides}
    return flat_theta(sim, vals, device="cpu", dtype=torch.float64)


class TestAnalyticLens:
    def test_identity_lens_matches_analytic_inverse(self):
        sim_u = _build_analytic_inverse()
        sim_l = _build_analytic_lens(NullLens())
        cube_u = sim_u.forward(_theta(sim_u))
        cube_l = sim_l.forward(_theta(sim_l))
        assert cube_u.shape == cube_l.shape == (6, 8, 8)
        assert cube_u.sum().item() > 0.1  # flux actually landed in band
        assert torch.allclose(cube_l, cube_u, atol=1e-14, rtol=0.0)

    def test_lens_center_nan_guard(self):
        # REGRESSION (ID141 fix): inf/nan raytraced pixels must be zeroed,
        # not allowed to NaN-poison the cube.
        sim_id = _build_analytic_lens(NullLens())
        sim_bad = _build_analytic_lens(BadCenterLens())
        cube_id = sim_id.forward(_theta(sim_id))
        cube_bad = sim_bad.forward(_theta(sim_bad))
        assert torch.isfinite(cube_bad).all()
        flux_id = cube_id.sum().item()
        flux_bad = cube_bad.sum().item()
        # Bad pixels carried flux under the identity lens, so the guarded
        # cube has slightly LESS total flux — but only slightly (3 hi-res
        # pixels out of 256 were zeroed).
        assert 0.0 < flux_bad < flux_id
        assert flux_bad > 0.9 * flux_id

    def test_all_nan_lens_gives_finite_zero_cube(self):
        sim = _build_analytic_lens(AllNaNLens())
        cube = sim.forward(_theta(sim))
        assert torch.isfinite(cube).all()
        assert torch.all(cube == 0)

    def test_x0_is_source_plane_offset(self):
        # With the identity lens, x0 acts exactly like AnalyticInverse's
        # sky offset (beta == theta).
        x0_shift = 0.15
        sim_l = _build_analytic_lens(NullLens())
        sim_u = _build_analytic_inverse()
        cube_l = sim_l.forward(_theta(sim_l, x0=x0_shift))
        cube_u = sim_u.forward(_theta(sim_u, x0=x0_shift))
        assert torch.allclose(cube_l, cube_u, atol=1e-14, rtol=0.0)
        # ... and the shift genuinely changes the cube.
        cube_l0 = sim_l.forward(_theta(sim_l, x0=0.0))
        assert not torch.allclose(cube_l, cube_l0, atol=1e-6, rtol=0.0)

    def test_chunk_v_matches_unchunked(self):
        # Small grid: N_pix_hi = 16 <= the 64-px tile, so the chunked path
        # processes one full-grid tile and must reproduce chunk_v=None.
        sim_ref = _build_analytic_lens(NullLens())
        sim_chunk = _build_analytic_lens(NullLens(), chunk_v=2)
        cube_ref = sim_ref.forward(_theta(sim_ref))
        cube_chunk = sim_chunk.forward(_theta(sim_chunk))
        assert cube_ref.sum().item() > 0.1
        assert torch.allclose(cube_chunk, cube_ref, atol=1e-14, rtol=0.0)

    @pytest.mark.xfail(
        raises=RuntimeError,
        reason="BUG: AnalyticLens chunk_v tiling passes full-grid "
        "Y_flat/X_flat/N_pix_hi/hw to scatter_quantiles_along_v for "
        "sub-grid tiles; any hi-res grid larger than the 64-px tile "
        "(N_pix_hi > 64) crashes with a shape mismatch "
        "(idx0 = iv0c*hw + baseY*N_pix_hi + baseX cannot broadcast).",
    )
    def test_chunk_v_multitile_matches_unchunked(self):
        # N_pix_x=40 * oversamp_xy=2 -> N_pix_hi=80 > tile=64: the tiling
        # loop actually splits the grid. Correct behavior: same cube as
        # chunk_v=None. Currently raises RuntimeError (see reason).
        sim_ref = _build_analytic_lens(NullLens(), n_pix=40)
        sim_chunk = _build_analytic_lens(NullLens(), n_pix=40, chunk_v=2)
        cube_ref = sim_ref.forward(_theta(sim_ref))
        cube_chunk = sim_chunk.forward(_theta(sim_chunk))
        assert torch.allclose(cube_chunk, cube_ref, atol=1e-12, rtol=0.0)


# ---------------------------------------------------------------------------
# CubeLens (caustics Pixelated resampling)
# ---------------------------------------------------------------------------
class _StubSourceCube:
    """Minimal source-cube simulator: fixed cube + device/dtype attrs."""

    device = "cpu"
    dtype = torch.float32

    def __init__(self, cube):
        self.cube = cube

    def forward(self):
        return self.cube


class TestCubeLens:
    def _build(self, src_cube, pixels_x_source=16, pixels_x_lens=8):
        lc = _lensed_cube_module()
        caustics = pytest.importorskip("caustics")
        cosmo = caustics.FlatLambdaCDM(name="cosmo")
        sie = caustics.SIE(cosmology=cosmo, z_l=0.5, z_s=1.0, x0=0.0, y0=0.0,
                           q=0.7, phi=0.3, Rein=0.0, name="sie")
        sim = lc.CubeLens(
            sie, _StubSourceCube(src_cube),
            pixelscale_source=0.05, pixelscale_lens=0.05,
            pixels_x_source=pixels_x_source, pixels_x_lens=pixels_x_lens,
            upsample_factor=2,
        )
        # Pixelated's x0/y0 default to None (dynamic); freeze at the origin.
        sim.src.x0.to_static(torch.tensor(0.0))
        sim.src.y0.to_static(torch.tensor(0.0))
        return sim

    def test_zero_deflection_sie_equals_plain_resampling(self):
        g = torch.Generator().manual_seed(7)
        src = torch.rand(3, 16, 16, generator=g)
        sim = self._build(src)
        lensed = sim.forward(lens_source=True)
        unlensed = sim.forward(lens_source=False)
        assert lensed.shape == (3, 8, 8)
        assert torch.isfinite(lensed).all()
        assert unlensed.abs().sum().item() > 0
        # Rein=0 -> zero deflection -> identical Pixelated resampling.
        assert torch.allclose(lensed, unlensed, atol=1e-6, rtol=0.0)

    def test_constant_source_resamples_to_constant(self):
        # The 8*0.05" lens-plane grid lies strictly inside the 16*0.05"
        # source extent, so interpolating a constant cube returns the
        # constant on every output pixel.
        const = 2.5
        src = torch.full((2, 16, 16), const)
        sim = self._build(src)
        out = sim.forward(lens_source=True)
        assert out.shape == (2, 8, 8)
        assert torch.allclose(out, torch.full_like(out, const), atol=1e-5)


# ---------------------------------------------------------------------------
# gaussian_quantile_offsets (lensed_cube) vs gaussian_quantile_offsets_flex
# ---------------------------------------------------------------------------
class TestGaussianQuantileOffsets:
    def _reference(self, K):
        # Independent reference: inverse normal CDF at mid-quantiles.
        from scipy.special import ndtri

        return ndtri((np.arange(K) + 0.5) / K)

    def test_scalar_sigma_matches_flex_and_scipy(self):
        lc = _lensed_cube_module()
        from supermage.simulators.analytic_cube import gaussian_quantile_offsets_flex

        K = 8
        sigma = torch.tensor(23.0, dtype=torch.float64)
        got = lc.gaussian_quantile_offsets(sigma, K, device="cpu",
                                           dtype=torch.float64)
        flex = gaussian_quantile_offsets_flex(sigma, K, device="cpu",
                                              dtype=torch.float64)
        assert got.shape == (K, 1, 1)
        assert torch.equal(got, flex)
        expect = 23.0 * self._reference(K)
        assert np.allclose(got.squeeze().numpy(), expect, atol=1e-12)

    def test_per_pixel_sigma_matches_flex_and_scipy(self):
        lc = _lensed_cube_module()
        from supermage.simulators.analytic_cube import gaussian_quantile_offsets_flex

        K, H, W = 5, 2, 3
        g = torch.Generator().manual_seed(19)
        sigma = torch.rand(H, W, generator=g, dtype=torch.float64) * 30.0 + 1.0
        got = lc.gaussian_quantile_offsets(sigma, K, device="cpu",
                                           dtype=torch.float64)
        flex = gaussian_quantile_offsets_flex(sigma, K, device="cpu",
                                              dtype=torch.float64)
        assert got.shape == (K, H, W)
        assert torch.equal(got, flex)
        expect = self._reference(K)[:, None, None] * sigma.numpy()[None]
        assert np.allclose(got.numpy(), expect, atol=1e-12)
