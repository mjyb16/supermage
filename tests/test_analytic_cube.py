"""Tests for the unlensed cube renderers (supermage/simulators/analytic_cube.py).

Covers:
- gaussian_quantile_offsets_flex / make_dv_table helpers against independent
  scipy references,
- AnalyticInverse geometry (projection round trip, point symmetry, face-on
  limit, integer-pixel offset sign convention) and intermediates,
- CloudCatalog sampling/flux bookkeeping,
- CloudRasterizerOversample trilinear flux conservation + out-of-bounds
  dropping (regression: no clamping to edge voxels),
- Monte-Carlo vs analytic renderer cross-validation (slow).
"""
import math

import numpy as np
import pytest
import torch
from scipy.stats import norm

from conftest import LINE_HZ, flat_theta, make_freq_axis
from supermage.simulators.analytic_cube import (
    AnalyticInverse,
    CloudCatalog,
    CloudRasterizerOversample,
    gaussian_quantile_offsets_flex,
    make_dv_table,
)
from supermage.simulators.intensity_models import ExponentialDisk2D
from supermage.simulators.velocity_models import SersicMGE

DEVICE = "cpu"
DTYPE = torch.float32

DIST_MPC = 16.5
DIST_PC = DIST_MPC * 1e6

N_PIX = 33          # odd -> pixel grid has an exact central pixel
PIXSCALE = 0.1      # arcsec / pix
N_CHAN = 16         # even -> velocity channels symmetric about 0
SPAN_KMS = 600.0
K_VEL = 8
OV_XY = 2
OV_V = 2

FREQ = make_freq_axis(n_chan=N_CHAN, span_kms=SPAN_KMS)

# Standard galaxy parameter values (keys = caskade Param names).
BASE_VALS = dict(
    scale=250.0,                    # pc, ExponentialDisk2D
    qintr=0.6,
    m_bh=8.0,                       # log10 Msun
    n=4.0,
    r_e=10.0,                       # arcsec
    intensity_r_e=3.0,              # log10
    inclination=math.radians(60.0),
    sky_rot=0.4,
    line_broadening=10.0,           # km/s
    velocity_shift=0.0,
    x0=0.0,
    y0=0.0,
    distance_pc=DIST_PC,
)


def make_galaxy():
    """Fresh intensity + velocity models (AnalyticInverse re-points inc)."""
    intensity = ExponentialDisk2D()
    velocity = SersicMGE(
        12, distance_Mpc=DIST_MPC, soft=1.0, device=DEVICE, dtype=DTYPE
    )
    return intensity, velocity


def make_sim(**kw):
    intensity, velocity = make_galaxy()
    args = dict(
        K_vel=K_VEL, oversamp_xy=OV_XY, oversamp_v=OV_V,
        device=DEVICE, dtype=DTYPE, line=LINE_HZ,
    )
    args.update(kw)
    return AnalyticInverse(intensity, velocity, FREQ, PIXSCALE, N_PIX, **args)


def sim_theta(sim, **overrides):
    vals = dict(BASE_VALS)
    vals.update(overrides)
    return flat_theta(sim, vals, DEVICE, DTYPE)


def make_catalog(N_clouds=2000, fov_half_pc=400.0, seed=7, **kw):
    intensity, velocity = make_galaxy()
    cat = CloudCatalog(
        intensity, velocity, fov_half_pc, N_clouds, K_VEL,
        None, DIST_PC, seed=seed, device=DEVICE, dtype=DTYPE, **kw
    )
    return cat, intensity


def cat_theta(cat, **overrides):
    vals = dict(BASE_VALS)
    vals.update(overrides)
    return flat_theta(cat, vals, DEVICE, DTYPE)


# ----------------------------------------------------------------------
# 1. gaussian_quantile_offsets_flex
# ----------------------------------------------------------------------
class TestGaussianQuantileOffsetsFlex:
    def test_scalar_sigma_matches_norm_ppf(self):
        K = 8
        sigma = torch.tensor(3.0, dtype=torch.float64)
        off = gaussian_quantile_offsets_flex(
            sigma, K, device=DEVICE, dtype=torch.float64
        )
        assert off.shape == (K, 1, 1)
        # Independent reference: inverse normal CDF at mid-quantiles.
        ref = 3.0 * norm.ppf((np.arange(K) + 0.5) / K)
        np.testing.assert_allclose(
            off.squeeze().numpy(), ref, rtol=1e-12, atol=1e-12
        )
        # Equal-probability offsets sum to zero and are antisymmetric (even K).
        assert abs(off.sum().item()) < 1e-12
        flat = off.squeeze()
        np.testing.assert_allclose(
            flat.numpy(), -flat.flip(0).numpy(), rtol=1e-12, atol=1e-12
        )

    def test_per_pixel_sigma_shape_and_linearity(self):
        K, H, W = 6, 5, 7
        g = torch.Generator().manual_seed(3)
        sigma = torch.rand((H, W), generator=g, dtype=torch.float64) + 0.5
        off = gaussian_quantile_offsets_flex(
            sigma, K, device=DEVICE, dtype=torch.float64
        )
        assert off.shape == (K, H, W)
        ref = norm.ppf((np.arange(K) + 0.5) / K)[:, None, None] * sigma.numpy()
        np.testing.assert_allclose(off.numpy(), ref, rtol=1e-12, atol=1e-12)
        # Linear in sigma: doubling sigma doubles every offset.
        off2 = gaussian_quantile_offsets_flex(
            2.0 * sigma, K, device=DEVICE, dtype=torch.float64
        )
        np.testing.assert_allclose(
            off2.numpy(), 2.0 * off.numpy(), rtol=1e-12, atol=1e-12
        )


# ----------------------------------------------------------------------
# 2. make_dv_table
# ----------------------------------------------------------------------
class TestMakeDvTable:
    def test_shape_and_seed_reproducibility(self):
        N, K = 256, 8
        t1 = make_dv_table(N, K, seed=11, device=DEVICE, dtype=torch.float64)
        t1b = make_dv_table(N, K, seed=11, device=DEVICE, dtype=torch.float64)
        t2 = make_dv_table(N, K, seed=12, device=DEVICE, dtype=torch.float64)
        assert t1.shape == (N, K)
        assert torch.equal(t1, t1b)
        assert not torch.equal(t1, t2)

    def test_column_statistics_standard_normal(self):
        N, K = 4096, 8
        t = make_dv_table(N, K, seed=11, device=DEVICE, dtype=torch.float64)
        assert torch.isfinite(t).all()
        np.testing.assert_allclose(t.mean(0).numpy(), np.zeros(K), atol=0.05)
        np.testing.assert_allclose(t.std(0).numpy(), np.ones(K), atol=0.05)


# ----------------------------------------------------------------------
# 3. AnalyticInverse.forward basics
# ----------------------------------------------------------------------
def test_forward_shape_finite_nonnegative():
    sim = make_sim()
    cube = sim.forward(sim_theta(sim))
    assert cube.shape == (N_CHAN, N_PIX, N_PIX)
    assert torch.isfinite(cube).all()
    assert (cube >= 0).all()
    assert cube.sum() > 0
    # Independent check of the frequency -> velocity plumbing (radio
    # convention, symmetric axis): v0 = +span/2, dv = -span/(n-1).
    assert sim.Nv_lo == N_CHAN
    assert sim.vel0_lo.item() == pytest.approx(SPAN_KMS / 2, rel=1e-6)
    assert sim.dv_lo == pytest.approx(-SPAN_KMS / (N_CHAN - 1), rel=1e-6)


# ----------------------------------------------------------------------
# 4. Point symmetry of an axisymmetric disk
# ----------------------------------------------------------------------
def test_point_symmetry_flux_and_cube():
    # x0 = y0 = 0, velocity_shift = 0, symmetric velocity grid: the cube must
    # be invariant under (v -> -v) x (180 deg sky rotation).
    sim = make_sim()
    cube = sim.forward(sim_theta(sim, x0=0.0, y0=0.0, velocity_shift=0.0))
    flux = cube.sum(dim=(1, 2))
    peak = cube.max().item()
    # per-channel fluxes mirror-symmetric about the line
    assert torch.allclose(flux, flux.flip(0), rtol=1e-4, atol=1e-4 * peak)
    # full cube: channel c equals the 180deg rotation of channel Nv-1-c
    rot = cube.flip(0).flip(1).flip(2)
    assert torch.allclose(cube, rot, atol=1e-3 * peak)
    # non-vacuous: individual channels are NOT left-right symmetric
    assert not torch.allclose(cube, cube.flip(2), atol=1e-3 * peak)


# ----------------------------------------------------------------------
# 5. return_intermediates: recompute v_los and I_map independently
# ----------------------------------------------------------------------
def test_return_intermediates_physics():
    sim = make_sim()
    velocity_model = sim.velocity_model
    theta = sim_theta(sim)
    cube, inter = sim.forward(theta, return_intermediates=True)

    N_hi = N_PIX * OV_XY
    for key in ("I_map", "v_los", "R", "x_gal", "y_gal", "x_sky", "y_sky"):
        assert inter[key].shape == (N_hi, N_hi), key

    R = inter["R"]
    # I(R) = exp(-R / scale)
    I_ref = torch.exp(-R / BASE_VALS["scale"])
    assert torch.allclose(inter["I_map"], I_ref, rtol=1e-6, atol=1e-8)

    # v_los = v_circ(R) * sin(i) * cos(theta) + velocity_shift,
    # with the rotation curve evaluated through the velocity model directly.
    # (After AnalyticInverse re-points inc, the velocity model's inclination
    # Param is the simulator's, named "inclination".)
    assert [p.name for p in velocity_model.dynamic_params] == [
        "inclination", "qintr", "m_bh", "n", "r_e", "intensity_r_e"
    ]
    theta_v = flat_theta(velocity_model, BASE_VALS, DEVICE, DTYPE)
    v_circ = velocity_model.velocity(theta_v, R_flat=R)
    inc = BASE_VALS["inclination"]
    v_ref = (
        v_circ * math.sin(inc) * inter["x_gal"] / (R + 1e-12)
        + BASE_VALS["velocity_shift"]
    )
    assert torch.allclose(inter["v_los"], v_ref, rtol=1e-5, atol=1e-3)
    # sanity: the disk actually rotates (a big chunk of the band is used)
    assert inter["v_los"].abs().max() > 100.0


# ----------------------------------------------------------------------
# 6. _sky_to_intrinsic round trip
# ----------------------------------------------------------------------
def test_sky_to_intrinsic_round_trip():
    sim = make_sim()
    g = torch.Generator().manual_seed(5)
    n = 200
    x_gal = (torch.rand(n, generator=g, dtype=torch.float64) - 0.5) * 400.0
    y_gal = (torch.rand(n, generator=g, dtype=torch.float64) - 0.5) * 400.0

    inc = torch.tensor(math.radians(55.0), dtype=torch.float64)
    pa = torch.tensor(0.7 + math.pi / 2.0, dtype=torch.float64)
    cos_i = torch.cos(inc)
    arcsec_per_pc = torch.tensor(206265.0 / DIST_PC, dtype=torch.float64)
    x0 = torch.tensor(0.3, dtype=torch.float64)
    y0 = torch.tensor(-0.15, dtype=torch.float64)

    # Documented forward projection.
    cos_pa, sin_pa = torch.cos(pa), torch.sin(pa)
    x_sky = (cos_pa * x_gal - sin_pa * (y_gal * cos_i)) * arcsec_per_pc + x0
    y_sky = (sin_pa * x_gal + cos_pa * (y_gal * cos_i)) * arcsec_per_pc + y0

    xg, yg, R = sim._sky_to_intrinsic(
        x_sky, y_sky, x0=x0, y0=y0, pa=pa, cos_i=cos_i,
        arcsec_per_pc=arcsec_per_pc,
    )
    assert torch.allclose(xg, x_gal, rtol=1e-5, atol=1e-7)
    assert torch.allclose(yg, y_gal, rtol=1e-5, atol=1e-7)
    assert torch.allclose(R, torch.hypot(x_gal, y_gal), rtol=1e-5, atol=1e-7)


# ----------------------------------------------------------------------
# 7. Face-on limit
# ----------------------------------------------------------------------
def test_face_on_limit():
    # inclination = 0 -> v_los = velocity_shift everywhere.
    sim = make_sim()
    shift = 37.0
    _, inter = sim.forward(
        sim_theta(sim, inclination=0.0, velocity_shift=shift,
                  line_broadening=1.0),
        return_intermediates=True,
    )
    assert torch.allclose(
        inter["v_los"], torch.full_like(inter["v_los"], shift),
        rtol=0, atol=1e-5,
    )

    # With shift = 0 and line_broadening << channel width (1 vs 40 km/s),
    # all flux lands in the two channels straddling v = 0.
    sim2 = make_sim()
    cube = sim2.forward(
        sim_theta(sim2, inclination=0.0, velocity_shift=0.0,
                  line_broadening=1.0)
    )
    flux = cube.sum(dim=(1, 2))
    total = flux.sum()
    assert total > 0
    central = flux[N_CHAN // 2 - 1] + flux[N_CHAN // 2]
    assert (total - central).abs() < 1e-6 * total


# ----------------------------------------------------------------------
# 8. Integer-pixel offsets: sign convention of x0 / y0
# ----------------------------------------------------------------------
def test_integer_pixel_offset_roll():
    sim = make_sim()
    cube0 = sim.forward(sim_theta(sim))
    peak = cube0.max().item()
    c = 3  # edge crop (torch.roll wraps around)

    # +x0 shifts the image toward +x (east / increasing column index, dim 2).
    cube_x = sim.forward(sim_theta(sim, x0=2 * PIXSCALE))
    expect = torch.roll(cube0, shifts=+2, dims=2)
    assert torch.allclose(
        cube_x[:, :, c:-c], expect[:, :, c:-c], atol=1e-4 * peak
    )
    # The opposite sign must NOT match (establishes the convention).
    wrong = torch.roll(cube0, shifts=-2, dims=2)
    assert not torch.allclose(
        cube_x[:, :, c:-c], wrong[:, :, c:-c], atol=1e-4 * peak
    )

    # +y0 shifts toward +y (north / increasing row index, dim 1).
    cube_y = sim.forward(sim_theta(sim, y0=2 * PIXSCALE))
    expect_y = torch.roll(cube0, shifts=+2, dims=1)
    assert torch.allclose(
        cube_y[:, c:-c, :], expect_y[:, c:-c, :], atol=1e-4 * peak
    )


# ----------------------------------------------------------------------
# 9. CloudCatalog
# ----------------------------------------------------------------------
class TestCloudCatalog:
    def test_subsample_shapes_and_flux_rows(self):
        N = 2000
        cat, intensity = make_catalog(N_clouds=N)
        theta = cat_theta(cat)
        pos_img, vel_chan, flux = cat.forward(theta, return_subsamples=True)
        assert pos_img.shape == (N, K_VEL, 2)
        assert vel_chan.shape == (N, K_VEL)
        assert flux.shape == (N, K_VEL)
        # Every row's flux = brightness(R_cloud) / K, verified against a
        # direct call of the intensity model on the catalogue radii.
        R = torch.hypot(cat.pos_gal0[:, 0], cat.pos_gal0[:, 1])
        theta_i = flat_theta(intensity, BASE_VALS, DEVICE, DTYPE)
        b = intensity.brightness(theta_i, R_map=R)
        ref = (b / K_VEL).unsqueeze(1).expand(-1, K_VEL)
        assert torch.allclose(flux, ref, rtol=1e-6, atol=0)

    def test_same_seed_same_positions(self):
        cat1, _ = make_catalog(N_clouds=512, seed=21)
        cat2, _ = make_catalog(N_clouds=512, seed=21)
        cat3, _ = make_catalog(N_clouds=512, seed=22)
        assert torch.equal(cat1.pos_gal0, cat2.pos_gal0)
        assert not torch.equal(cat1.pos_gal0, cat3.pos_gal0)

    @pytest.mark.parametrize("method", ["sobol_uniform", "uniform"])
    def test_positions_within_fov(self, method):
        fov = 350.0
        cat, _ = make_catalog(
            N_clouds=1024, fov_half_pc=fov, sampling_method=method
        )
        assert cat.pos_gal0.shape == (1024, 2)
        assert (cat.pos_gal0 >= -fov).all()
        assert (cat.pos_gal0 <= fov).all()
        # non-degenerate: points actually spread over the square
        assert cat.pos_gal0.abs().max() > 0.5 * fov

    def test_unknown_sampling_method_raises(self):
        with pytest.raises(ValueError):
            make_catalog(N_clouds=16, sampling_method="not_a_method")

    def test_dict_return_path(self):
        cat, _ = make_catalog(N_clouds=256)
        theta = cat_theta(cat)
        tup = cat.forward(theta, return_subsamples=True)
        d = cat.forward(theta, return_subsamples=False)
        assert set(d.keys()) == {"pos_img", "vel_chan", "flux"}
        assert torch.equal(d["pos_img"], tup[0])
        assert torch.equal(d["vel_chan"], tup[1])
        assert torch.equal(d["flux"], tup[2])

    def test_sobol_jitter_path_total_flux(self):
        cat, _ = make_catalog(N_clouds=512)
        theta = cat_theta(cat)
        _, vel_q, flux_q = cat.forward(theta, return_subsamples=True)
        _, vel_s, flux_s = cat.forward(
            theta, return_subsamples=True, gaussian_quantile=False
        )
        # Total flux identical between the two velocity-jitter schemes.
        assert torch.allclose(flux_s.sum(), flux_q.sum(), rtol=1e-6)
        assert flux_s.shape == flux_q.shape
        # but the jitter itself differs (stochastic table vs quantiles)
        assert not torch.allclose(vel_s, vel_q)


# ----------------------------------------------------------------------
# 10. CloudRasterizerOversample flux accounting
# ----------------------------------------------------------------------
def make_rasterizer(cat):
    return CloudRasterizerOversample(
        cat, FREQ, PIXSCALE, N_PIX, oversamp_xy=OV_XY, oversamp_v=OV_V,
        device=DEVICE, dtype=DTYPE, line=LINE_HZ,
    )


class TestCloudRasterizerOversample:
    def test_single_cloud_conserves_flux(self):
        cat, _ = make_catalog(N_clouds=16)
        rr = make_rasterizer(cat)
        t = lambda v: torch.tensor([v], device=DEVICE, dtype=DTYPE)
        # well inside the FOV (half-width 1.65") and the band (+-300 km/s)
        cube_hi = rr._rasterise_hi(t(0.3), t(-0.2), t(25.0), t(2.5))
        assert cube_hi.shape == (N_CHAN * OV_V, N_PIX * OV_XY, N_PIX * OV_XY)
        # trilinear weights sum to 1 -> total deposited flux == input flux
        assert cube_hi.sum().item() == pytest.approx(2.5, rel=1e-5)
        # exactly 8 (or fewer) voxels touched
        assert (cube_hi > 0).sum() <= 8

    def test_out_of_bounds_clouds_dropped(self):
        # REGRESSION: out-of-FOV / out-of-band subsamples must contribute
        # exactly nothing (weights zeroed), not be clamped into edge voxels.
        cat, _ = make_catalog(N_clouds=16)
        rr = make_rasterizer(cat)
        ra = torch.tensor([50.0, -50.0, 0.0, 0.0, 1.7], device=DEVICE, dtype=DTYPE)
        dec = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], device=DEVICE, dtype=DTYPE)
        vel = torch.tensor([0.0, 0.0, 9999.0, -9999.0, 0.0], device=DEVICE, dtype=DTYPE)
        flx = torch.ones(5, device=DEVICE, dtype=DTYPE)
        cube_hi = rr._rasterise_hi(ra, dec, vel, flx)
        assert cube_hi.sum().item() == 0.0
        assert (cube_hi != 0).sum().item() == 0

    def test_box_filter_flux_accounting(self):
        cat, _ = make_catalog(N_clouds=2000)
        rr = make_rasterizer(cat)
        theta = flat_theta(rr, BASE_VALS, DEVICE, DTYPE)
        cube_lo = rr.forward(theta)
        assert cube_lo.shape == (N_CHAN, N_PIX, N_PIX)

        # Rebuild the hi-res deposit from the same catalogue output: the
        # box filter is a plain mean, so sum(lo) * ov_v * ov_xy^2 == sum(hi).
        pos_img, vel_chan, flux = cat.forward(theta, return_subsamples=True)
        M = pos_img.numel() // 2
        cube_hi = rr._rasterise_hi(
            pos_img[..., 0].reshape(M), pos_img[..., 1].reshape(M),
            vel_chan.reshape(M), flux.reshape(M),
        )
        lhs = cube_lo.sum().item() * OV_V * OV_XY**2
        assert lhs == pytest.approx(cube_hi.sum().item(), rel=1e-5)
        # and a nonzero fraction of the catalogue flux actually landed
        assert cube_hi.sum() > 0.1 * flux.sum()


# ----------------------------------------------------------------------
# 11. Monte-Carlo vs analytic renderer cross-validation
# ----------------------------------------------------------------------
@pytest.mark.slow
def test_mc_matches_analytic_renderer():
    # fov_half_pc must cover the deprojected sky FOV: corner radius
    # sqrt(2) * 1.65" * 80 pc/" / cos(60deg) ~ 373 pc -> 400 pc with margin.
    cat, _ = make_catalog(N_clouds=300_000, fov_half_pc=400.0, seed=3)
    rr = make_rasterizer(cat)
    cube_mc = rr.forward(flat_theta(rr, BASE_VALS, DEVICE, DTYPE))

    sim = make_sim()
    cube_an = sim.forward(sim_theta(sim))

    a = cube_mc / cube_mc.sum()
    b = cube_an / cube_an.sum()
    l1 = (a - b).abs().sum().item()
    assert l1 < 0.15, f"normalized L1 = {l1}"
