"""Tests for supermage/simulators/visibility_cube.py.

Covers: Hz validation, constructor shape/parity checks, total-flux
normalization (DC-sum invariant), the fftshift/ifftshift centering
convention (point-source flatness + shift-theorem phase gradient),
Hermitian symmetry, half-plane slab consistency, image-plane taper
multiplication (against an independent numpy FFT), primary-beam
behavior, zero-padding invariance, JointVisibilityCube parity with
VisibilityCubePadded, and the _expected_mask_shape helper.

Everything runs on CPU in float64 so tolerances can be tight.
"""
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from caskade import Module, forward

from conftest import make_freq_axis, flat_theta, LINE_HZ

from supermage.simulators.visibility_cube import (
    VisibilityCubePadded,
    JointVisibilityCube,
    _check_hz,
    _expected_mask_shape,
)
from supermage.utils.doppler_velocities import create_velocity_grid_stable

# ---------------------------------------------------------------------------
# Shared configuration
# ---------------------------------------------------------------------------
N_CHAN = 4
S = 15          # small (source) cube side, odd
NPIX = 21       # padded uv-grid side, odd
DTYPE = torch.float64
FREQS = make_freq_axis(n_chan=N_CHAN, span_kms=600.0)  # Hz, ascending
# Radio-convention velocity is linear in f, so the channel width is exactly
# span / (n_chan - 1) km/s — an analytic value independent of the code path.
DV_KMS = 600.0 / (N_CHAN - 1)


class StubCube(Module):
    """Caskade stub cube simulator returning a fixed positive cube."""

    def __init__(self, cube):
        super().__init__()
        self.device = cube.device
        self.dtype = cube.dtype
        self.N_pix = cube.shape[-1]
        self.cube_data = cube
        # A param-less leaf module never gets update_graph() triggered by a
        # link event, leaving caskade internals (subgraph_kwargs) unset.
        self.update_graph()

    @forward
    def forward(self):
        return self.cube_data


class PlainStub:
    """Plain (non-caskade) duck-typed cube simulator."""

    def __init__(self, cube):
        self.device = cube.device
        self.dtype = cube.dtype
        self.N_pix = cube.shape[-1]
        self.cube_data = cube

    def forward(self):
        return self.cube_data


def make_cube(n_chan=N_CHAN, side=S, seed=1234):
    g = torch.Generator().manual_seed(seed)
    return torch.rand((n_chan, side, side), generator=g, dtype=DTYPE) + 0.1


def delta_cube(row, col, n_chan=N_CHAN, side=S):
    cube = torch.zeros((n_chan, side, side), dtype=DTYPE)
    cube[:, row, col] = 1.0
    return cube


def full_mask(n_chan=N_CHAN, npix=NPIX):
    return np.ones((n_chan, npix, npix), dtype=bool)


def build(cube=None, npix=NPIX, mask=None, dish=None, taper=None, half=False,
          pixelscale=0.5, freqs=FREQS, stub_cls=StubCube, **kw):
    cube = make_cube() if cube is None else cube
    if mask is None:
        if half:
            mask = np.ones((len(freqs), (npix + 1) // 2, npix), dtype=bool)
        else:
            mask = full_mask(len(freqs), npix)
    return VisibilityCubePadded(
        stub_cls(cube), mask, freqs, npix, pixelscale,
        dish_diameter=dish, image_taper_map=taper, output_half_plane=half,
        **kw,
    )


def theta(flux):
    return torch.tensor([flux], dtype=DTYPE)


def dv_abs(freqs=FREQS):
    """|dv| via create_velocity_grid_stable (the code's own dv source)."""
    _, steps = create_velocity_grid_stable(
        f_start=freqs[0], f_end=freqs[-1], num_points=len(freqs),
        target_dtype=DTYPE, device="cpu", line=LINE_HZ,
    )
    return float(torch.abs(steps[0]))


# ---------------------------------------------------------------------------
# 1. _check_hz
# ---------------------------------------------------------------------------
def test_check_hz():
    with pytest.raises(ValueError):
        _check_hz(230.538, "f")            # GHz value
    _check_hz(230.538e9, "f")              # Hz passes silently
    with pytest.raises(ValueError):
        _check_hz(1e14, "f")               # too large


def test_velocity_step_matches_analytic():
    # sanity-pin the dv used by the normalization tests below
    assert np.isclose(dv_abs(), DV_KMS, rtol=1e-12)


# ---------------------------------------------------------------------------
# 2. Constructor validation
# ---------------------------------------------------------------------------
def test_ctor_mask_shape_mismatch_raises():
    bad = np.ones((N_CHAN, NPIX, NPIX - 1), dtype=bool)
    with pytest.raises(ValueError, match="mask shape"):
        build(mask=bad)


def test_ctor_parity_mismatch_raises():
    with pytest.raises(ValueError, match="[Pp]arity"):
        build(cube=make_cube(side=14), npix=21)


def test_ctor_small_cube_larger_than_npix_raises():
    with pytest.raises(ValueError, match="N_pix"):
        build(cube=make_cube(side=25), npix=21)


def test_ctor_half_plane_requires_odd_npix():
    with pytest.raises(ValueError, match="odd npix"):
        build(cube=make_cube(side=14), npix=20, half=True)


def test_ctor_half_plane_mask_must_be_slab():
    # full-plane mask with output_half_plane=True must be rejected
    with pytest.raises(ValueError, match="mask shape"):
        build(mask=full_mask(), half=True)


def test_ctor_ghz_freqs_raise():
    with pytest.raises(ValueError, match="Hz"):
        build(freqs=FREQS / 1e9)


def test_ctor_ghz_line_raises():
    with pytest.raises(ValueError, match="Hz"):
        build(line=230.538)


# ---------------------------------------------------------------------------
# 3. Flux normalization
# ---------------------------------------------------------------------------
def test_flux_normalization_dc_sum():
    """Sum over channels of the DC visibility == flux / |dv| (dv analytic)."""
    sim = build(dish=None)
    flux = 3.25
    vis = sim.forward(theta(flux))
    c = NPIX // 2
    dc_sum = vis[:, c, c].sum()
    assert abs(float(dc_sum.imag)) < 1e-12 * abs(float(dc_sum.real))
    assert np.isclose(float(dc_sum.real), flux / DV_KMS, rtol=1e-5)
    # tighter check against the code's own dv source
    assert np.isclose(float(dc_sum.real), flux / dv_abs(), rtol=1e-12)


def test_flux_scales_output_linearly():
    sim = build(dish=None)
    v1 = sim.forward(theta(1.0))
    v2 = sim.forward(theta(3.7))
    assert torch.allclose(v2, 3.7 * v1, rtol=1e-12, atol=0.0)


# ---------------------------------------------------------------------------
# 4. Point source: centering convention + shift theorem
# ---------------------------------------------------------------------------
def test_point_source_at_center_gives_flat_real_visibilities():
    c_img = S // 2
    sim = build(cube=delta_cube(c_img, c_img), dish=None)
    vis = sim.forward(theta(1.0))
    amp = vis.abs()
    for ch in range(N_CHAN):
        a = amp[ch]
        assert float(a.std() / a.mean()) < 1e-5
        # per-channel amplitude = flux / |dv| / N_chan (equal split of a
        # channel-uniform hot pixel)
        assert np.isclose(float(a.mean()), 1.0 / DV_KMS / N_CHAN, rtol=1e-10)
        # imaginary part ~ 0 pins the ifftshift centering: the hot pixel at
        # image index npix//2 IS the FFT phase center
        assert float(vis[ch].imag.abs().max()) < 1e-12 * float(a.mean())


def test_point_source_offset_gives_linear_phase_gradient():
    """+1 pixel in x (last axis) -> phase slope -2*pi/npix along u."""
    c_img = S // 2
    sim = build(cube=delta_cube(c_img, c_img + 1), dish=None)
    vis = sim.forward(theta(1.0))
    row = vis[0, NPIX // 2, :].detach().numpy()      # v = 0 row
    # magnitude unchanged by a pure shift
    mags = np.abs(row)
    assert float(mags.std() / mags.mean()) < 1e-10
    phase = np.unwrap(np.angle(row))
    slope = np.polyfit(np.arange(NPIX), phase, 1)[0]
    # F(x - shift) convention here: fftshift(fft2(ifftshift(.))) with the
    # delta at center+1 gives V(u) = exp(-2*pi*i*u*shift/npix)
    assert np.isclose(slope, -2 * np.pi / NPIX, rtol=1e-8)
    # phase is exactly zero at u = 0
    assert abs(float(np.angle(row[NPIX // 2]))) < 1e-10


# ---------------------------------------------------------------------------
# 5. Hermitian symmetry (full plane, odd npix)
# ---------------------------------------------------------------------------
def test_hermitian_symmetry_full_plane():
    sim = build(dish=None)
    vis = sim.forward(theta(1.0))
    mirrored = torch.flip(vis, dims=(-2, -1)).conj()
    assert torch.allclose(vis, mirrored, rtol=1e-10, atol=1e-14)


# ---------------------------------------------------------------------------
# 6. Half-plane consistency
# ---------------------------------------------------------------------------
def test_half_plane_equals_full_plane_rows():
    cube = make_cube()
    full = build(cube=cube, dish=None)
    half_mask = full_mask()[:, NPIX // 2:, :]
    half = build(cube=cube, dish=None, mask=half_mask, half=True)
    v_full = full.forward(theta(2.0))
    v_half = half.forward(theta(2.0))
    assert v_half.shape == (N_CHAN, (NPIX + 1) // 2, NPIX)
    assert torch.allclose(v_half, v_full[:, NPIX // 2:, :], rtol=1e-13, atol=0.0)


# ---------------------------------------------------------------------------
# 7. Image-plane taper
# ---------------------------------------------------------------------------
def test_uniform_half_taper_halves_output():
    cube = make_cube()
    plain = build(cube=cube, dish=None)
    tapered = build(cube=cube, dish=None, taper=0.5 * np.ones((NPIX, NPIX)))
    v0 = plain.forward(theta(1.0))
    v1 = tapered.forward(theta(1.0))
    assert torch.allclose(v1, 0.5 * v0, rtol=1e-13, atol=0.0)


def test_hann_taper_matches_independent_numpy_fft():
    """Taper must be MULTIPLIED into the padded image before the FFT."""
    cube = make_cube()
    hann = np.outer(np.hanning(NPIX), np.hanning(NPIX))
    sim = build(cube=cube, dish=None, taper=hann)
    flux = 1.0
    vis = sim.forward(theta(flux)).detach().numpy()

    # independent numpy reference
    img = cube.numpy() * flux / dv_abs() / cube.numpy().sum()
    pad = (NPIX - S) // 2
    img = np.pad(img, ((0, 0), (pad, pad), (pad, pad)))
    img = img * hann[None, :, :]
    ref = np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(img, axes=(-2, -1)), axes=(-2, -1)),
        axes=(-2, -1),
    )
    assert np.allclose(vis, ref, rtol=1e-10, atol=1e-12 * np.abs(ref).max())


def test_wrong_taper_shape_raises():
    with pytest.raises(ValueError, match="image_taper_map"):
        build(dish=None, taper=np.ones((S, S)))  # small-grid shape is wrong


# ---------------------------------------------------------------------------
# 8. Primary beam
# ---------------------------------------------------------------------------
def test_primary_beam_changes_output_but_preserves_flux_norm():
    cube = make_cube()
    # pixelscale=3'' on a 15-pixel grid spans +/-22.5'', comparable to the
    # ~25'' ALMA 12m PB FWHM at 230 GHz -> strong PB variation
    no_pb = build(cube=cube, dish=None, pixelscale=3.0)
    with_pb = build(cube=cube, dish=12.0, pixelscale=3.0)
    v0 = no_pb.forward(theta(1.0))
    v1 = with_pb.forward(theta(1.0))
    assert not torch.allclose(v0, v1, rtol=1e-2)
    # DC-sum flux invariant holds EXACTLY with the PB on: normalization by
    # cube.sum() happens AFTER PB attenuation
    c = NPIX // 2
    dc_sum = v1[:, c, c].sum()
    assert np.isclose(float(dc_sum.real), 1.0 / dv_abs(), rtol=1e-12)
    assert abs(float(dc_sum.imag)) < 1e-12 * abs(float(dc_sum.real))


# ---------------------------------------------------------------------------
# 9. Padding
# ---------------------------------------------------------------------------
def test_padding_preserves_dc_and_sets_shape():
    cube = make_cube()
    sim21 = build(cube=cube, npix=21, dish=None)
    sim25 = build(cube=cube, npix=25, dish=None,
                  mask=full_mask(N_CHAN, 25))
    v21 = sim21.forward(theta(1.5))
    v25 = sim25.forward(theta(1.5))
    assert v21.shape == (N_CHAN, 21, 21)
    assert v25.shape == (N_CHAN, 25, 25)
    # zero-padding never changes the image sum -> per-channel DC identical
    assert torch.allclose(v21[:, 10, 10], v25[:, 12, 12], rtol=1e-13, atol=0.0)


# ---------------------------------------------------------------------------
# 10. JointVisibilityCube
# ---------------------------------------------------------------------------
def make_ds(name="a", npix=NPIX, n_px=S, freqs=FREQS, pixelscale=3.0, **over):
    if "mask" not in over:
        if over.get("output_half_plane", False):
            over["mask"] = np.ones(
                (len(freqs), (npix + 1) // 2, npix), dtype=bool)
        else:
            over["mask"] = full_mask(len(freqs), npix)
    ds = dict(name=name, freqs=freqs, npix=npix, pixelscale=pixelscale,
              N_px=n_px)
    ds.update(over)
    return ds


def test_joint_single_dataset_matches_visibility_cube_padded():
    cube = make_cube()
    vcp = build(cube=cube, dish=12.0, pixelscale=3.0)
    joint = JointVisibilityCube(StubCube(cube), [make_ds()],
                                dish_diameter=12.0, line=LINE_HZ)
    v_ref = vcp.forward(theta(2.5))
    outs = joint.forward(theta(2.5))
    assert isinstance(outs, list) and len(outs) == 1
    assert torch.allclose(outs[0], v_ref, rtol=1e-13, atol=0.0)


def test_joint_two_datasets_independent_fluxes():
    cube = make_cube()
    joint = JointVisibilityCube(
        StubCube(cube), [make_ds("a"), make_ds("b")],
        dish_diameter=12.0, line=LINE_HZ,
    )
    assert [p.name for p in joint.dynamic_params] == ["flux_a", "flux_b"]
    th = flat_theta(joint, {"flux_a": 1.0, "flux_b": 2.0}, dtype=DTYPE)
    outs = joint.forward(th)
    assert len(outs) == 2
    # identical measurement chains -> only the flux Param differs
    assert torch.allclose(outs[1], 2.0 * outs[0], rtol=1e-12, atol=0.0)


def test_joint_downsampled_dataset_keeps_flux_normalization():
    cube = make_cube()  # fine grid 15
    joint = JointVisibilityCube(
        StubCube(cube), [make_ds("c", n_px=9)],
        dish_diameter=None, line=LINE_HZ,
    )
    outs = joint.forward(theta(4.0))
    assert outs[0].shape == (N_CHAN, NPIX, NPIX)
    dc_sum = outs[0][:, NPIX // 2, NPIX // 2].sum()
    # normalization happens after the area downsample -> invariant survives
    assert np.isclose(float(dc_sum.real), 4.0 / dv_abs(), rtol=1e-12)


@pytest.mark.parametrize(
    "ds_kwargs, match",
    [
        (dict(n_px=17), "N_px"),                                   # > fine grid
        (dict(n_px=14), "parity"),                                 # 14 vs 21
        (dict(n_px=14, npix=20, output_half_plane=True), "odd npix"),
        (dict(mask=np.ones((N_CHAN, 20, 20), dtype=bool)), "mask shape"),
        (dict(taper_map=np.ones((S, S))), "taper_map"),
        (dict(freqs=FREQS / 1e9), "Hz"),                           # GHz freqs
    ],
    ids=["Npx_gt_fine", "parity", "half_plane_even", "mask_shape",
         "taper_shape", "ghz_freqs"],
)
def test_joint_validation_errors(ds_kwargs, match):
    cube = make_cube()
    with pytest.raises(ValueError, match=match):
        JointVisibilityCube(StubCube(cube), [make_ds(**ds_kwargs)],
                            dish_diameter=None, line=LINE_HZ)


def test_joint_ghz_line_raises():
    with pytest.raises(ValueError, match="Hz"):
        JointVisibilityCube(StubCube(make_cube()), [make_ds()],
                            dish_diameter=None, line=230.538)


# ---------------------------------------------------------------------------
# 11. _expected_mask_shape
# ---------------------------------------------------------------------------
def test_expected_mask_shape():
    assert _expected_mask_shape(4, 21, False) == (4, 21, 21)
    assert _expected_mask_shape(4, 21, True) == (4, 11, 21)
    assert _expected_mask_shape(3, 25, True) == (3, 13, 25)


# ---------------------------------------------------------------------------
# 12. Plain (non-caskade) cube simulator duck-typing
# ---------------------------------------------------------------------------
def test_plain_object_stub_matches_module_stub():
    cube = make_cube()
    v_mod = build(cube=cube, dish=None).forward(theta(1.0))
    v_plain = build(cube=cube, dish=None, stub_cls=PlainStub).forward(theta(1.0))
    assert torch.allclose(v_plain, v_mod, rtol=0.0, atol=0.0)
