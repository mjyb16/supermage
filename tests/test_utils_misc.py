"""Tests for supermage.utils: angular_distance, doppler_velocities,
primary_beams, plotting.

All expected values are derived independently (analytic formulas or plain
numpy reference implementations), not by re-running the code under test.
"""
import numpy as np
import pytest
import torch

from conftest import C_KMS, LINE_HZ

from supermage.utils.angular_distance import arcsec_to_parsec, parsec_to_arcsec
from supermage.utils.doppler_velocities import (
    create_velocity_grid_stable,
    freq_to_vel_absolute,
)
from supermage.utils.primary_beams import gaussian_pb
from supermage.utils.plotting import (
    create_pvd,
    dirty_cube_tool,
    rotate_spectral_cube_center_offset_arcsec,
    smooth_mask,
    velocity_map,
    weighted_dirty_cube_tool,
)

C_M_S = 299792458.0  # speed of light [m/s], exact (SI definition)


# ---------------------------------------------------------------------------
# 1. angular_distance
# ---------------------------------------------------------------------------
class TestAngularDistance:
    def test_one_arcsec_at_one_mpc(self):
        # small-angle: s = D * theta ; 1 arcsec at 1 Mpc in pc
        expected_pc = 1e6 * np.radians(1.0 / 3600.0)  # ~4.84813681 pc
        got = arcsec_to_parsec(1.0, 1.0)
        assert np.isclose(got, expected_pc, rtol=1e-10)
        # sanity: the canonical number
        assert np.isclose(got, 4.84813681, rtol=1e-8)

    def test_vectorizes_over_arrays(self):
        d_mpc = 16.5
        ang = np.array([0.1, 1.0, 2.5, 30.0])
        got = arcsec_to_parsec(d_mpc, ang)
        expected = d_mpc * 1e6 * np.radians(ang / 3600.0)
        assert got.shape == ang.shape
        assert np.allclose(got, expected, rtol=1e-12)

    def test_parsec_to_arcsec_is_exact_inverse(self):
        d_mpc = np.array([1.0, 11.4, 87.3])
        ang = np.array([0.05, 1.7, 250.0])
        round_trip = parsec_to_arcsec(d_mpc, arcsec_to_parsec(d_mpc, ang))
        assert np.allclose(round_trip, ang, rtol=1e-12)
        # and the other direction
        pc = np.array([0.3, 48.0, 9e3])
        round_trip_pc = arcsec_to_parsec(d_mpc, parsec_to_arcsec(d_mpc, pc))
        assert np.allclose(round_trip_pc, pc, rtol=1e-12)


# ---------------------------------------------------------------------------
# 2. freq_to_vel_absolute
# ---------------------------------------------------------------------------
class TestFreqToVelAbsolute:
    def test_zero_at_rest_frequency(self):
        f = torch.tensor([LINE_HZ], dtype=torch.float64)
        v = freq_to_vel_absolute(f, rest_frame_freq=LINE_HZ)
        assert torch.abs(v).max().item() == pytest.approx(0.0, abs=1e-12)

    def test_matches_radio_convention_formula(self):
        f = torch.linspace(229.8e9, 231.2e9, 11, dtype=torch.float64)
        v = freq_to_vel_absolute(f, rest_frame_freq=LINE_HZ)
        # independent float64 numpy reference: v = c * (f0 - f) / f0
        expected = C_KMS * (LINE_HZ - f.numpy()) / LINE_HZ
        assert np.allclose(v.numpy(), expected, rtol=1e-12)

    def test_linear_in_frequency(self):
        # v(f) is affine in f -> second differences of a uniform grid vanish
        f = torch.linspace(229.0e9, 232.0e9, 7, dtype=torch.float64)
        v = freq_to_vel_absolute(f, rest_frame_freq=LINE_HZ).numpy()
        second_diff = np.diff(v, n=2)
        assert np.abs(second_diff).max() < 1e-9  # km/s
        # midpoint property
        v_mid = freq_to_vel_absolute(
            torch.tensor((229.0e9 + 232.0e9) / 2, dtype=torch.float64),
            rest_frame_freq=LINE_HZ,
        ).item()
        assert v_mid == pytest.approx((v[0] + v[-1]) / 2, rel=1e-12)

    def test_unit_agnostic(self):
        f_hz = torch.linspace(230.0e9, 231.0e9, 9, dtype=torch.float64)
        v_hz = freq_to_vel_absolute(f_hz, rest_frame_freq=LINE_HZ)
        v_ghz = freq_to_vel_absolute(f_hz / 1e9, rest_frame_freq=LINE_HZ / 1e9)
        assert torch.allclose(v_hz, v_ghz, rtol=1e-12)


# ---------------------------------------------------------------------------
# 3. create_velocity_grid_stable
# ---------------------------------------------------------------------------
class TestCreateVelocityGridStable:
    F_START, F_END, N = 230.3e9, 230.8e9, 41  # ascending, brackets the line

    def _grid(self, dtype=torch.float32):
        return create_velocity_grid_stable(
            self.F_START, self.F_END, self.N,
            target_dtype=dtype, device="cpu", line=LINE_HZ,
        )

    def test_uniform_spacing_in_float32(self):
        v, _ = self._grid()
        diffs = (v[1:] - v[:-1]).to(torch.float64).numpy()
        # the stability guarantee: naive per-channel float32 conversion of
        # ~2.3e11 Hz frequencies would give catastrophic-cancellation jitter
        # of order km/s; the stable grid must be uniform to well under 1e-3.
        assert np.abs(diffs - diffs.mean()).max() < 1e-3

    def test_first_velocity_matches_float64_reference(self):
        v, _ = self._grid()
        v0_ref = freq_to_vel_absolute(
            torch.tensor(self.F_START, dtype=torch.float64),
            rest_frame_freq=LINE_HZ,
        ).item()
        eps32 = float(np.finfo(np.float32).eps)
        assert abs(v[0].item() - v0_ref) <= 2 * eps32 * max(1.0, abs(v0_ref))

    def test_ascending_frequency_gives_negative_dv(self):
        v, dv = self._grid()
        assert (dv < 0).all()
        # magnitude check against the analytic channel width in float64
        df = (self.F_END - self.F_START) / (self.N - 1)
        dv_expected = -C_KMS * df / LINE_HZ
        assert dv.to(torch.float64).mean().item() == pytest.approx(
            dv_expected, rel=1e-5
        )

    def test_length_and_steps_consistency(self):
        v, dv = self._grid()
        assert len(v) == self.N
        assert len(dv) == self.N - 1
        assert torch.equal(dv, v[1:] - v[:-1])

    def test_dtype_and_device_respected(self):
        v32, dv32 = self._grid(torch.float32)
        v64, dv64 = self._grid(torch.float64)
        assert v32.dtype == torch.float32 and dv32.dtype == torch.float32
        assert v64.dtype == torch.float64 and dv64.dtype == torch.float64
        assert v32.device.type == "cpu" and v64.device.type == "cpu"
        # float64 grid should agree with the float32 grid to float32 accuracy
        assert torch.allclose(
            v64.to(torch.float32), v32, atol=1e-3, rtol=0
        )


# ---------------------------------------------------------------------------
# 4. gaussian_pb
# ---------------------------------------------------------------------------
class TestGaussianPB:
    FREQ_HZ = 230.538e9
    SHAPE = (201, 201)
    DELTAL = 0.5  # arcsec/px -> FWHM ~25" ~ 50 px: well resolved

    def test_peak_is_one_at_central_pixel(self):
        pb, _ = gaussian_pb(
            diameter=12, freq_hz=self.FREQ_HZ, shape=self.SHAPE,
            deltal=self.DELTAL, device="cpu",
        )
        cy, cx = self.SHAPE[0] // 2, self.SHAPE[1] // 2
        assert pb[cy, cx].item() == pytest.approx(1.0, abs=1e-14)
        assert pb.max().item() == pytest.approx(1.0, abs=1e-14)

    def test_returned_fwhm_matches_analytic_value(self):
        fwhm_factor = 1.13
        pb, fwhm = gaussian_pb(
            diameter=12, freq_hz=self.FREQ_HZ, shape=self.SHAPE,
            deltal=self.DELTAL, fwhm_factor=fwhm_factor, device="cpu",
        )
        # FWHM = factor * lambda/D, radians -> arcsec (1 rad = 206264.8...")
        rad_to_arcsec = np.degrees(1.0) * 3600.0  # ~206265
        expected = fwhm_factor * (C_M_S / self.FREQ_HZ) / 12.0 * rad_to_arcsec
        assert float(fwhm) == pytest.approx(expected, rel=1e-6)

    def test_measured_fwhm_matches_returned_value(self):
        pb, fwhm = gaussian_pb(
            diameter=12, freq_hz=self.FREQ_HZ, shape=self.SHAPE,
            deltal=self.DELTAL, device="cpu",
        )
        n = self.SHAPE[0]
        half_fov = self.DELTAL * n / 2
        x = np.linspace(-half_fov, half_fov, n)  # the grid the code uses
        row = pb[n // 2, :].numpy()
        above = row >= 0.5
        i_l = int(np.argmax(above))                    # first pixel above half max
        i_r = int(len(row) - 1 - np.argmax(above[::-1]))  # last pixel above
        # linear interpolation of the half-max crossings on each side
        x_l = np.interp(0.5, [row[i_l - 1], row[i_l]], [x[i_l - 1], x[i_l]])
        x_r = np.interp(0.5, [row[i_r + 1], row[i_r]], [x[i_r + 1], x[i_r]])
        measured = x_r - x_l
        assert measured == pytest.approx(float(fwhm), rel=0.03)

    def test_ghz_input_raises(self):
        with pytest.raises(ValueError):
            gaussian_pb(freq_hz=230.538, shape=(11, 11), device="cpu")

    def test_fwhm_halves_when_diameter_doubles(self):
        _, fwhm12 = gaussian_pb(
            diameter=12, freq_hz=self.FREQ_HZ, shape=(11, 11), device="cpu"
        )
        _, fwhm24 = gaussian_pb(
            diameter=24, freq_hz=self.FREQ_HZ, shape=(11, 11), device="cpu"
        )
        assert float(fwhm24) == pytest.approx(float(fwhm12) / 2, rel=1e-12)


# ---------------------------------------------------------------------------
# 5. dirty_cube_tool
# ---------------------------------------------------------------------------
def _forward_vis(img_cube):
    """vis = fftshift(fft2(ifftshift(img))) per channel; img_cube (F, N, N)."""
    vis = np.stack(
        [
            np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img_cube[i])))
            for i in range(img_cube.shape[0])
        ]
    )
    return vis.real.copy(), vis.imag.copy()


class TestDirtyCubeTool:
    def test_round_trip_full_roi(self):
        rng = np.random.default_rng(42)
        F, N = 3, 33
        img = rng.normal(size=(F, N, N))
        re, im = _forward_vis(img)
        dirty = dirty_cube_tool(re, im, 0, N)
        assert dirty.shape == (N, N, F)  # channel axis LAST
        for f in range(F):
            np.testing.assert_allclose(dirty[:, :, f], img[f], atol=1e-10)

    def test_roi_crop_returns_correct_subblock(self):
        rng = np.random.default_rng(7)
        F, N = 2, 33
        img = rng.normal(size=(F, N, N))
        re, im = _forward_vis(img)
        a, b = 5, 20
        dirty = dirty_cube_tool(re, im, a, b)
        assert dirty.shape == (b - a, b - a, F)
        for f in range(F):
            np.testing.assert_allclose(dirty[:, :, f], img[f, a:b, a:b], atol=1e-10)


# ---------------------------------------------------------------------------
# 6. weighted_dirty_cube_tool
# ---------------------------------------------------------------------------
class TestWeightedDirtyCubeTool:
    def test_unit_std_equals_plain_dirty_cube(self):
        rng = np.random.default_rng(3)
        F, N = 2, 17
        img = rng.normal(size=(F, N, N))
        re, im = _forward_vis(img)
        ones = np.ones_like(re)
        plain = dirty_cube_tool(re, im, 0, N)
        weighted = weighted_dirty_cube_tool(re, im, 0, N, ones, ones)
        np.testing.assert_allclose(weighted, plain, atol=1e-12)

    def test_zero_std_cells_are_finite_and_zero_weighted(self):
        rng = np.random.default_rng(4)
        F, N = 2, 17
        img = rng.normal(size=(F, N, N))
        re, im = _forward_vis(img)
        ones = np.ones_like(re)

        # poison one uv cell with a huge value and zero its std
        re_bad, im_bad = re.copy(), im.copy()
        re_bad[0, 3, 4] = 1e8
        im_bad[0, 3, 4] = -1e8
        std_r = ones.copy()
        std_i = ones.copy()
        std_r[0, 3, 4] = 0.0
        std_i[0, 3, 4] = 0.0

        out = weighted_dirty_cube_tool(re_bad, im_bad, 0, N, std_r, std_i)
        assert np.isfinite(out).all()

        # a zero-std cell must contribute (essentially) nothing: result equals
        # the unit-weight image with that cell's visibility zeroed out
        re_ref, im_ref = re.copy(), im.copy()
        re_ref[0, 3, 4] = 0.0
        im_ref[0, 3, 4] = 0.0
        ref = weighted_dirty_cube_tool(re_ref, im_ref, 0, N, ones, ones)
        np.testing.assert_allclose(out, ref, atol=1e-10)


# ---------------------------------------------------------------------------
# 7. velocity_map
# ---------------------------------------------------------------------------
class TestVelocityMap:
    def test_hand_computed_moment1(self):
        velocities = np.array([-100.0, 0.0, 100.0])
        cube = np.zeros((1, 2, 3))
        cube[0, 0] = [1.0, 0.0, 1.0]  # symmetric -> mean v = 0
        cube[0, 1] = [1.0, 0.0, 3.0]  # (-100 + 300)/4 = 50
        vmap = velocity_map(cube, velocities, backend="numpy")
        assert vmap.shape == (1, 2)
        assert vmap[0, 0] == pytest.approx(0.0, abs=1e-12)
        assert vmap[0, 1] == pytest.approx(50.0, rel=1e-12)

    def test_pytorch_backend_matches_numpy(self):
        rng = np.random.default_rng(11)
        cube = rng.uniform(0.1, 1.0, size=(4, 5, 6))
        velocities = np.linspace(-250.0, 250.0, 6)
        v_np = velocity_map(cube, velocities, backend="numpy")
        v_t = velocity_map(
            torch.tensor(cube), torch.tensor(velocities), backend="pytorch"
        )
        np.testing.assert_allclose(v_t.numpy(), v_np, rtol=1e-12)

    def test_zero_intensity_pixel_gives_nan(self):
        velocities = np.array([-100.0, 0.0, 100.0])
        cube = np.zeros((2, 1, 3))
        cube[0, 0] = [1.0, 1.0, 1.0]
        # cube[1, 0] stays all-zero
        with np.errstate(invalid="ignore"):
            vmap = velocity_map(cube, velocities, backend="numpy")
        assert np.isfinite(vmap[0, 0])
        assert np.isnan(vmap[1, 0])


# ---------------------------------------------------------------------------
# 8. create_pvd
# ---------------------------------------------------------------------------
class TestCreatePVD:
    def test_shape_flux_and_location(self):
        n_minor, n_major, n_freq = 7, 9, 5
        cube = np.zeros((n_minor, n_major, n_freq))
        m0, x0, f0, flux = 3, 2, 4, 3.0
        cube[m0, x0, f0] = flux  # inside slit rows [2, 5)
        pvd = create_pvd(cube, 2, 5)
        assert pvd.shape == (n_freq, n_major)
        assert pvd.sum() == pytest.approx(cube[2:5].sum())
        # flip(rot90(sum)) maps (major=x0, freq=f0) -> (f0, n_major-1-x0):
        # rot90 (CCW): B[i, j] = S[j, n_freq-1-i]; flip (both axes):
        # P[i, j] = B[n_freq-1-i, n_major-1-j] = S[n_major-1-j, i]
        assert pvd[f0, n_major - 1 - x0] == pytest.approx(flux)
        assert np.count_nonzero(pvd) == 1

    def test_voxel_outside_slit_excluded(self):
        n_minor, n_major, n_freq = 7, 9, 5
        cube = np.zeros((n_minor, n_major, n_freq))
        cube[3, 2, 4] = 3.0  # inside rows [2, 5)
        cube[6, 5, 1] = 7.0  # OUTSIDE rows [2, 5)
        pvd = create_pvd(cube, 2, 5)
        assert pvd.sum() == pytest.approx(3.0)  # outside voxel excluded
        pvd_full = create_pvd(cube, 0, n_minor)
        assert pvd_full.sum() == pytest.approx(10.0)  # full slit includes it


# ---------------------------------------------------------------------------
# 9. rotate_spectral_cube_center_offset_arcsec
# ---------------------------------------------------------------------------
class TestRotateSpectralCube:
    def test_angle_zero_identity_on_odd_cube(self):
        rng = np.random.default_rng(0)
        cube = rng.normal(size=(11, 11, 3))
        rot, extent = rotate_spectral_cube_center_offset_arcsec(
            cube, 0.0, (0.0, 0.0), pixel_scale=1.0
        )
        # odd cube, centered rotation point -> no padding needed; crop the
        # (possibly padded) center back out and compare
        py = (rot.shape[0] - cube.shape[0]) // 2
        px = (rot.shape[1] - cube.shape[1]) // 2
        core = rot[py:py + 11, px:px + 11, :]
        np.testing.assert_allclose(core, cube, atol=1e-12)

    def test_angle_90_moves_blob_to_expected_pixel(self):
        ny = nx = 11
        cube = np.zeros((ny, nx, 2))
        r0, c0 = 2, 7  # off-center blob
        cube[r0, c0, 0] = 5.0
        rot, _ = rotate_spectral_cube_center_offset_arcsec(
            cube, 90.0, (0.0, 0.0), pixel_scale=1.0, interp_order=1
        )
        assert rot.shape == cube.shape  # odd + centered -> no padding
        # scipy.ndimage.rotate(angle=+90, reshape=False) about the center C
        # maps (row, col) -> (C - (col - C), C + (row - C))  [verified
        # empirically against scipy]; here (2, 7) -> (3, 2)
        C = (nx - 1) // 2
        expected = (C - (c0 - C), C + (r0 - C))
        found = np.unravel_index(np.argmax(rot[:, :, 0]), rot[:, :, 0].shape)
        assert tuple(found) == expected
        assert rot[expected[0], expected[1], 0] == pytest.approx(5.0, rel=1e-12)
        # other channel untouched (stays empty)
        assert np.abs(rot[:, :, 1]).max() == pytest.approx(0.0, abs=1e-12)

    def test_extent_is_ordered_and_scales_with_pixel_scale(self):
        cube = np.zeros((9, 9, 1))
        _, ext1 = rotate_spectral_cube_center_offset_arcsec(
            cube, 30.0, (0.0, 0.0), pixel_scale=1.0
        )
        _, ext2 = rotate_spectral_cube_center_offset_arcsec(
            cube, 30.0, (0.0, 0.0), pixel_scale=2.5
        )
        assert len(ext1) == 4
        x_min, x_max, y_min, y_max = ext1
        assert x_min < x_max and y_min < y_max
        np.testing.assert_allclose(np.asarray(ext2), 2.5 * np.asarray(ext1),
                                   rtol=1e-12)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "BUG: for even-sized cubes the fractional-center padding "
            "bookkeeping is inconsistent — nx_pad = 2*half_width+1 (odd) but "
            "the actual padded array gets nx + pad_left + pad_right = nx+2 "
            "pixels (e.g. 10 -> 12 while nx_pad claims 11), so the reported "
            "extent describes a grid one pixel smaller than the returned "
            "cube. (Related: the half-pixel extent margin uses "
            "(x_max-x_min)/n_pix instead of pixel_scale, shrinking the "
            "extent by ~n/(n+1) even for odd cubes.)"
        ),
    )
    def test_even_cube_extent_consistent_with_returned_shape(self):
        cube = np.zeros((10, 10, 1))
        rot, extent = rotate_spectral_cube_center_offset_arcsec(
            cube, 0.0, (0.0, 0.0), pixel_scale=1.0
        )
        x_min, x_max, y_min, y_max = extent
        # matplotlib-style extent spans pixel EDGES: width must equal
        # n_pixels * pixel_scale for the extent to describe the output cube
        assert (x_max - x_min) == pytest.approx(rot.shape[1] * 1.0, rel=1e-9)
        assert (y_max - y_min) == pytest.approx(rot.shape[0] * 1.0, rel=1e-9)


# ---------------------------------------------------------------------------
# 10. smooth_mask
# ---------------------------------------------------------------------------
class TestSmoothMask:
    @staticmethod
    def _blob_cube():
        # (ny, nx, nchan) 3-D Gaussian blob, amplitude 1, near-zero background
        y, x, v = np.indices((32, 32, 16)).astype(float)
        blob = np.exp(
            -((y - 16) ** 2 + (x - 16) ** 2) / (2 * 3.0 ** 2)
            - (v - 8) ** 2 / (2 * 2.0 ** 2)
        )
        return blob

    def test_mask_true_at_core_false_at_corner(self, capsys):
        cube = self._blob_cube()
        mask = smooth_mask(cube, sigma=2, hann=5, clip=0.02)
        capsys.readouterr()  # swallow the prints
        assert mask.dtype == bool
        assert mask.shape == cube.shape
        assert mask[16, 16, 8]          # blob core
        assert not mask[0, 0, 0]        # far corner
        assert not mask[31, 31, 15]     # opposite corner
        assert 0 < mask.sum() < mask.size

    def test_clip_above_peak_gives_all_false(self, capsys):
        cube = self._blob_cube()
        # smoothing can only reduce the max, so clip > raw peak => empty mask
        mask = smooth_mask(cube, sigma=2, hann=5, clip=2.0)
        capsys.readouterr()
        assert not mask.any()
