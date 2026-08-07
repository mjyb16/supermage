"""Tests for the analytic profile functions, the precomputed MGE projector and
the interpolation/quadrature helpers in ``supermage.simulators.velocity_models``.

All numerical references are derived independently (numpy formulas, analytic
antiderivatives, hand-computed extrapolations) — never by re-running the code
under test.
"""
import numpy as np
import pytest
import torch

from supermage.simulators.velocity_models import (
    PrecomputedMGEProjector,
    core_sersic_torch_1d,
    interp1d,
    interpolate_velocity,
    leggauss_interval,
    nuker_profile_torch_1d,
    sersic_profile_torch_1d,
    transform_DE,
)

F64 = torch.float64


def t64(v):
    return torch.tensor(v, dtype=F64)


def _b_n(n):
    """Ciotti & Bertin (1999) asymptotic expansion for the Sersic b_n."""
    return 2.0 * n - 1.0 / 3.0 + 4.0 / (405.0 * n) + 46.0 / (25515.0 * n * n)


def _log_slope(fn, R0, h=1.01):
    """Numerical logarithmic slope dlnI/dlnR of ``fn`` at radius ``R0``."""
    R = t64([R0 / h, R0 * h])
    I = fn(R)
    return float((torch.log(I[1]) - torch.log(I[0])) / np.log(h * h))


# ---------------------------------------------------------------------------
# sersic_profile_torch_1d
# ---------------------------------------------------------------------------
class TestSersic:
    def test_value_at_effective_radius_is_exactly_Ie(self):
        I_e, R_e, n = 3.7, 520.0, 2.3
        out = sersic_profile_torch_1d(t64([R_e]), t64(I_e), t64(R_e), t64(n))
        # at R = R_e the exponent is exactly zero -> I = I_e bit-for-bit
        assert out.item() == I_e

    def test_strictly_decreasing(self):
        R = torch.tensor(np.geomspace(0.1, 1e4, 300), dtype=F64)
        I = sersic_profile_torch_1d(R, t64(1.0), t64(400.0), t64(4.0))
        assert (I[1:] < I[:-1]).all()

    @pytest.mark.parametrize("n", [0.7, 1.0, 2.5, 4.0, 8.0])
    def test_matches_independent_numpy_reference(self, n):
        I_e, R_e = 2.0, 750.0
        R_np = np.geomspace(0.5, 2e4, 200)
        b = _b_n(n)
        ref = I_e * np.exp(-b * ((R_np / R_e) ** (1.0 / n) - 1.0))
        got = sersic_profile_torch_1d(
            torch.tensor(R_np, dtype=F64), t64(I_e), t64(R_e), t64(n)
        ).numpy()
        np.testing.assert_allclose(got, ref, rtol=1e-12)

    @pytest.mark.xfail(
        reason="docstring says I_e/R_e/n may be plain floats, but "
        "torch.clamp(float, min=...) raises TypeError — doc/API mismatch",
        strict=True,
    )
    def test_accepts_python_floats_as_documented(self):
        R = t64([100.0])
        out = sersic_profile_torch_1d(R, 1.0, 500.0, 4.0)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# nuker_profile_torch_1d
# ---------------------------------------------------------------------------
class TestNuker:
    def test_value_at_break_radius_is_exactly_Ib(self):
        # At x = R/r_b = 1: prefactor 2^((beta-gamma)/alpha) is cancelled by
        # (1 + 1)^((gamma-beta)/alpha) and x^(-gamma) = 1, so I(r_b) = I_b.
        I_b, r_b = 5.5, 60.0
        out = nuker_profile_torch_1d(
            t64([r_b]), t64(I_b), t64(r_b), t64(1.7), t64(1.9), t64(0.4)
        )
        assert out.item() == pytest.approx(I_b, rel=1e-14)

    @pytest.mark.parametrize("alpha,beta,gamma", [(2.0, 1.8, 0.4), (0.8, 2.5, 0.0)])
    def test_asymptotic_log_slopes(self, alpha, beta, gamma):
        r_b, I_b = 50.0, 7.0

        def fn(R):
            return nuker_profile_torch_1d(
                R, t64(I_b), t64(r_b), t64(alpha), t64(beta), t64(gamma)
            )

        inner = _log_slope(fn, 1e-4 * r_b)
        outer = _log_slope(fn, 1e5 * r_b)
        assert inner == pytest.approx(-gamma, abs=0.02)
        assert outer == pytest.approx(-beta, abs=0.02)

    def test_matches_independent_numpy_reference(self):
        I_b, r_b, alpha, beta, gamma = 3.0, 45.0, 1.5, 2.2, 0.6
        R_np = np.geomspace(0.01, 1e5, 250)
        x = R_np / r_b
        ref = (
            I_b
            * 2.0 ** ((beta - gamma) / alpha)
            * x ** (-gamma)
            * (1.0 + x ** alpha) ** ((gamma - beta) / alpha)
        )
        got = nuker_profile_torch_1d(
            torch.tensor(R_np, dtype=F64),
            t64(I_b), t64(r_b), t64(alpha), t64(beta), t64(gamma),
        ).numpy()
        np.testing.assert_allclose(got, ref, rtol=1e-12)


# ---------------------------------------------------------------------------
# core_sersic_torch_1d
# ---------------------------------------------------------------------------
class TestCoreSersic:
    # pc-scale parameters that overflowed float32 before the 2026-06-24
    # log-space fix (R_e**alpha = (1.2e4)**9.6 ~ 1e39 > float32 max).
    OVERFLOW_PARS = dict(I_b=1e3, R_b=100.0, R_e=1.2e4, alpha=9.6, gamma=0.3, n=4.0)

    def _eval(self, R, dtype, **pars):
        def T(v):
            return torch.tensor(v, dtype=dtype)

        return core_sersic_torch_1d(
            R.to(dtype), T(pars["I_b"]), T(pars["R_b"]), T(pars["R_e"]),
            T(pars["alpha"]), T(pars["gamma"]), T(pars["n"]),
        )

    def test_float32_overflow_regression(self):
        """REGRESSION (fixed 2026-06-24): float32 must stay finite and match
        float64 for params where R_e**alpha overflows float32."""
        R = torch.tensor(np.geomspace(1.0, 3e4, 400), dtype=F64)
        I32 = self._eval(R, torch.float32, **self.OVERFLOW_PARS)
        I64 = self._eval(R, F64, **self.OVERFLOW_PARS)
        assert torch.isfinite(I32).all()
        assert (I32 > 0).all()
        np.testing.assert_allclose(
            I32.to(F64).numpy(), I64.numpy(), rtol=1e-4
        )

    def test_matches_direct_float64_reference(self):
        """Log-space form == direct algebraic form (safe in float64 with
        moderate parameters)."""
        I_b, R_b, R_e, alpha, gamma, n = 40.0, 30.0, 900.0, 3.0, 0.5, 3.5
        R_np = np.geomspace(0.5, 1e4, 200)
        b = _b_n(n)
        ref = (
            I_b
            * (1.0 + (R_b / R_np) ** alpha) ** (gamma / alpha)
            * np.exp(-b * (((R_np ** alpha + R_b ** alpha) / R_e ** alpha)
                           ** (1.0 / (alpha * n)) - 1.0))
        )
        got = self._eval(
            torch.tensor(R_np, dtype=F64), F64,
            I_b=I_b, R_b=R_b, R_e=R_e, alpha=alpha, gamma=gamma, n=n,
        ).numpy()
        np.testing.assert_allclose(got, ref, rtol=1e-12)

    def test_inner_slope_is_minus_gamma(self):
        pars = self.OVERFLOW_PARS

        def fn(R):
            return self._eval(R, F64, **pars)

        for fac in (1e-3, 1e-4):
            slope = _log_slope(fn, fac * pars["R_b"])
            assert slope == pytest.approx(-pars["gamma"], abs=0.02)

    def test_reduces_to_sersic_for_tiny_break_radius(self):
        R = torch.tensor(np.geomspace(1.0, 1e4, 150), dtype=F64)
        I_b, R_e, n = 5.0, 800.0, 3.0
        cs = self._eval(R, F64, I_b=I_b, R_b=1e-8, R_e=R_e,
                        alpha=5.0, gamma=0.5, n=n)
        ss = sersic_profile_torch_1d(R, t64(I_b), t64(R_e), t64(n))
        # R >> R_b everywhere on this grid
        np.testing.assert_allclose(cs.numpy(), ss.numpy(), rtol=1e-3)

    def test_finite_positive_on_wide_log_grid(self):
        R = torch.tensor(np.geomspace(1e-3, 1e6, 500), dtype=F64)
        I = self._eval(R, F64, **self.OVERFLOW_PARS)
        assert torch.isfinite(I).all()
        assert (I > 0).all()


# ---------------------------------------------------------------------------
# PrecomputedMGEProjector
# ---------------------------------------------------------------------------
N_GAUSS = 20
LAM_BASE = 1e-6


def _make_projector(dtype=F64):
    r_np = np.geomspace(1.0, 1e4, 100)
    # log-spaced sigma basis, bin midpoints of [r_min, r_max]/sqrt(3) — same
    # construction as _PreLSMGEBase.
    lo = np.log10(r_np[0] / np.sqrt(3.0))
    hi = np.log10(r_np[-1] / np.sqrt(3.0))
    dx = (hi - lo) / N_GAUSS
    sig_np = 10.0 ** (lo + (0.5 + np.arange(N_GAUSS)) * dx)
    proj = PrecomputedMGEProjector(
        r_grid_pc=torch.tensor(r_np, dtype=dtype),
        sigma_grid_pc=torch.tensor(sig_np, dtype=dtype),
        lam_base=LAM_BASE,
        dtype=dtype,
    )
    return proj, r_np


class TestPrecomputedMGEProjector:
    def test_shapes_and_lam(self):
        proj, r_np = _make_projector()
        R, K = len(r_np), N_GAUSS
        assert tuple(proj.A.shape) == (R, K)
        assert tuple(proj.P.shape) == (K, R)
        assert proj.lam_used >= LAM_BASE

    def test_design_matrix_is_gaussian_basis(self):
        proj, r_np = _make_projector()
        sig_np = proj.sigma_grid.numpy()
        ref = np.exp(-(r_np[:, None] ** 2) / (2.0 * sig_np[None, :] ** 2))
        np.testing.assert_allclose(proj.A.numpy(), ref, rtol=1e-12)

    def test_sersic_roundtrip_interior_accuracy(self):
        proj, r_np = _make_projector()
        r = torch.tensor(r_np, dtype=F64)
        y = sersic_profile_torch_1d(r, t64(1.0), t64(1000.0), t64(4.0))
        rec = proj.profile_from_surf(proj.surf_from_profile(y))
        rel = ((rec - y) / y).abs().numpy()
        logr = np.log10(r_np)
        frac = (logr - logr[0]) / (logr[-1] - logr[0])
        interior = frac <= 0.85  # exclude outer 15% in log-radius
        assert rel[interior].max() < 1e-2

    def test_in_span_profile_roundtrip_near_exact(self):
        """A profile that IS a sum of the basis Gaussians reconstructs to
        ~the ridge bias (lam=1e-6), i.e. far below any fitting error."""
        proj, _ = _make_projector()
        g = torch.Generator().manual_seed(3)
        surf_true = torch.rand(N_GAUSS, generator=g, dtype=F64) + 0.5
        y = proj.profile_from_surf(surf_true)
        rec = proj.profile_from_surf(proj.surf_from_profile(y))
        assert ((rec - y).abs().max() / y.abs().max()).item() < 1e-5

    def test_batched_matches_1d(self):
        proj, r_np = _make_projector()
        r = torch.tensor(r_np, dtype=F64)
        y1 = sersic_profile_torch_1d(r, t64(1.0), t64(1000.0), t64(4.0))
        y2 = sersic_profile_torch_1d(r, t64(2.0), t64(300.0), t64(1.0))
        batch = torch.stack([y1, y2])  # (2, R)

        surf_b = proj.surf_from_profile(batch)
        assert tuple(surf_b.shape) == (2, N_GAUSS)
        torch.testing.assert_close(surf_b[0], proj.surf_from_profile(y1))
        torch.testing.assert_close(surf_b[1], proj.surf_from_profile(y2))

        prof_b = proj.profile_from_surf(surf_b)
        assert tuple(prof_b.shape) == (2, len(r_np))
        torch.testing.assert_close(prof_b[0], proj.profile_from_surf(surf_b[0]))

    def test_3d_input_raises_valueerror(self):
        proj, _ = _make_projector()
        with pytest.raises(ValueError):
            proj.surf_from_profile(torch.zeros(2, 3, 100, dtype=F64))
        with pytest.raises(ValueError):
            proj.profile_from_surf(torch.zeros(2, 3, N_GAUSS, dtype=F64))


# ---------------------------------------------------------------------------
# interp1d
# ---------------------------------------------------------------------------
class TestInterp1d:
    def _knots(self):
        x = torch.linspace(0.0, 2.0 * np.pi, 50, dtype=F64)
        return x, torch.sin(x)

    def test_exact_at_knots(self):
        x, y = self._knots()
        out = interp1d(x, y, x.clone())
        torch.testing.assert_close(out, y, rtol=0.0, atol=1e-14)

    def test_smooth_accuracy_vs_sin(self):
        x, y = self._knots()
        xs = torch.linspace(0.05, 2.0 * np.pi - 0.05, 333, dtype=F64)
        out = interp1d(x, y, xs)
        # dense knots: cubic-spline-level accuracy (measured ~3.4e-5)
        assert (out - torch.sin(xs)).abs().max().item() < 1e-3

    def test_extend_const_returns_last_value(self):
        x, y = self._knots()
        xs = torch.tensor([2.0 * np.pi + 0.5, 2.0 * np.pi + 1.7], dtype=F64)
        out = interp1d(x, y, xs, extend="const")
        torch.testing.assert_close(
            out, y[-1].expand(2), rtol=0.0, atol=0.0
        )

    def test_extend_linear_matches_hand_computed(self):
        x, y = self._knots()
        xs = torch.tensor([2.0 * np.pi + 0.5, 2.0 * np.pi + 1.7], dtype=F64)
        out = interp1d(x, y, xs, extend="linear")
        slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
        expected = y[-1] + (xs - x[-1]) * slope
        torch.testing.assert_close(out, expected, rtol=0.0, atol=1e-14)


# ---------------------------------------------------------------------------
# interpolate_velocity
# ---------------------------------------------------------------------------
class TestInterpolateVelocity:
    def _grid(self):
        R_grid = torch.tensor(np.linspace(0.5, 10.0, 25), dtype=F64)
        v_grid = 3.0 * torch.sqrt(R_grid)
        return R_grid, v_grid

    def test_matches_np_interp_interior(self):
        R_grid, v_grid = self._grid()
        rng = np.random.default_rng(1)
        q_np = rng.uniform(0.6, 9.9, size=(4, 7))
        got = interpolate_velocity(R_grid, torch.tensor(q_np, dtype=F64), v_grid)
        ref = np.interp(q_np, R_grid.numpy(), v_grid.numpy())
        np.testing.assert_allclose(got.numpy(), ref, rtol=1e-14, atol=0.0)

    def test_clamps_to_edge_values(self):
        R_grid, v_grid = self._grid()
        q = torch.tensor([0.0, 0.4, 11.0, 1e6], dtype=F64)
        got = interpolate_velocity(R_grid, q, v_grid)
        expected = torch.stack([v_grid[0], v_grid[0], v_grid[-1], v_grid[-1]])
        torch.testing.assert_close(got, expected, rtol=0.0, atol=0.0)

    def test_exact_at_grid_points(self):
        R_grid, v_grid = self._grid()
        got = interpolate_velocity(R_grid, R_grid.clone(), v_grid)
        torch.testing.assert_close(got, v_grid, rtol=0.0, atol=1e-14)

    def test_autograd_through_v_grid(self):
        R_grid, v_grid = self._grid()
        v = v_grid.clone().requires_grad_(True)
        q = torch.tensor([[1.3, 4.4], [7.7, 9.2]], dtype=F64)
        interpolate_velocity(R_grid, q, v).sum().backward()
        assert v.grad is not None
        assert torch.isfinite(v.grad).all()
        # each of the 4 queries distributes unit weight over two neighbours
        assert v.grad.sum().item() == pytest.approx(4.0, rel=1e-12)
        assert v.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# leggauss_interval
# ---------------------------------------------------------------------------
class TestLeggaussInterval:
    def test_integrates_polynomials_exactly(self):
        a, b = 0.3, 2.1
        x, w = leggauss_interval(8, t64(a), t64(b), device="cpu", dtype=F64)
        assert x.shape[-1] == 8
        for k in (0, 5, 7):
            got = (w * x ** k).sum().item()
            exact = (b ** (k + 1) - a ** (k + 1)) / (k + 1)
            assert got == pytest.approx(exact, rel=1e-12)

    def test_batched_intervals_broadcast(self):
        t_low = t64([0.3, 1.0])
        t_high = t64([2.1, 3.0])
        x, w = leggauss_interval(8, t_low, t_high, device="cpu", dtype=F64)
        assert tuple(x.shape) == (2, 8)
        assert tuple(w.shape) == (2, 8)
        for k in (5, 7):
            got = (w * x ** k).sum(dim=-1)
            exact = (t_high ** (k + 1) - t_low ** (k + 1)) / (k + 1)
            torch.testing.assert_close(got, exact, rtol=1e-12, atol=0.0)
        # nodes stay inside their own interval
        assert (x[0] > 0.3).all() and (x[0] < 2.1).all()
        assert (x[1] > 1.0).all() and (x[1] < 3.0).all()


# ---------------------------------------------------------------------------
# transform_DE
# ---------------------------------------------------------------------------
class TestTransformDE:
    def test_values_match_numpy(self):
        t_np = np.array([-2.0, -0.5, 0.0, 1.0, 2.5])
        u, du = transform_DE(torch.tensor(t_np, dtype=F64))
        u_ref = np.exp((np.pi / 2.0) * np.sinh(t_np))
        du_ref = (np.pi / 2.0) * np.cosh(t_np) * u_ref
        np.testing.assert_allclose(u.numpy(), u_ref, rtol=1e-14)
        np.testing.assert_allclose(du.numpy(), du_ref, rtol=1e-14)

    def test_jacobian_matches_autograd(self):
        t = torch.tensor([-2.0, -0.5, 0.0, 1.0, 2.5], dtype=F64,
                         requires_grad=True)
        u, du_dt = transform_DE(t)
        (grad,) = torch.autograd.grad(u.sum(), t)
        torch.testing.assert_close(grad, du_dt.detach(), rtol=1e-10, atol=0.0)
