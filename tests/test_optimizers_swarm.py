"""Tests for supermage.solvers.optimizers and supermage.solvers.swarm_optimization.

Toy problems
------------
* A smooth 3-parameter exponential model ``y = a*exp(-b*x) + c`` on
  ``x = linspace(0, 5, 200)`` (float64) with truth ``(2.0, 1.3, 0.5)`` for the
  local optimizers.
* A 2-parameter linear least-squares "bowl" ``chi2 = ||A(x - x_true)||^2``
  (known global minimum, chi2_min = 0) for the Sobol swarm wrappers.

All references are computed independently in the tests (dense
``torch.linalg.solve``, analytic Jacobians, explicit chi2 loops).
"""
import math

import pytest
import torch
from torch.quasirandom import SobolEngine

from supermage.solvers.optimizers import (
    adam_optimizer,
    cg_solve,
    lm_cg_autograd_stable,
    lm_direct,
)
from supermage.solvers.swarm_optimization import (
    batch_chi2,
    iterated_chi2,
    sobol_ga_swarm_efficient,
    sobol_ga_swarm_no_nan,
    sobol_ga_swarm_opt,
    sobol_sample,
    sobol_swarm_opt,
)

DTYPE = torch.float64

# ---------------------------------------------------------------------------
# Toy problem 1: y = a * exp(-b * x) + c on x in [0, 5]
# ---------------------------------------------------------------------------
X_GRID = torch.linspace(0.0, 5.0, 200, dtype=DTYPE)
TRUTH = torch.tensor([2.0, 1.3, 0.5], dtype=DTYPE)
START = torch.tensor([1.0, 2.0, 0.0], dtype=DTYPE)
N_CHI = X_GRID.numel()


def exp_model(theta):
    return theta[0] * torch.exp(-theta[1] * X_GRID) + theta[2]


def exp_jacobian(theta):
    """Analytic (Dout, 3) Jacobian of exp_model — independent of autograd."""
    e = torch.exp(-theta[1] * X_GRID)
    return torch.stack([e, -theta[0] * X_GRID * e, torch.ones_like(X_GRID)], dim=1)


Y_CLEAN = exp_model(TRUTH)


# ---------------------------------------------------------------------------
# Toy problem 2: linear least-squares bowl chi2(x) = ||A x - A x_true||^2
# ---------------------------------------------------------------------------
def make_bowl(x_true, n_data=50, seed=123):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n_data, 2, generator=g, dtype=DTYPE)
    Y = A @ x_true
    return A, Y


BOWL_TRUE = torch.tensor([0.7, -0.3], dtype=DTYPE)
BOWL_A, BOWL_Y = make_bowl(BOWL_TRUE)


def bowl_f(x):
    return BOWL_A @ x


BOWL_BOUNDS = [(-3.0, 3.0), (-3.0, 3.0)]
BOWL_LM_KWARGS = dict(Y=BOWL_Y, f=bowl_f, n_chi=BOWL_Y.numel(), C=None,
                      max_iter=25, verbose=False)


# ===========================================================================
# 1. cg_solve
# ===========================================================================
def test_cg_solve_matches_dense_solve():
    g = torch.Generator().manual_seed(11)
    M = torch.randn(8, 8, generator=g, dtype=DTYPE)
    A = M @ M.T + 8 * torch.eye(8, dtype=DTYPE)  # SPD
    b = torch.randn(8, generator=g, dtype=DTYPE)

    x_cg = cg_solve(lambda v: A @ v, b, maxiter=50, tol=1e-10)
    x_ref = torch.linalg.solve(A, b)

    assert x_cg.shape == b.shape
    assert torch.allclose(x_cg, x_ref, rtol=1e-5, atol=1e-10)


# ===========================================================================
# 2. lm_direct
# ===========================================================================
def test_lm_direct_noiseless_recovery():
    X0 = START.clone()
    X0_saved = X0.clone()

    result = lm_direct(X0, Y_CLEAN, exp_model, n_chi=N_CHI, verbose=False)
    assert isinstance(result, tuple) and len(result) == 3
    X_opt, L_final, chi2 = result

    assert torch.allclose(X_opt, TRUTH, atol=1e-5)
    assert chi2.item() < 1e-10
    # returned chi2 is consistent with an independent recomputation at X_opt
    chi2_recomputed = ((Y_CLEAN - exp_model(X_opt)) ** 2).sum()
    assert math.isclose(chi2.item(), chi2_recomputed.item(),
                        rel_tol=1e-6, abs_tol=1e-20)
    # input must not be modified in place
    assert torch.equal(X0, X0_saved)


def test_lm_direct_unit_variances_match_C_none():
    X_none, _, chi2_none = lm_direct(
        START.clone(), Y_CLEAN, exp_model, n_chi=N_CHI, C=None, verbose=False
    )
    X_ones, _, chi2_ones = lm_direct(
        START.clone(), Y_CLEAN, exp_model, n_chi=N_CHI,
        C=torch.ones_like(Y_CLEAN), verbose=False,
    )
    assert torch.allclose(X_none, X_ones, atol=1e-9)
    assert math.isclose(chi2_none.item(), chi2_ones.item(),
                        rel_tol=1e-6, abs_tol=1e-18)


def test_lm_direct_full_covariance_matches_diagonal_variances():
    g = torch.Generator().manual_seed(7)
    var = 0.5 + torch.rand(N_CHI, generator=g, dtype=DTYPE)  # nonuniform variances
    gn = torch.Generator().manual_seed(1234)
    Y_noisy = Y_CLEAN + 0.01 * torch.randn(N_CHI, generator=gn, dtype=DTYPE)

    X_diag, _, chi2_diag = lm_direct(
        START.clone(), Y_noisy, exp_model, n_chi=N_CHI, C=var, verbose=False
    )
    X_full, _, chi2_full = lm_direct(
        START.clone(), Y_noisy, exp_model, n_chi=N_CHI, C=torch.diag(var),
        verbose=False,
    )
    assert torch.allclose(X_diag, X_full, atol=1e-8)
    assert math.isclose(chi2_diag.item(), chi2_full.item(), rel_tol=1e-8)


def test_lm_direct_noisy_recovery_within_laplace_errors():
    sigma = 0.01
    g = torch.Generator().manual_seed(1234)
    Y_noisy = Y_CLEAN + sigma * torch.randn(N_CHI, generator=g, dtype=DTYPE)

    X_opt, _, chi2 = lm_direct(START.clone(), Y_noisy, exp_model,
                               n_chi=N_CHI, verbose=False)

    # Laplace / Gauss-Newton parameter covariance with unit weights:
    # cov = sigma^2 (J^T J)^{-1}, J the ANALYTIC Jacobian at truth.
    J = exp_jacobian(TRUTH)
    cov = sigma**2 * torch.linalg.inv(J.T @ J)
    param_err = torch.sqrt(torch.diag(cov))

    pulls = (X_opt - TRUTH) / param_err
    assert torch.all(pulls.abs() < 5.0), f"pulls={pulls.tolist()}"
    # chi2 should be near n_chi for correctly weighted noise (loose sanity)
    assert 0.5 * N_CHI * sigma**2 < chi2.item() < 2.0 * N_CHI * sigma**2


# ===========================================================================
# 3. lm_cg_autograd_stable
# ===========================================================================
def test_lm_cg_autograd_stable_converges():
    # NOTE: needs a gentler initial damping than the default L=1.0 on this
    # problem — see the xfail test below documenting the default-damping abort.
    X_opt, L_final, chi2 = lm_cg_autograd_stable(
        START.clone(), Y_CLEAN, exp_model, n_chi=N_CHI,
        L=1e-3, max_iter=60, cg_maxiter=30, cg_tol=1e-10,
        stopping=1e-9, verbose=False,
    )
    assert torch.allclose(X_opt, TRUTH, atol=1e-3)
    # chi2 comparable to lm_direct's (both effectively zero)
    assert chi2.item() < 1e-10


@pytest.mark.xfail(
    strict=True,
    reason=(
        "BUG: lm_cg_autograd_stable's divergence guard "
        "(`rho.item() < -100 -> break`) fires on a REJECTED trial step. "
        "With the default initial damping L=1.0 on this smooth exponential "
        "problem, iteration 2 proposes a wild step (chi2_new ~ 1e223) that is "
        "correctly rejected, but the loop then aborts entirely instead of "
        "raising L and continuing, returning X far from the minimum "
        "(chi2/n ~ 6.8e-2 instead of ~0). Standard LM would recover; the "
        "guard should only trip on divergence of the ACCEPTED state."
    ),
)
def test_lm_cg_autograd_stable_default_damping_premature_abort():
    X_opt, _, chi2 = lm_cg_autograd_stable(
        START.clone(), Y_CLEAN, exp_model, n_chi=N_CHI,
        max_iter=200, stopping=1e-12, verbose=False,
    )
    assert torch.allclose(X_opt, TRUTH, atol=1e-3)
    assert chi2.item() < 1e-6


# ===========================================================================
# 4. adam_optimizer
# ===========================================================================
def test_adam_optimizer_reduces_chi2_and_return_convention():
    chi2_init = ((Y_CLEAN - exp_model(START)) ** 2).sum().item()

    result = adam_optimizer(START.clone(), Y_CLEAN, exp_model,
                            max_iter=300, lr=0.05, verbose=False)
    assert isinstance(result, tuple) and len(result) == 3
    X_opt, placeholder, chi2 = result

    assert placeholder is None
    assert X_opt.requires_grad is False
    assert chi2.item() < chi2_init / 10.0
    # chi2 returned is the chi2 AT the returned X
    chi2_recomputed = ((Y_CLEAN - exp_model(X_opt)) ** 2).sum().item()
    assert math.isclose(chi2.item(), chi2_recomputed, rel_tol=1e-9, abs_tol=1e-15)


# ===========================================================================
# 5. sobol_sample
# ===========================================================================
def test_sobol_sample_shape_bounds_dtype_device():
    bounds = [(-2.0, 3.0), (0.5, 1.5), (10.0, 20.0)]
    low = torch.tensor([b[0] for b in bounds], dtype=DTYPE)
    high = torch.tensor([b[1] for b in bounds], dtype=DTYPE)

    for dt in (torch.float32, torch.float64):
        s = sobol_sample(bounds, 17, dtype=dt, device="cpu", seed=5)
        assert s.shape == (17, 3)
        assert s.dtype == dt
        assert s.device.type == "cpu"
        assert torch.all(s >= low.to(dt)) and torch.all(s <= high.to(dt))


def test_sobol_sample_seed_reproducibility():
    bounds = [(-1.0, 1.0), (0.0, 4.0)]
    s1 = sobol_sample(bounds, 16, dtype=DTYPE, device="cpu", seed=5)
    s2 = sobol_sample(bounds, 16, dtype=DTYPE, device="cpu", seed=5)
    s3 = sobol_sample(bounds, 16, dtype=DTYPE, device="cpu", seed=6)
    assert torch.equal(s1, s2)
    assert not torch.equal(s1, s3)


# ===========================================================================
# 6. iterated_chi2 / batch_chi2
# ===========================================================================
def _direct_chi2_loop(points, Y, f, Cinv):
    return torch.stack([((Y - f(p)) ** 2 * Cinv).sum() for p in points])


def test_iterated_chi2_matches_direct_loop():
    points = sobol_sample(BOWL_BOUNDS, 6, dtype=DTYPE, device="cpu", seed=3)
    g = torch.Generator().manual_seed(21)
    Cinv = 1.0 / (0.5 + torch.rand(BOWL_Y.numel(), generator=g, dtype=DTYPE))

    chi2 = iterated_chi2(points, Y=BOWL_Y, f=bowl_f, Cinv=Cinv)
    ref = _direct_chi2_loop(points, BOWL_Y, bowl_f, Cinv)

    assert chi2.shape == (6,)
    assert torch.allclose(chi2, ref, rtol=1e-12, atol=1e-12)


def test_batch_chi2_matches_direct_loop():
    points = sobol_sample(BOWL_BOUNDS, 7, dtype=DTYPE, device="cpu", seed=3)
    g = torch.Generator().manual_seed(22)
    Cinv = 1.0 / (0.5 + torch.rand(BOWL_Y.numel(), generator=g, dtype=DTYPE))

    def f_batched(xb):  # (B, D) -> (B, Dout)
        return xb @ BOWL_A.T

    chi2 = batch_chi2(points, Y=BOWL_Y, f=f_batched, Cinv=Cinv, batch_size=3)
    ref = _direct_chi2_loop(points, BOWL_Y, bowl_f, Cinv)

    assert chi2.shape == (7,)
    assert torch.allclose(chi2, ref, rtol=1e-12, atol=1e-12)


# ===========================================================================
# 7. sobol_swarm_opt
# ===========================================================================
def test_sobol_swarm_opt_finds_bowl_minimum():
    result = sobol_swarm_opt(
        lm_direct, BOWL_BOUNDS, 4, dtype=DTYPE, device="cpu",
        lm_kwargs=dict(BOWL_LM_KWARGS), verbose=False,
    )
    assert isinstance(result, tuple) and len(result) == 4
    best_X, best_val, best_L, history = result

    assert best_val.item() == pytest.approx(0.0, abs=1e-8)
    assert torch.allclose(best_X, BOWL_TRUE, atol=1e-4)

    assert len(history) == 4
    for i, entry in enumerate(history, 1):
        assert set(entry.keys()) == {"idx", "X", "chi2", "L"}
        assert entry["idx"] == i
        assert entry["X"].shape == (2,)
        assert isinstance(entry["chi2"], float)
        assert isinstance(entry["L"], float)


# ===========================================================================
# 8. sobol_ga_swarm_no_nan (NaN-region forward model)
# ===========================================================================
def test_sobol_ga_swarm_no_nan_filters_scouts_and_converges():
    x_true = torch.tensor([1.0, -0.5], dtype=DTYPE)  # minimum has x[0] > 0
    Y = BOWL_A @ x_true
    bounds = [(-2.0, 2.0), (-2.0, 2.0)]
    n_particles, oversample, seed = 4, 8, 42
    n_scouts = oversample * n_particles

    def f_nan(x):
        gate = torch.where(
            x[0] > 0,
            torch.tensor(1.0, dtype=x.dtype),
            torch.tensor(float("nan"), dtype=x.dtype),
        )
        return gate * (BOWL_A @ x)

    result = sobol_ga_swarm_no_nan(
        lm_direct, bounds, n_particles, Y.numel(),
        oversample=oversample, dtype=DTYPE, device="cpu",
        lm_kwargs=dict(Y=Y, f=f_nan, n_chi=Y.numel(), C=None,
                       max_iter=25, verbose=False),
        verbose=False, seed=seed,
    )
    assert isinstance(result, tuple) and len(result) == 4
    best_X, best_val, best_L, history = result

    assert torch.isfinite(best_val)
    assert best_val.item() < 1e-8
    assert torch.allclose(best_X, x_true, atol=1e-4)
    assert best_X[0].item() > 0

    # Independently reproduce the scout draw and verify NaN scouts really
    # were filtered: survivors = min(int(0.1 * n_scouts), n_valid).
    engine = SobolEngine(dimension=2, scramble=True, seed=seed)
    u = engine.draw(n_scouts).to(dtype=DTYPE)
    lo = torch.tensor([b[0] for b in bounds], dtype=DTYPE)
    hi = torch.tensor([b[1] for b in bounds], dtype=DTYPE)
    scouts = lo + (hi - lo) * u
    n_valid = int((scouts[:, 0] > 0).sum())
    assert 0 < n_valid < n_scouts  # NaN region actually sampled
    expected_k = min(max(1, int(0.10 * n_scouts)), n_valid)
    assert len(history) == expected_k


# ===========================================================================
# 9. sobol_ga_swarm_opt (clean bowl)
# ===========================================================================
def test_sobol_ga_swarm_opt_finds_bowl_minimum():
    result = sobol_ga_swarm_opt(
        lm_direct, BOWL_BOUNDS, 4, BOWL_Y.numel(),
        oversample=8, dtype=DTYPE, device="cpu",
        lm_kwargs=dict(BOWL_LM_KWARGS), verbose=False, seed=42,
    )
    assert isinstance(result, tuple) and len(result) == 4
    best_X, best_val, best_L, history = result

    assert best_val.item() == pytest.approx(0.0, abs=1e-8)
    assert torch.allclose(best_X, BOWL_TRUE, atol=1e-4)
    # keep_frac=0.1 of 32 scouts -> 3 survivors
    assert len(history) == max(1, int(0.10 * 8 * 4))


# ===========================================================================
# 10. sobol_ga_swarm_efficient (CPU, no workers)
# ===========================================================================
def test_sobol_ga_swarm_efficient_cpu():
    result = sobol_ga_swarm_efficient(
        lm_direct, BOWL_BOUNDS, 4, BOWL_Y.numel(),
        oversample=8, n_workers=0, dtype=DTYPE, device="cpu",
        lm_kwargs=dict(BOWL_LM_KWARGS), verbose=False, seed=42,
    )
    assert isinstance(result, tuple) and len(result) == 4
    best_X, best_val, best_L, history = result

    assert best_val.item() == pytest.approx(0.0, abs=1e-8)
    assert torch.allclose(best_X, BOWL_TRUE, atol=1e-4)
    assert isinstance(best_L, float)
    assert len(history) == max(1, int(0.10 * 8 * 4))
    for entry in history:
        assert set(entry.keys()) == {"idx", "X", "chi2", "L"}
