"""Tests for supermage.solvers.MCMC — MALA sampler + log-prob building blocks.

Covers:
- log_like_gaussian (value vs manual float64 mirror, output dtype, autograd
  gradient vs analytic linear-model formula)
- log_prior_tophat / log_prior_tophat_vmappable (in/out of box, vmap batch,
  differentiable zero inside the box)
- _logp_and_grad_batch vs _logp_and_grad_batch_vmap agreement + analytic grads
- mala correctness on an analytic 2-D correlated Gaussian target (moments,
  acceptance rate, output shapes, chi2_trace consistency)
- hastings=False path, determinism, use_vmap=False path
- checkpointed_mala: no-checkpoint mode, checkpoint write, already-complete
  resume, partial resume continuation
- mala_chi alias identity
"""
import numpy as np
import pytest
import torch

from supermage.solvers.MCMC import (
    _logp_and_grad_batch,
    _logp_and_grad_batch_vmap,
    checkpointed_mala,
    log_like_gaussian,
    log_prior_tophat,
    log_prior_tophat_vmappable,
    mala,
    mala_chi,
)

F64 = torch.float64


# ---------------------------------------------------------------------------
# Shared analytic targets
# ---------------------------------------------------------------------------

MU = torch.tensor([1.0, -2.0], dtype=F64)
COV = torch.tensor([[1.0, 0.6], [0.6, 0.5]], dtype=F64)
PREC = torch.linalg.inv(COV)


def gauss_logp(theta):
    """Log-density (up to a constant) of N(MU, COV); vmappable + differentiable."""
    d = theta - MU
    return -0.5 * (d @ (PREC @ d))


def std_normal_logp(theta):
    return -0.5 * (theta * theta).sum()


def make_gauss_init(C=8, seed=7):
    g = torch.Generator().manual_seed(seed)
    return MU + 0.05 * torch.randn(C, 2, generator=g, dtype=F64)


# ---------------------------------------------------------------------------
# 1. log_like_gaussian
# ---------------------------------------------------------------------------

class TestLogLikeGaussian:
    def test_value_matches_manual_float64_and_dtype(self):
        """-0.5*sum((Y - f(theta))^2 * Cinv) with float64 reduction; exact mirror."""
        g = torch.Generator().manual_seed(11)
        N, D = 12, 3
        M = torch.randn(N, D, generator=g, dtype=torch.float32)
        theta = torch.randn(D, generator=g, dtype=torch.float32)
        Y = torch.randn(N, generator=g, dtype=torch.float32)
        Cinv = torch.rand(N, generator=g, dtype=torch.float32) + 0.5

        val = log_like_gaussian(theta, Y, lambda t: M @ t, Cinv)

        # mirror the implementation's op order exactly: residual in float32,
        # THEN upcast to float64 before square+sum
        dY = (Y - M @ theta).double()
        expected = -0.5 * (dY.square() * Cinv.double()).sum()

        assert val.dtype == torch.float64  # float64 even for float32 inputs
        assert torch.equal(val, expected)  # bitwise identical mirror

    def test_value_manual_float64_inputs(self):
        g = torch.Generator().manual_seed(12)
        N, D = 20, 4
        M = torch.randn(N, D, generator=g, dtype=F64)
        theta = torch.randn(D, generator=g, dtype=F64)
        Y = torch.randn(N, generator=g, dtype=F64)
        Cinv = torch.rand(N, generator=g, dtype=F64) + 0.1

        val = log_like_gaussian(theta, Y, lambda t: M @ t, Cinv)
        r = (Y - M @ theta).numpy()
        expected = -0.5 * np.sum(r**2 * Cinv.numpy())
        np.testing.assert_allclose(val.item(), expected, rtol=1e-14)

    def test_gradient_linear_forward_matches_analytic(self):
        """d/dtheta [-0.5 sum((Y-M theta)^2 Cinv)] = M^T (Cinv * (Y - M theta))."""
        g = torch.Generator().manual_seed(13)
        N, D = 15, 3
        M = torch.randn(N, D, generator=g, dtype=F64)
        theta = torch.randn(D, generator=g, dtype=F64).requires_grad_(True)
        Y = torch.randn(N, generator=g, dtype=F64)
        Cinv = torch.rand(N, generator=g, dtype=F64) + 0.5

        ll = log_like_gaussian(theta, Y, lambda t: M @ t, Cinv)
        ll.backward()

        with torch.no_grad():
            grad_analytic = M.T @ (Cinv * (Y - M @ theta))
        np.testing.assert_allclose(
            theta.grad.numpy(), grad_analytic.numpy(), rtol=1e-6
        )

    def test_gradient_flows_through_float32_inputs(self):
        """Autograd flows through the double() cast for float32 inputs too."""
        g = torch.Generator().manual_seed(14)
        N, D = 10, 2
        M = torch.randn(N, D, generator=g, dtype=torch.float32)
        theta = torch.randn(D, generator=g, dtype=torch.float32).requires_grad_(True)
        Y = torch.randn(N, generator=g, dtype=torch.float32)
        Cinv = torch.ones(N, dtype=torch.float32)

        ll = log_like_gaussian(theta, Y, lambda t: M @ t, Cinv)
        ll.backward()
        assert theta.grad is not None
        with torch.no_grad():
            grad_analytic = (M.T.double() @ (Y - M @ theta).double()).float()
        np.testing.assert_allclose(
            theta.grad.numpy(), grad_analytic.numpy(), rtol=2e-5, atol=1e-6
        )


# ---------------------------------------------------------------------------
# 2. Top-hat priors
# ---------------------------------------------------------------------------

class TestTophatPriors:
    low = torch.tensor([-1.0, 0.0], dtype=F64)
    high = torch.tensor([1.0, 2.0], dtype=F64)

    # (theta, expected finite?) cases: inside, below dim0, above dim0,
    # below dim1, above dim1, and exactly on the boundary (inclusive)
    cases = [
        (torch.tensor([0.0, 1.0], dtype=F64), True),
        (torch.tensor([-1.5, 1.0], dtype=F64), False),
        (torch.tensor([1.5, 1.0], dtype=F64), False),
        (torch.tensor([0.0, -0.5], dtype=F64), False),
        (torch.tensor([0.0, 2.5], dtype=F64), False),
        (torch.tensor([-1.0, 2.0], dtype=F64), True),  # boundary is inclusive
    ]

    def test_log_prior_tophat_in_out(self):
        for theta, inside in self.cases:
            lp = log_prior_tophat(theta, self.low, self.high)
            if inside:
                assert lp.item() == 0.0
            else:
                assert lp.item() == -np.inf

    def test_vmappable_matches_plain(self):
        for theta, _ in self.cases:
            lp_plain = log_prior_tophat(theta, self.low, self.high)
            lp_v = log_prior_tophat_vmappable(theta, self.low, self.high)
            assert lp_plain.item() == lp_v.item()

    def test_vmappable_under_vmap_batch(self):
        batch = torch.stack([t for t, _ in self.cases])  # (B, D)
        expected = np.array(
            [0.0 if inside else -np.inf for _, inside in self.cases]
        )
        out = torch.func.vmap(
            lambda t: log_prior_tophat_vmappable(t, self.low, self.high)
        )(batch)
        assert out.shape == (len(self.cases),)
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_vmappable_differentiable_zero_inside(self):
        theta = torch.tensor([0.3, 1.2], dtype=F64, requires_grad=True)
        lp = log_prior_tophat_vmappable(theta, self.low, self.high)
        assert lp.grad_fn is not None  # differentiable zero, graph preserved
        lp.backward()
        assert torch.equal(theta.grad, torch.zeros(2, dtype=F64))


# ---------------------------------------------------------------------------
# 3. Batched logp+grad: sequential vs vmap
# ---------------------------------------------------------------------------

class TestLogpAndGradBatch:
    def test_sequential_vs_vmap_quadratic(self):
        g = torch.Generator().manual_seed(21)
        C, D = 6, 4
        A_half = torch.randn(D, D, generator=g, dtype=F64)
        A = A_half @ A_half.T + D * torch.eye(D, dtype=F64)  # SPD, symmetric
        b = torch.randn(D, generator=g, dtype=F64)

        def quad_logp(t):
            return -0.5 * (t @ (A @ t)) + b @ t

        x = torch.randn(C, D, generator=g, dtype=F64)

        logps_seq, grads_seq = _logp_and_grad_batch(x, quad_logp)
        logps_v, grads_v = _logp_and_grad_batch_vmap(x, quad_logp)

        assert logps_seq.shape == (C,) and grads_seq.shape == (C, D)
        assert logps_v.shape == (C,) and grads_v.shape == (C, D)
        np.testing.assert_allclose(logps_seq.numpy(), logps_v.numpy(), rtol=1e-6)
        np.testing.assert_allclose(grads_seq.numpy(), grads_v.numpy(), rtol=1e-6)

        # independent analytic gradient: grad = -A x + b (A symmetric)
        grad_analytic = (-x @ A.T + b).numpy()
        np.testing.assert_allclose(grads_seq.numpy(), grad_analytic, rtol=1e-8)
        np.testing.assert_allclose(
            logps_seq.numpy(),
            (-0.5 * torch.einsum("cd,de,ce->c", x, A, x) + x @ b).numpy(),
            rtol=1e-8,
        )

        # returned tensors are detached
        assert logps_seq.grad_fn is None and grads_seq.grad_fn is None
        assert logps_v.grad_fn is None and grads_v.grad_fn is None


# ---------------------------------------------------------------------------
# 4-7. mala on an analytic 2-D correlated Gaussian
# ---------------------------------------------------------------------------

class TestMalaGaussianTarget:
    def test_moments_acceptance_shapes_chi2(self):
        C, D, n_steps, burn = 8, 2, 1500, 300
        init = make_gauss_init(C)
        torch.manual_seed(0)
        samples, acc_mask, chi2_trace = mala(
            gauss_logp,
            init,
            n_steps=n_steps,
            step_size=1.2,
            mass_matrix=COV.numpy(),
            hastings=True,
            progress=False,
        )

        # shapes and dtypes
        assert samples.shape == (n_steps, C, D)
        assert acc_mask.shape == (n_steps, C) and acc_mask.dtype == bool
        assert chi2_trace.shape == (n_steps, C)
        assert np.isfinite(samples).all()

        # acceptance rate
        acc_rate = acc_mask.mean()
        assert 0.3 < acc_rate < 0.98, f"acceptance rate {acc_rate:.3f}"

        # mala must not mutate the caller's init tensor
        assert torch.equal(init, make_gauss_init(C))

        # pooled post-burn-in moments vs analytic target
        pooled = samples[burn:].reshape(-1, D)
        np.testing.assert_allclose(pooled.mean(axis=0), MU.numpy(), atol=0.1)
        cov_emp = np.cov(pooled.T)
        # rtol 0.15: this (seeded, deterministic) run measures a max
        # elementwise rel. err. of 0.027 (~5x headroom), while a sampler with
        # a broken MH proposal-correction term (measured by rerunning with
        # hastings=False at this step size) lands at ~0.38 — a 0.35 tolerance
        # only marginally caught that bias; 0.15 catches it robustly.
        np.testing.assert_allclose(cov_emp, COV.numpy(), rtol=0.15)

        # acc_mask semantics: a chain moves at step t iff its proposal was
        # accepted (rejection keeps x bitwise identical; an accepted Gaussian
        # proposal differs almost surely)
        moved = (samples[1:] != samples[:-1]).any(axis=-1)  # (n-1, C)
        np.testing.assert_array_equal(moved, acc_mask[1:])
        moved0 = (samples[0] != init.numpy()).any(axis=-1)
        np.testing.assert_array_equal(moved0, acc_mask[0])

        # chi2_trace at EVERY step == -2 * logp recomputed independently at
        # the stored sample (flat target: chi2 = (x-mu)^T PREC (x-mu))
        d = samples - MU.numpy()  # (n, C, D)
        chi2_expected = np.einsum("ncd,de,nce->nc", d, PREC.numpy(), d)
        np.testing.assert_allclose(chi2_trace, chi2_expected, rtol=1e-9, atol=1e-12)

    def test_hastings_false_runs_finite(self):
        """ULA-style path (no MH proposal correction) runs and stays finite."""
        init = make_gauss_init(4)
        torch.manual_seed(0)
        samples, acc_mask, chi2_trace = mala(
            gauss_logp,
            init,
            n_steps=300,
            step_size=0.5,
            mass_matrix=COV.numpy(),
            hastings=False,
            progress=False,
        )
        assert samples.shape == (300, 4, 2)
        assert np.isfinite(samples).all()
        assert np.isfinite(chi2_trace).all()
        assert acc_mask.dtype == bool  # no stronger claim about semantics

    def test_determinism_same_seed(self):
        init = make_gauss_init(4)
        results = []
        for _ in range(2):
            torch.manual_seed(0)
            results.append(
                mala(
                    gauss_logp,
                    init,
                    n_steps=200,
                    step_size=1.0,
                    mass_matrix=COV.numpy(),
                    progress=False,
                )
            )
        (s1, a1, c1), (s2, a2, c2) = results
        assert np.array_equal(s1, s2)
        assert np.array_equal(a1, a2)
        assert np.array_equal(c1, c2)

    def test_use_vmap_false_path(self):
        init = make_gauss_init(6)
        torch.manual_seed(0)
        samples, acc_mask, chi2_trace = mala(
            gauss_logp,
            init,
            n_steps=400,
            step_size=1.2,
            mass_matrix=COV.numpy(),
            hastings=True,
            progress=False,
            use_vmap=False,
        )
        assert samples.shape == (400, 6, 2)
        assert np.isfinite(samples).all()
        assert np.isfinite(chi2_trace).all()
        acc_rate = acc_mask.mean()
        assert 0.3 < acc_rate < 0.98, f"acceptance rate {acc_rate:.3f}"

    def test_flat_prior_plus_likelihood_target(self):
        """mala with tophat-prior + gaussian-loglike composition stays exact:
        chi2_trace equals -2*(loglike) since the flat prior contributes 0."""
        g = torch.Generator().manual_seed(31)
        N, D = 8, 2
        M = torch.randn(N, D, generator=g, dtype=F64)
        theta_true = torch.tensor([0.5, -0.3], dtype=F64)
        Y = M @ theta_true
        Cinv = torch.full((N,), 4.0, dtype=F64)
        low = torch.full((D,), -10.0, dtype=F64)
        high = torch.full((D,), 10.0, dtype=F64)

        def logp(t):
            return log_prior_tophat_vmappable(t, low, high) + log_like_gaussian(
                t, Y, lambda th: M @ th, Cinv
            )

        init = theta_true + 0.01 * torch.randn(3, D, generator=g, dtype=F64)
        torch.manual_seed(0)
        samples, acc_mask, chi2_trace = mala(
            logp, init, n_steps=100, step_size=0.3, progress=False
        )
        assert np.isfinite(chi2_trace).all()
        last = torch.tensor(samples[-1], dtype=F64)
        chi2_manual = np.array(
            [
                float(((Y - M @ r).square() * Cinv).sum())
                for r in last
            ]
        )
        np.testing.assert_allclose(chi2_trace[-1], chi2_manual, rtol=1e-9)


# ---------------------------------------------------------------------------
# 8. checkpointed_mala
# ---------------------------------------------------------------------------

class TestCheckpointedMala:
    C, D = 3, 2

    def _init(self):
        g = torch.Generator().manual_seed(42)
        return torch.randn(self.C, self.D, generator=g, dtype=torch.float32)

    def test_no_checkpoint_dir_writes_nothing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        torch.manual_seed(0)
        samples, acc_mask, chi2_trace = checkpointed_mala(
            std_normal_logp,
            self._init(),
            n_steps=30,
            step_size=1.0,
            checkpoint_dir=None,
            progress=False,
        )
        assert samples.shape == (30, self.C, self.D)
        assert acc_mask.shape == (30, self.C)
        assert chi2_trace.shape == (30, self.C)
        assert np.isfinite(samples).all()
        assert list(tmp_path.iterdir()) == []  # nothing written anywhere

    def test_checkpoint_write_complete_resume_and_continue(self, tmp_path):
        ckpt_dir = tmp_path / "ck"
        common = dict(
            step_size=1.0,
            checkpoint_dir=str(ckpt_dir),
            checkpoint_every=20,
            resume=True,
            progress=False,
        )

        # (b) fresh 60-step run writes mala_checkpoint.npz
        torch.manual_seed(123)
        s_b, a_b, c_b = checkpointed_mala(
            std_normal_logp, self._init(), n_steps=60, **common
        )
        ckpt_file = ckpt_dir / "mala_checkpoint.npz"
        assert ckpt_file.exists()
        assert s_b.shape == (60, self.C, self.D)
        with np.load(ckpt_file) as ck:
            assert int(ck["step"]) == 60

        # (c) identical rerun hits the already-complete branch -> byte-equal
        # content (both sides are the same float32 arrays, npz round-trip is
        # lossless, and no sampling runs — so exact equality must hold)
        torch.manual_seed(999)  # different seed must not matter: no sampling runs
        s_c, a_c, c_c = checkpointed_mala(
            std_normal_logp, self._init(), n_steps=60, **common
        )
        assert np.array_equal(s_b, s_c)
        assert np.array_equal(a_b, a_c)
        assert np.array_equal(c_b, c_c)

        # (d) extend to 100 steps: resumes from step 60, prefix identical to (b)
        torch.manual_seed(7)  # RNG state is restored from the checkpoint anyway
        s_d, a_d, c_d = checkpointed_mala(
            std_normal_logp, self._init(), n_steps=100, **common
        )
        assert s_d.shape == (100, self.C, self.D)
        assert a_d.shape == (100, self.C)
        assert c_d.shape == (100, self.C)
        assert np.array_equal(s_d[:60], s_b)
        assert np.array_equal(a_d[:60], a_b)
        assert np.array_equal(c_d[:60], c_b)
        assert np.isfinite(s_d).all()
        assert np.isfinite(c_d).all()
        # final checkpoint updated to 100 completed steps
        with np.load(ckpt_file) as ck:
            assert int(ck["step"]) == 100


# ---------------------------------------------------------------------------
# 9. alias
# ---------------------------------------------------------------------------

def test_mala_chi_alias_identity():
    assert mala_chi is mala
