"""Shared fixtures and helpers for the SuperMAGE test suite.

Conventions
-----------
- Everything runs on CPU by default so the suite passes on any machine;
  tests that specifically exercise CUDA behavior are marked ``@pytest.mark.cuda``
  and skipped when no GPU is present.
- Every test is seeded (autouse fixture) — no flaky randomness.
- caskade models are called with a flat parameter tensor as the FIRST
  positional argument, ordered by ``[p.name for p in model.dynamic_params]``;
  use :func:`flat_theta` to assemble it from a name->value dict.
"""
import numpy as np
import pytest
import torch

# Rest-frame CO(2-1) frequency [Hz] used throughout the tutorials/tests.
LINE_HZ = 230.538e9
C_KMS = 299792.458
# G in pc (km/s)^2 / M_sun — the package default.
G_PC = 0.004301


@pytest.fixture(autouse=True)
def _seed_everything():
    torch.manual_seed(0)
    np.random.seed(0)
    yield


@pytest.fixture
def device():
    return "cpu"


@pytest.fixture
def dtype():
    return torch.float32


def make_freq_axis(n_chan=20, span_kms=600.0, line_hz=LINE_HZ):
    """Uniform ascending frequency axis [Hz] spanning ``span_kms`` around the line."""
    df = line_hz * (span_kms / C_KMS) / (n_chan - 1)
    return line_hz + (np.arange(n_chan) - (n_chan - 1) / 2) * df


def flat_theta(model, values, device="cpu", dtype=torch.float32):
    """Assemble the flat parameter tensor in ``model.dynamic_params`` order.

    ``values`` maps param name -> float or list (for shape-(1,) params a
    plain float is fine; lists are flattened in).
    """
    out = []
    for p in model.dynamic_params:
        v = values[p.name]
        out.extend(v if isinstance(v, (list, tuple)) else [v])
    return torch.tensor(out, device=device, dtype=dtype)


def requires_cuda():
    return pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
