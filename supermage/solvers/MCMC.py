"""Gradient-based MCMC sampling (MALA) plus log-probability building blocks.

Provides a batched Metropolis-adjusted Langevin sampler (:func:`mala`, and
:func:`checkpointed_mala` with disk checkpoint/resume), Gaussian
log-likelihood and top-hat log-prior helpers (including
:func:`log_like_gaussian_flux_marginal`, the likelihood with a pixelated
flux distribution analytically marginalized under a Gaussian prior), and
vmap-accelerated log-probability + gradient evaluation.  All samplers run
``C`` independent chains in parallel on the GPU; gradients come from
autograd through the (differentiable) SuperMAGE forward models.
"""
import math, torch, numpy as np

def log_like_gaussian(theta, Y_obs, forward_func, Cinv):
    """Gaussian log-likelihood ``-0.5 * chi^2`` with a float64 reduction.

    Parameters
    ----------
    theta : Tensor, shape (D,)
        Parameter vector.
    Y_obs : Tensor
        Observed data (any shape).
    forward_func : callable
        Forward model mapping ``theta`` to a tensor with the same shape and
        device as ``Y_obs``.
    Cinv : Tensor
        Inverse variance weights, broadcastable against ``Y_obs``.

    Returns
    -------
    Tensor (scalar, float64)
        ``-0.5 * sum((Y_obs - f(theta))^2 * Cinv)``.  The residual is
        upcast to float64 before square+sum to avoid logL quantization
        plateaus (see inline comment); autograd flows through the cast.
    """
    fY   = forward_func(theta)
    dY   = Y_obs - fY
    # float64 reduction: forward model may be float32, but accumulating chi^2 in float32 at
    # |logL| ~ 1e5-1e6 quantises the likelihood (~0.016-0.25 ULP) -> plateaus / sampler issues.
    # Upcasting the residual before square+sum removes this at negligible cost (autograd-safe).
    chi2 = (dY.double().square() * Cinv.double()).sum()
    return -0.5 * chi2

def _flux_marginal_terms(theta, Y_obs, forward_func, Cinv,
                         flux_response, flux_prior_std, gram=None, jitter=0.0):
    """Shared pieces of the flux-marginalized Gaussian likelihood.

    Computes, for the linear-flux model ``y = f(theta) + A(theta) @ delta_f``
    with noise ``N(0, Cinv^-1)`` and flux-deviation prior
    ``delta_f ~ N(0, diag(s)^2)``:

    * ``chi2_base`` : the ordinary whitened ``chi^2`` of the prior-mean model
      (float64 reduction, same convention as :func:`log_like_gaussian`);
    * ``b = S A^T Cinv r`` : the prior-scaled whitened adjoint of the
      residual (shape ``(K,)``, float64);
    * ``L`` : Cholesky factor of ``M = I_K + S A^T Cinv A S`` (float64);
    * ``s`` : the resolved per-pixel prior std (shape ``(K,)``, float64).

    The two data-sized contractions (``A^T Cinv r`` and, when ``gram`` is not
    supplied, ``A^T Cinv A``) run in ``A``'s dtype (typically float32, like
    every SuperMAGE forward); all K-sized quantities are float64.
    """
    fY = forward_func(theta)
    r = (Y_obs - fY).reshape(-1)
    cinv = torch.broadcast_to(Cinv, Y_obs.shape).reshape(-1)
    chi2_base = (r.double().square() * cinv.double()).sum()

    A = flux_response(theta) if callable(flux_response) else flux_response
    if A.ndim != 2 or A.shape[0] != r.numel():
        raise ValueError(
            f"flux_response must be (N_data, K) with N_data == Y_obs.numel() "
            f"= {r.numel()}; got {tuple(A.shape)}")
    K = A.shape[1]

    s = flux_prior_std(theta) if callable(flux_prior_std) else flux_prior_std
    s = torch.as_tensor(s, device=r.device).to(torch.float64).reshape(-1)
    if s.numel() == 1:
        s = s.expand(K)
    elif s.numel() != K:
        raise ValueError(
            f"flux_prior_std must be a scalar or length-{K}; got {s.numel()}")

    v = (cinv * r).to(A.dtype)
    b = s * (A.mT @ v).double()

    if gram is None:
        gram = (A * cinv.to(A.dtype).unsqueeze(-1)).mT @ A
    M = s.unsqueeze(-1) * gram.double() * s.unsqueeze(-2)
    M = M + (1.0 + jitter) * torch.eye(K, dtype=torch.float64, device=r.device)
    L = torch.linalg.cholesky(M)
    return chi2_base, b, L, s


def log_like_gaussian_flux_marginal(theta, Y_obs, forward_func, Cinv,
                                    flux_response, flux_prior_std,
                                    gram=None, jitter=0.0):
    """Gaussian log-likelihood with a pixelated flux map marginalized out.

    Model: the data are Gaussian around a forward model that is **linear** in
    a per-pixel flux deviation ``delta_f`` (K pixels) about the parametric
    flux distribution rendered by ``forward_func`` itself,

    .. math::

        y = f(\\theta) + A(\\theta)\\,\\delta f + n,\\qquad
        n \\sim N(0, C_n),\\ C_n^{-1} = \\mathrm{diag(Cinv)},

    with the Gaussian prior ``delta_f ~ N(0, diag(s)^2)``.  Because the prior
    is centered on the *rendered parametric* flux map (e.g. the exponential
    disk at the current ``theta``) rather than on zero, the prior-mean model
    is exactly the standard SuperMAGE forward and this likelihood reduces to
    :func:`log_like_gaussian` as ``s -> 0``.

    Marginalizing ``delta_f`` analytically (Woodbury + matrix determinant
    lemma, all dense algebra in the K-dimensional pixel space) gives

    .. math::

        \\log L(\\theta) = -\\tfrac12\\left(\\tilde r^T \\tilde r
            - b^T M^{-1} b\\right) - \\tfrac12 \\log\\det M + \\mathrm{const},

    with the whitened residual ``r_t = sqrt(Cinv) * (Y_obs - f(theta))``,
    ``B = sqrt(Cinv)[:,None] * A * s[None,:]``, ``b = B^T r_t`` and
    ``M = I_K + B^T B``.  Relative to the ordinary likelihood this adds a
    flux-refit credit ``+ b^T M^{-1} b / 2`` (how much a Gaussian-regularized
    pixel refit could improve chi^2) and the Occam penalty
    ``- log det M / 2`` that prices the extra freedom — both fully
    ``theta``-dependent, so the marginal is a proper likelihood for MCMC /
    nested sampling over the kinematic parameters.  The dropped constant
    (``-N/2 log 2pi - 1/2 log det C_n``) is independent of ``theta`` *and* of
    ``s``, so ``flux_prior_std`` may itself be a sampled hyperparameter.

    Parameters
    ----------
    theta : Tensor, shape (D,)
        Parameter vector.
    Y_obs : Tensor
        Observed data (any shape; complex data should be pre-stacked as
        real/imag, as in the pipeline ``fwd_single`` convention).
    forward_func : callable
        ``theta -> model`` with the prior-mean flux distribution folded
        in (the standard production forward: its parametric flux map IS the
        prior center).
    Cinv : Tensor
        Inverse variance, broadcastable against ``Y_obs``.
    flux_response : Tensor or callable
        ``(N_data, K)`` linear response of the *flattened, unwhitened* model
        to per-pixel flux deviations, columns ``d model / d f_i`` — the dense
        form of the Task-3 ``FluxOperator`` map with the total-flux
        normalization frozen at the current ``theta`` (freezing the
        normalization is what makes the map linear; the total-flux degree of
        freedom then lives in ``delta_f`` and is regularized by the prior).
        Pass a callable ``theta -> A`` for a ``theta``-dependent operator, or
        a fixed tensor for the frozen-kinematics approximation.
    flux_prior_std : float, Tensor (K,), or callable
        Prior std of each pixel's flux deviation.  A useful choice is a
        fractional width ``alpha * f_exp.flatten()``: with ``alpha`` modest
        (<~ 0.3) most prior mass stays positive, which is the practical
        substitute for the non-negativity constraint of the projected-CG
        solve — **positivity itself cannot be kept in closed form** (a
        truncated-Gaussian marginal has no analytic integral); this is the
        one ingredient of the pixelated analysis this likelihood gives up.
    gram : Tensor (K, K), optional
        Precomputed ``A^T diag(Cinv) A`` (unwhitened-by-prior).  When ``A``
        and ``Cinv`` are fixed this removes the O(N_data * K^2) cost per
        call, leaving O(N_data * K + K^3).
    jitter : float, optional
        Relative diagonal jitter added to the unit diagonal of ``M`` for
        Cholesky robustness.

    Returns
    -------
    Tensor (scalar, float64)
        The marginal log-likelihood up to the ``theta``-independent constant
        (same convention as :func:`log_like_gaussian`).  Differentiable
        w.r.t. ``theta`` (through ``forward_func``, a callable
        ``flux_response`` and a callable ``flux_prior_std``) and vmappable,
        so it drops into :func:`mala` unchanged.

    Notes
    -----
    Cost per evaluation: one forward, one ``(K, N_data)`` matvec, plus
    ``O(N_data * K^2)`` for the Gram matrix when ``gram`` is not supplied and
    ``O(K^3)`` for the Cholesky — practical for K up to a few thousand
    (e.g. the 69x69 low-res model grid, K = 4761), not for K ~ 3e5
    (549x549); at that scale a matrix-free treatment would be needed.
    """
    chi2_base, b, L, _ = _flux_marginal_terms(
        theta, Y_obs, forward_func, Cinv, flux_response, flux_prior_std,
        gram=gram, jitter=jitter)
    u = torch.cholesky_solve(b.unsqueeze(-1), L).squeeze(-1)
    quad = (b * u).sum()
    logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum()
    return -0.5 * (chi2_base - quad) - 0.5 * logdet


def flux_marginal_posterior(theta, Y_obs, forward_func, Cinv,
                            flux_response, flux_prior_std,
                            gram=None, jitter=0.0):
    """Conditional posterior of the flux deviation at fixed ``theta``.

    For the same linear-Gaussian model as
    :func:`log_like_gaussian_flux_marginal`, the flux deviation posterior is
    Gaussian in closed form:

    .. math::

        \\delta f \\mid d, \\theta \\sim N\\left(S M^{-1} b,\\;
        S M^{-1} S\\right)

    (``S = diag(s)``).  The full flux map is ``f_exp(theta) + delta_f`` —
    useful as the Bayesian analogue of the Task-3 ``f_opt`` diagnostic (this
    mean is the Gaussian-prior ridge solution rather than the
    non-negativity-projected CG solution).

    Returns
    -------
    (Tensor, Tensor)
        ``delta_f_mean`` of shape ``(K,)`` and posterior covariance of shape
        ``(K, K)``, both float64.
    """
    _, b, L, s = _flux_marginal_terms(
        theta, Y_obs, forward_func, Cinv, flux_response, flux_prior_std,
        gram=gram, jitter=jitter)
    u = torch.cholesky_solve(b.unsqueeze(-1), L).squeeze(-1)
    delta_f_mean = s * u
    Minv = torch.cholesky_inverse(L)
    cov = s.unsqueeze(-1) * Minv * s.unsqueeze(-2)
    return delta_f_mean, cov


def log_prior_tophat(theta, low, high):
    """Flat prior: 0 inside the box ``[low, high]``, ``-inf`` outside.

    Works on a single ``theta``; uses Python control flow, so it is NOT
    vmappable — use :func:`log_prior_tophat_vmappable` inside
    ``torch.func.vmap``.

    Parameters
    ----------
    theta : Tensor, shape (D,)
        Parameter vector.
    low, high : Tensor, shape (D,)
        Box bounds.

    Returns
    -------
    Tensor (scalar)
        ``0.`` or ``-inf``.
    """
    device = low.device
    dtype = low.dtype
    in_box = (theta >= low).all() & (theta <= high).all()
    return torch.tensor(0., device = device, dtype = dtype) if in_box else torch.tensor(-torch.inf, device = device, dtype = dtype)

def log_prior_tophat_vmappable(theta, low, high):
    """torch.where-based flat prior — vmappable AND autograd-compatible.

    Uses ``(theta * 0.).sum()`` to produce a differentiable zero inside the
    box so that the autograd graph is preserved for MALA's backward pass.
    The gradient w.r.t. theta is identically zero everywhere inside the box,
    which is correct for a flat prior.
    """
    in_box = ((theta >= low) & (theta <= high)).all()
    zero   = (theta * 0.).sum()   # grad w.r.t. theta = 0; preserves grad_fn
    return torch.where(
        in_box,
        zero,
        torch.full((), float("-inf"), device=theta.device, dtype=theta.dtype),
    )

def _logp_and_grad_batch(x, log_prob_fn):
    """Sequential fallback: log-prob and gradient for each chain in a loop.

    One stacked forward builds the graph; a single backward yields all
    gradients.  ``x`` is ``(C, D)``; returns ``(logps (C,), grads (C, D))``,
    both detached.
    """
    # one forward builds graph; one backward gives all grads
    x = x.detach().clone().requires_grad_(True)
    logps = torch.stack([log_prob_fn(xi) for xi in x])      # (C,)
    logps.sum().backward()
    grads = x.grad.detach()                                  # (C,D)
    return logps.detach(), grads

def _logp_and_grad_batch_vmap(x, log_prob_fn):
    """
    Vectorised log-prob + gradient via torch.func.vmap + grad_and_value.

    x           : (C, D)
    log_prob_fn : xi -> scalar tensor

    Returns:
        logps : (C,)
        grads : (C, D)
    """
    from torch.func import vmap, grad_and_value

    single_gv = grad_and_value(log_prob_fn)
    grads, logps = vmap(single_gv)(x)

    return logps.detach(), grads.detach()

def mala(
    log_prob_fn,
    init,                        # (C,D) torch
    n_steps=2_000,
    step_size=3e-1,
    mass_matrix=None,            # Σ
    hastings=True,
    progress=True,
    use_vmap: bool = True,
):
    """Batched Metropolis-adjusted Langevin (MALA) sampler.

    Runs ``C`` chains in parallel.  Proposals are
    ``x' = x + 0.5 eps^2 (grad @ Sigma) + eps * L z`` with ``Sigma = L L^T``
    the (preconditioning) mass matrix; the Metropolis-Hastings correction
    makes the chain exact.  Gradients are computed for all chains at once
    with ``torch.func.vmap`` when possible, falling back to a sequential
    loop on failure.

    Parameters
    ----------
    log_prob_fn : callable
        Maps a single ``(D,)`` parameter tensor to a scalar log-probability
        (prior + likelihood); must be autograd-differentiable.  Use
        :func:`log_prior_tophat_vmappable` for box priors when
        ``use_vmap=True``.
    init : Tensor, shape (C, D)
        Initial chain positions.
    n_steps : int, optional
        Number of MALA steps.
    step_size : float, optional
        Step size ``eps``.
    mass_matrix : array-like, optional
        Positive-definite ``(D, D)`` preconditioner ``Sigma`` (posterior
        covariance estimate works well); identity if None.
    hastings : bool, optional
        Apply the MH accept/reject correction (False = unadjusted ULA).
    progress : bool, optional
        Show a tqdm progress bar with running acceptance rate and chi^2.
    use_vmap : bool, optional
        Try vmap-batched gradients first.

    Returns
    -------
    samples : np.ndarray, shape (n_steps, C, D)
    acc_mask : np.ndarray of bool, shape (n_steps, C)
    chi2_trace : np.ndarray, shape (n_steps, C)
        ``-2 * logp`` per chain per step (``inf`` where the prior is violated).
    """
    x = init.detach().clone()
    dtype, device = x.dtype, x.device
    C, D = x.shape

    Σ = torch.eye(D, dtype=dtype, device=device) if mass_matrix is None \
        else torch.as_tensor(mass_matrix, dtype=dtype, device=device)
    L = torch.linalg.cholesky(Σ)

    samples    = torch.empty((n_steps, C, D), dtype=dtype, device=device)
    acc_mask   = torch.empty((n_steps, C),    dtype=torch.bool, device=device)
    chi2_trace = torch.empty((n_steps, C),    dtype=dtype, device=device)

    # cache current logp and grad once
    if use_vmap:
        try:
            logp_cur, grad_cur = _logp_and_grad_batch_vmap(x, log_prob_fn)
        except Exception as e:
            print("[MALA] vmap failed:", repr(e))
            use_vmap = False
            logp_cur, grad_cur = _logp_and_grad_batch(x, log_prob_fn)
    else:
        logp_cur, grad_cur = _logp_and_grad_batch(x, log_prob_fn)

    # RNG (device-local)
    rng = torch.Generator(device=device)
    rng.manual_seed(16)

    it = range(n_steps)
    if progress:
        from tqdm.auto import tqdm
        it = tqdm(it, desc="MALA")

    for t in it:
        eps   = step_size
        mu_x  = x + 0.5 * (eps**2) * (grad_cur @ Σ)                 # (C,D)
        noise = torch.randn(C, D, generator=rng, device=device, dtype=dtype) @ L.T
        x_prop = mu_x + eps * noise

        # single forward+backward at proposal
        if use_vmap:
            try:
                logp_prop, grad_prop = _logp_and_grad_batch_vmap(x_prop, log_prob_fn)
            except Exception:
                use_vmap = False
                logp_prop, grad_prop = _logp_and_grad_batch(x_prop, log_prob_fn)
        else:
            logp_prop, grad_prop = _logp_and_grad_batch(x_prop, log_prob_fn)

        if hastings:
            mu_xp = x_prop + 0.5 * (eps**2) * (grad_prop @ Σ)
            d1 = x      - mu_xp
            d2 = x_prop - mu_x

            # δ^T Σ^{-1} δ via triangular solve
            y1 = torch.linalg.solve_triangular(L, d1.mT, upper=False).mT
            y2 = torch.linalg.solve_triangular(L, d2.mT, upper=False).mT
            q1 = (y1*y1).sum(-1)
            q2 = (y2*y2).sum(-1)

            corr = -0.5 * (q1 - q2) / (eps**2)
            log_alpha = (logp_prop - logp_cur) + corr
        else:
            log_alpha = (logp_prop - logp_cur)

        accept = torch.log(torch.rand(C, device=device, dtype=dtype)) < log_alpha

        # update x, logp, grad where accepted
        x[accept]        = x_prop[accept]
        logp_cur[accept] = logp_prop[accept]
        grad_cur[accept] = grad_prop[accept]

        # record outputs
        samples[t]    = x
        acc_mask[t]   = accept
        # χ² = -2 * logp when prior is finite (top-hat gives 0 inside); becomes +inf if logp=-inf
        chi2_trace[t] = torch.where(torch.isfinite(logp_cur), -2.0 * logp_cur, torch.tensor(float('inf'), device=device, dtype=dtype))

        if progress:
            it.set_postfix(acc_rate=float(acc_mask[:t+1].float().mean()),
                           chi2=float(chi2_trace[t].mean().item()))

    return samples.cpu().numpy(), acc_mask.cpu().numpy(), chi2_trace.cpu().numpy()


# Alias kept for backward-compatibility with scripts that import mala_chi
mala_chi = mala


def checkpointed_mala(
    log_prob_fn,
    init,                        # (C, D) torch tensor — starting positions
    n_steps=2_000,
    step_size=3e-1,
    mass_matrix=None,            # (D, D) covariance Σ for the proposal
    hastings=True,
    progress=True,
    checkpoint_dir=None,         # directory for checkpoint files; None = disabled
    checkpoint_every=500,        # save a checkpoint every this many steps
    resume=True,                 # if True, try to reload an existing checkpoint
    use_vmap: bool = True,
):
    """
    MALA sampler with disk checkpointing.

    Identical to :func:`mala` in algorithm, but periodically saves the full
    sampler state to ``{checkpoint_dir}/mala_checkpoint.npz`` so that a run
    interrupted by a wallclock limit can be resumed without discarding any
    accepted samples.

    Parameters
    ----------
    log_prob_fn : callable
        Maps a 1-D parameter tensor → scalar log-probability (prior + likelihood).
    init : torch.Tensor, shape (C, D)
        Initial chain positions.  Ignored when resuming from a checkpoint.
    n_steps : int
        Total number of MALA steps (including any already completed at resume time).
    step_size : float
        MALA step size ε.
    mass_matrix : array-like or None
        Positive-definite (D, D) mass matrix Σ.  Identity if None.
    hastings : bool
        Whether to apply the Metropolis-Hastings correction.
    progress : bool
        Show a tqdm progress bar.
    checkpoint_dir : str or None
        Directory where ``mala_checkpoint.npz`` is written.  Created if absent.
        Pass ``None`` to disable checkpointing entirely.
    checkpoint_every : int
        Write a checkpoint after every this many completed steps.
    resume : bool
        If True and a checkpoint file exists in ``checkpoint_dir``, resume from it.
    use_vmap : bool
        If True, attempt to use torch.func.vmap for batched gradient computation.
        Falls back to serial loop on failure.

    Returns
    -------
    samples    : np.ndarray, shape (n_steps, C, D)
    acc_mask   : np.ndarray, shape (n_steps, C), dtype bool
    chi2_trace : np.ndarray, shape (n_steps, C)
    """
    import os

    x = init.detach().clone()
    dtype, device = x.dtype, x.device
    C, D = x.shape
    _use_vmap_initial = use_vmap   # remember for backend-summary message

    Σ = torch.eye(D, dtype=dtype, device=device) if mass_matrix is None \
        else torch.as_tensor(mass_matrix, dtype=dtype, device=device)
    L = torch.linalg.cholesky(Σ)

    # Pre-allocate CPU output arrays for the full run
    samples    = np.empty((n_steps, C, D), dtype=np.float32)
    acc_mask   = np.empty((n_steps, C),    dtype=bool)
    chi2_trace = np.empty((n_steps, C),    dtype=np.float32)

    start_step = 0

    # ── Checkpoint path ───────────────────────────────────────────────────────
    ckpt_path = None
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(checkpoint_dir, "mala_checkpoint.npz")

    # ── Resume from checkpoint ────────────────────────────────────────────────
    if ckpt_path and resume and os.path.exists(ckpt_path):
        print(f"[checkpoint] Loading {ckpt_path} …")
        ckpt = np.load(ckpt_path)
        start_step = int(ckpt["step"])

        if start_step >= n_steps:
            print(f"[checkpoint] Run already complete ({start_step}/{n_steps} steps). "
                  f"Returning saved results.")
            return ckpt["samples"], ckpt["acc_mask"], ckpt["chi2_trace"]

        # Restore accumulated outputs
        samples   [:start_step] = ckpt["samples"]   [:start_step]
        acc_mask  [:start_step] = ckpt["acc_mask"]  [:start_step]
        chi2_trace[:start_step] = ckpt["chi2_trace"][:start_step]

        # Restore chain state (move to device)
        x        = torch.tensor(ckpt["x"],        device=device, dtype=dtype)
        logp_cur = torch.tensor(ckpt["logp_cur"], device=device, dtype=dtype)
        grad_cur = torch.tensor(ckpt["grad_cur"], device=device, dtype=dtype)

        # Restore RNG state
        torch.set_rng_state(torch.tensor(ckpt["rng_cpu"], dtype=torch.uint8))
        if "rng_cuda" in ckpt and device.type == 'cuda':
            torch.cuda.set_rng_state(torch.tensor(ckpt["rng_cuda"], dtype=torch.uint8), device)

        print(f"[checkpoint] Resumed at step {start_step}/{n_steps}  "
              f"(χ²_mean={float(chi2_trace[start_step-1].mean()):.4f})")
    else:
        # Fresh start: compute initial logp and gradient
        if use_vmap:
            try:
                logp_cur, grad_cur = _logp_and_grad_batch_vmap(x, log_prob_fn)
            except Exception as _ve:
                print(f"[MALA] vmap unavailable at init "
                      f"({type(_ve).__name__}: {_ve}) — "
                      f"switching to sequential gradient loop.")
                use_vmap = False
                logp_cur, grad_cur = _logp_and_grad_batch(x, log_prob_fn)
        else:
            logp_cur, grad_cur = _logp_and_grad_batch(x, log_prob_fn)

    # ── Backend summary (printed once, after fresh-start or resume) ───────────
    if not use_vmap:
        _why = "use_vmap=False" if not _use_vmap_initial else "vmap failed — see above"
        print(f"[MALA] gradient backend: sequential loop ({_why})")
    else:
        print(f"[MALA] gradient backend: vmap  (C={C}, D={D})")

    # ── Save helper ───────────────────────────────────────────────────────────
    def _save_checkpoint(step_done):
        if ckpt_path is None:
            return
        rng_cpu = torch.get_rng_state().numpy()
        save_kwargs = dict(
            step       = np.array(step_done),
            x          = x.cpu().numpy(),
            logp_cur   = logp_cur.cpu().numpy(),
            grad_cur   = grad_cur.cpu().numpy(),
            samples    = samples,
            acc_mask   = acc_mask,
            chi2_trace = chi2_trace,
            rng_cpu    = rng_cpu,
        )
        if device.type == 'cuda':
            rng_cuda = torch.cuda.get_rng_state(device).numpy()
            save_kwargs["rng_cuda"] = rng_cuda
        np.savez(ckpt_path, **save_kwargs)

    # ── Main sampling loop ────────────────────────────────────────────────────
    it = range(start_step, n_steps)
    if progress:
        from tqdm.auto import tqdm
        it = tqdm(it, initial=start_step, total=n_steps, desc="MALA")

    for t in it:
        eps    = step_size
        mu_x   = x + 0.5 * (eps ** 2) * (grad_cur @ Σ)
        noise  = torch.randn(C, D, device=device, dtype=dtype) @ L.T
        x_prop = mu_x + eps * noise

        if use_vmap:
            try:
                logp_prop, grad_prop = _logp_and_grad_batch_vmap(x_prop, log_prob_fn)
            except Exception as _ve:
                print(f"\n[MALA] vmap failed at step {t} "
                      f"({type(_ve).__name__}: {_ve}) — "
                      f"switching to sequential gradient loop.")
                use_vmap = False
                logp_prop, grad_prop = _logp_and_grad_batch(x_prop, log_prob_fn)
        else:
            logp_prop, grad_prop = _logp_and_grad_batch(x_prop, log_prob_fn)

        if hastings:
            mu_xp = x_prop + 0.5 * (eps ** 2) * (grad_prop @ Σ)
            d1 = x      - mu_xp
            d2 = x_prop - mu_x
            y1 = torch.linalg.solve_triangular(L, d1.mT, upper=False).mT
            y2 = torch.linalg.solve_triangular(L, d2.mT, upper=False).mT
            corr      = -0.5 * ((y1 * y1).sum(-1) - (y2 * y2).sum(-1)) / (eps ** 2)
            log_alpha = (logp_prop - logp_cur) + corr
        else:
            log_alpha = (logp_prop - logp_cur)

        accept = torch.log(torch.rand(C, device=device, dtype=dtype)) < log_alpha

        x[accept]        = x_prop[accept]
        logp_cur[accept] = logp_prop[accept]
        grad_cur[accept] = grad_prop[accept]

        # Store to CPU arrays
        samples[t]    = x.cpu().numpy()
        acc_mask[t]   = accept.cpu().numpy()
        chi2_trace[t] = torch.where(
            torch.isfinite(logp_cur),
            -2.0 * logp_cur,
            torch.full_like(logp_cur, float("inf")),
        ).cpu().numpy()

        if progress:
            it.set_postfix(
                acc=f"{acc_mask[start_step:t+1].mean():.3f}",
                chi2=f"{chi2_trace[t].mean():.2f}",
            )

        # Periodic checkpoint
        if ckpt_path and (t + 1) % checkpoint_every == 0:
            _save_checkpoint(t + 1)

    # Final checkpoint
    _save_checkpoint(n_steps)

    return samples, acc_mask, chi2_trace
