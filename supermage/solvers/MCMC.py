import math, torch, numpy as np

def log_like_gaussian(theta, Y_obs, forward_func, Cinv):
    """
    theta is a (D,) torch Tensor.
    The forward model `forward_flat` must return a tensor with same
    shape & device as Y_obs.
    """
    fY   = forward_func(theta)
    dY   = Y_obs - fY
    chi2 = (dY.square() * Cinv).sum()
    return -0.5 * chi2

def log_prior_tophat(theta, low, high):
    """Flat prior inside the box, –inf outside (works on a single θ)."""
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
            except Exception:
                use_vmap = False
                logp_cur, grad_cur = _logp_and_grad_batch(x, log_prob_fn)
        else:
            logp_cur, grad_cur = _logp_and_grad_batch(x, log_prob_fn)

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
            except Exception:
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
