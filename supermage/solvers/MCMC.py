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

def _logp_and_grad_batch(x, log_prob_fn):
    # one forward builds graph; one backward gives all grads
    x = x.detach().clone().requires_grad_(True)
    logps = torch.stack([log_prob_fn(xi) for xi in x])      # (C,)
    logps.sum().backward()
    grads = x.grad.detach()                                  # (C,D)
    return logps.detach(), grads

def mala(
    log_prob_fn,
    init,                        # (C,D) torch
    n_steps=2_000,
    step_size=3e-1,
    mass_matrix=None,            # Σ
    hastings=True,
    progress=True,
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