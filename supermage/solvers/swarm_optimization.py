import math, torch
from torch.quasirandom import SobolEngine
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────
# 1. Sobol' sampler for uniform priors
# ──────────────────────────────────────────────────────────────
def sobol_sample(param_bounds,           # list/tuple/array of shape (D, 2)
                 n_samples, 
                 *, 
                 dtype=torch.float64,
                 device="cpu",
                 scramble=True,
                 seed=None):
    """
    Draw `n_samples` Sobol' points within hyper‑rectangular bounds.
    
    Parameters
    ----------
    param_bounds : sequence[(low, high), …] or (D, 2) tensor
        Uniform prior box for each parameter.
    n_samples    : int
        Number of Sobol' points to generate.
    dtype, device, scramble, seed : usual Torch options.
    
    Returns
    -------
    samples : Tensor[n_samples, D]  in the requested dtype/device
    """
    bounds = torch.as_tensor(param_bounds, dtype=dtype, device=device)
    low, high = bounds[:, 0], bounds[:, 1]
    engine = SobolEngine(dimension=len(bounds), scramble=scramble, seed=seed)
    u = engine.draw(n_samples).to(dtype=dtype, device=device)           # ∈ [0,1)
    return low + (high - low) * u                                       # rescale


# ──────────────────────────────────────────────────────────────
# 2. Serial (memory‑friendly) Sobol' swarm wrapper
# ──────────────────────────────────────────────────────────────
def sobol_swarm_opt(lm_fn,                    # your lm_direct
                    param_bounds, 
                    n_particles,
                    *,
                    dtype      = torch.float64,
                    device     = "cpu",
                    lm_kwargs  = None,        # extra kwargs to lm_fn
                    verbose    = True,
                    seed       = 42):
    """
    Runs `lm_fn` once for each Sobol‑initialised particle.
    Only one particle lives in memory at any moment.
    
    Returns
    -------
    best_X   : Tensor[D]      – parameters of the best run
    best_val : float or Tensor (scalar) – objective value (e.g. χ²) of the best run
    history  : list of dicts  – log of every particle
    """
    if lm_kwargs is None:
        lm_kwargs = {}

    # 2‑a. Generate Sobol' particles
    particles = sobol_sample(param_bounds, n_particles, 
                             dtype=dtype, device=device, seed = 16)
    
    best_val = torch.tensor(float("inf"), dtype=dtype, device=device)
    best_X   = None
    best_L   = None
    history  = []

    # 2‑b. Serial optimisation loop
    for i, x0 in enumerate(particles, 1):
        # Important: clone so lm_fn can modify the tensor safely
        result = lm_fn(x0.clone(), **lm_kwargs)
        X_opt, L_opt, chi2_opt = result                  # your lm_direct returns these

        # Log & book‑keeping
        history.append(dict(idx=i, X=X_opt.detach().cpu(), 
                            chi2=float(chi2_opt), L=float(L_opt)))
        if chi2_opt < best_val:
            best_val = chi2_opt.detach()
            best_X   = X_opt.detach().clone()
            best_L   = L_opt

        if verbose:
            print(f"[{i:>3}/{n_particles}] χ²={chi2_opt.item():.4g}   "
                  f"best={best_val.item():.4g}")

    return best_X, best_val, best_L, history



# ──────────────────────────────────────────────────────────────
# 3. Serial (memory‑friendly) Sobol' swarm wrapper
# ──────────────────────────────────────────────────────────────
def batch_chi2(points, *, Y, f, Cinv, batch_size=128):
    """
    Compute χ² for every point in `points` without building grads.
    Returns a 1‑D tensor of shape (N,).
    """
    dataset = TensorDataset(points)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    chi2_all = []
    with torch.no_grad():                      # no autograd graph
        for (x,) in loader:
            fY = f(x)                          # (B, …)
            dY = Y.expand_as(fY) - fY          # broadcast if needed
            chi2 = (dY**2 * Cinv).sum(dim=tuple(range(1, dY.ndim)))  # (B,)
            chi2_all.append(chi2)
    return torch.cat(chi2_all)                 # (N,)

def iterated_chi2(points, *, Y, f, Cinv):
    """
    Compute χ² for every Sobol scout individually (no batching).
    Suitable when the forward model only accepts (D,) inputs.
    """
    chi2_list = []
    with torch.no_grad():
        for i in tqdm(range(len(points)), desc="Pruning Sobol' swarm"): 
            fY   = f(points[i])                      # same shape as Y
            dY   = Y - fY
            chi2 = (dY**2 * Cinv).sum()
            chi2_list.append(chi2)
    return torch.stack(chi2_list)            # (N_scouts,)

# ---------------------------------------------------------------------
# 3.1. Sobol + “genetic” pre‑selection + LM loop
# ---------------------------------------------------------------------
def sobol_ga_swarm_opt(
        lm_fn,                   # your lm_cg_og or lm_direct
        param_bounds,
        n_particles,
        n_chi,
        *,
        oversample   = 10,       # generate oversample × n_particles scouts
        keep_frac    = 0.10,     # keep this fraction (≈ n_particles)
        eval_bs      = 128,      # batch‑size for χ² scouting
        dtype        = torch.float64,
        device       = "cpu",
        lm_kwargs    = None,
        verbose      = True,
        seed         = 42,
):
    """
    Two‑stage optimisation:
      1. Draw oversample×n_particles Sobol points, keep the best keep_frac.
      2. Run `lm_fn` once on each survivor (serially, so memory stays low).

    Returns
    -------
    best_X, best_chi2, best_L, history   (same as sobol_swarm_opt)
    """
    if lm_kwargs is None:
        lm_kwargs = {}

    # unpack items needed to compute χ² quickly
    Y = lm_kwargs["Y"]
    f = lm_kwargs["f"]
    C = lm_kwargs.get("C", None)

    # --- build Cinv ONCE (matches what lm_fn will do) -----------------
    if C is None:
        Cinv = torch.ones_like(Y, dtype=dtype, device=device)
    elif C.ndim == 1:
        Cinv = 1.0 / C.to(dtype=dtype, device=device)
    else:
        Cinv = torch.linalg.inv(C.to(dtype=dtype, device=device))

    # ------------------------------------------------------------------
    # Stage 1 – oversampled scouting
    # ------------------------------------------------------------------
    n_scouts = int(math.ceil(oversample * n_particles))
    engine   = SobolEngine(dimension=len(param_bounds), scramble=True, seed=seed)

    bounds = torch.as_tensor(param_bounds, dtype=dtype, device=device)
    low, high = bounds[:, 0], bounds[:, 1]

    scouts_u = engine.draw(n_scouts).to(dtype=dtype, device=device)
    scouts   = low + (high - low) * scouts_u       # (n_scouts, D)

    # fast χ² evaluation (no grads)
    chi2_scouts = iterated_chi2(
        scouts, Y=Y.to(device), f=f, Cinv=Cinv,
    )

    # keep the best ‘keep_frac’ fraction
    k = max(1, int(keep_frac * n_scouts))
    top_vals, top_idx = torch.topk(-chi2_scouts, k)  # negative → ascending
    survivors = scouts[top_idx]                      # (k, D)

    if verbose:
        best_raw = (-top_vals).min().item()
        print(f"⇢ Scouted {n_scouts} points, best raw χ² = {best_raw/n_chi:.4g}")
        print(f"⇢ Keeping {k} best points for LM refinement …")
        print(top_vals.cpu().numpy()/n_chi)

    # ------------------------------------------------------------------
    # Stage 2 – serial LM refinement
    # ------------------------------------------------------------------
    history  = []
    best_val = torch.tensor(float("inf"), dtype=dtype, device=device)
    best_X   = None
    best_L   = None

    for i, x0 in enumerate(survivors, 1):
        res = lm_fn(x0.clone(), **lm_kwargs)           # (X, L, chi2)
        X_opt, L_opt, chi2_opt = res

        history.append(dict(idx=i,
                            X=X_opt.detach().cpu(),
                            chi2=float(chi2_opt),
                            L=float(L_opt)))

        if chi2_opt < best_val:
            best_val = chi2_opt.detach()
            best_X   = X_opt.detach().clone()
            best_L   = L_opt

        if verbose:
            print(f"[{i:>3}/{k}] χ²={chi2_opt.item():.4g}   "
                  f"best={best_val.item():.4g}")

    return best_X, best_val, best_L, history

def sobol_ga_swarm_no_nan(
        lm_fn,                   # your lm_cg_og or lm_direct
        param_bounds,
        n_particles,
        n_chi,
        *,
        oversample   = 10,       # generate oversample × n_particles scouts
        keep_frac    = 0.10,     # keep this fraction (≈ n_particles)
        eval_bs      = 128,      # batch‑size for χ² scouting
        dtype        = torch.float64,
        device       = "cpu",
        lm_kwargs    = None,
        verbose      = True,
        seed         = 42,
):
    """
    Two‑stage optimisation:
      1. Draw oversample×n_particles Sobol points, keep the best keep_frac.
      2. Run `lm_fn` once on each survivor (serially, so memory stays low).

    Returns
    -------
    best_X, best_chi2, best_L, history   (same as sobol_swarm_opt)
    """
    if lm_kwargs is None:
        lm_kwargs = {}

    # unpack items needed to compute χ² quickly
    Y = lm_kwargs["Y"]
    f = lm_kwargs["f"]
    C = lm_kwargs.get("C", None)

    # --- build Cinv ONCE (matches what lm_fn will do) -----------------
    if C is None:
        Cinv = torch.ones_like(Y, dtype=dtype, device=device)
    elif C.ndim == 1:
        Cinv = 1.0 / C.to(dtype=dtype, device=device)
    else:
        Cinv = torch.linalg.inv(C.to(dtype=dtype, device=device))

    # ------------------------------------------------------------------
    # Stage 1 – oversampled scouting
    # ------------------------------------------------------------------
    n_scouts = int(math.ceil(oversample * n_particles))
    engine   = SobolEngine(dimension=len(param_bounds), scramble=True, seed=seed)

    bounds = torch.as_tensor(param_bounds, dtype=dtype, device=device)
    low, high = bounds[:, 0], bounds[:, 1]

    scouts_u = engine.draw(n_scouts).to(dtype=dtype, device=device)
    scouts   = low + (high - low) * scouts_u       # (n_scouts, D)

    # fast χ² evaluation (no grads)
    chi2_scouts = iterated_chi2(
        scouts, Y=Y.to(device), f=f, Cinv=Cinv,
    )

    # --- Start of Patch ---
    # Filter out NaNs from the scouts' chi2 values
    valid_mask = ~torch.isnan(chi2_scouts)
    valid_chi2_scouts = chi2_scouts[valid_mask]
    valid_scouts = scouts[valid_mask]

    if verbose and not torch.all(valid_mask):
        print(f"⇢ Filtered out {torch.sum(~valid_mask)} NaN values.")

    # keep the best ‘keep_frac’ fraction from the valid scouts
    k = max(1, int(keep_frac * n_scouts))
    
    # Ensure k is not larger than the number of valid scouts
    num_valid_scouts = valid_scouts.shape[0]
    k = min(k, num_valid_scouts)

    if k > 0:
        top_vals, top_idx = torch.topk(-valid_chi2_scouts, k)  # negative → ascending
        survivors = valid_scouts[top_idx]                      # (k, D)
    else:
        # Handle the case where there are no valid scouts left
        survivors = torch.empty((0, scouts.shape[1]), dtype=dtype, device=device)
    # --- End of Patch ---

    if verbose:
        if k > 0:
            best_raw = (-top_vals).min().item()
            print(f"⇢ Scouted {n_scouts} points, found {num_valid_scouts} valid, best raw χ² = {best_raw/n_chi:.4g}")
            print(f"⇢ Keeping {k} best points for LM refinement …")
            print((-top_vals).cpu().numpy()/n_chi)
        else:
            print(f"⇢ Scouted {n_scouts} points, but no valid (non-NaN) points were found.")


    # ------------------------------------------------------------------
    # Stage 2 – serial LM refinement
    # ------------------------------------------------------------------
    history  = []
    best_val = torch.tensor(float("inf"), dtype=dtype, device=device)
    best_X   = None
    best_L   = None

    for i, x0 in enumerate(survivors, 1):
        res = lm_fn(x0.clone(), **lm_kwargs)           # (X, L, chi2)
        X_opt, L_opt, chi2_opt = res

        history.append(dict(idx=i,
                            X=X_opt.detach().cpu(),
                            chi2=float(chi2_opt),
                            L=float(L_opt)))

        if chi2_opt < best_val:
            best_val = chi2_opt.detach()
            best_X   = X_opt.detach().clone()
            best_L   = L_opt

        if verbose:
            print(f"[{i:>3}/{k}] χ²={chi2_opt.item():.4g}   "
                  f"best={best_val.item():.4g}")

    return best_X, best_val, best_L, history


# ---------------------------------------------------------------------
# 4. Efficient variant: DataLoader-based parallel scouting + memory flush
# ---------------------------------------------------------------------
def sobol_ga_swarm_efficient(
        lm_fn,
        param_bounds,
        n_particles,
        n_chi,
        *,
        oversample   = 10,       # generate oversample × n_particles scouts
        keep_frac    = 0.10,     # keep this fraction after scouting
        n_workers    = 2,        # DataLoader prefetch workers
        dtype        = torch.float64,
        device       = "cpu",
        lm_kwargs    = None,
        verbose      = True,
        seed         = 42,
):
    """
    Two-stage optimisation identical to ``sobol_ga_swarm_no_nan``, with two
    efficiency improvements:

    **Stage 1 – DataLoader-based parallel prefetch scouting**
        Scouts are wrapped in a ``TensorDataset`` / ``DataLoader`` (mirroring
        ``batch_chi2``).  ``n_workers`` background processes prefetch the next
        scout parameter vector while the GPU is busy with the current forward
        pass, overlapping CPU↔GPU transfers with compute.  ``pin_memory=True``
        is enabled automatically when running on CUDA so the async transfer
        path is used.  Each chi² scalar is moved to CPU immediately after
        evaluation, releasing the forward-pass activations before the next
        scout starts.

    **Stage 1→2 memory flush**
        Once the top-k survivors have been selected, *all* scouting tensors
        (scouts, chi² values, Cinv, …) are explicitly deleted and
        ``torch.cuda.empty_cache()`` is called before entering the LM loop.
        This returns the full GPU memory budget to the Jacobian allocations
        inside the LM solver.  ``torch.cuda.empty_cache()`` is also called
        after every individual LM run to prevent fragmentation.

    Parameters
    ----------
    n_workers : int
        Number of DataLoader worker processes for scout prefetching.
        0 disables multiprocessing (useful for debugging).

    All other parameters are identical to ``sobol_ga_swarm_no_nan``.

    Returns
    -------
    best_X, best_chi2, best_L, history
    """
    if lm_kwargs is None:
        lm_kwargs = {}

    Y = lm_kwargs["Y"]
    f = lm_kwargs["f"]
    C = lm_kwargs.get("C", None)

    # Build Cinv once on device
    if C is None:
        Cinv = torch.ones_like(Y, dtype=dtype, device=device)
    elif C.ndim == 1:
        Cinv = 1.0 / C.to(dtype=dtype, device=device)
    else:
        Cinv = torch.linalg.inv(C.to(dtype=dtype, device=device))

    Y_dev    = Y.to(device)
    Cinv_dev = Cinv.to(device)

    # ------------------------------------------------------------------
    # Stage 1 – Sobol sampling  (scouts kept on CPU for DataLoader workers)
    # ------------------------------------------------------------------
    n_scouts = int(math.ceil(oversample * n_particles))
    engine   = SobolEngine(dimension=len(param_bounds), scramble=True, seed=seed)
    bounds   = torch.as_tensor(param_bounds, dtype=dtype)          # CPU
    low, high = bounds[:, 0], bounds[:, 1]
    scouts_u  = engine.draw(n_scouts).to(dtype=dtype)              # CPU
    scouts_cpu = low + (high - low) * scouts_u                     # (N, D) CPU

    # ------------------------------------------------------------------
    # Stage 1 – DataLoader-based chi² scouting
    # DataLoader workers prefetch scout vectors on CPU while the GPU runs
    # the previous forward pass, overlapping transfer with compute.
    # ------------------------------------------------------------------
    use_pin = (device != "cpu") and torch.cuda.is_available()
    dataset = TensorDataset(scouts_cpu)
    loader  = DataLoader(
        dataset,
        batch_size  = 1,
        shuffle     = False,
        num_workers = n_workers,
        pin_memory  = use_pin,
    )

    chi2_list = []
    with torch.no_grad():
        for (x_batch,) in tqdm(loader, desc="Scouting"):
            # squeeze batch dim; non_blocking overlaps transfer with GPU work
            x_i  = x_batch.squeeze(0).to(device, non_blocking=use_pin)
            fY   = f(x_i)
            dY   = Y_dev - fY
            chi2 = (dY ** 2 * Cinv_dev).sum()
            chi2_list.append(chi2.cpu())   # CPU immediately → frees activations

    chi2_scouts = torch.stack(chi2_list)   # (N,)  on CPU

    # Keep scouts on CPU for NaN filtering, then move survivors to device
    valid_mask   = ~torch.isnan(chi2_scouts)
    valid_chi2   = chi2_scouts[valid_mask]
    valid_scouts = scouts_cpu[valid_mask]  # still CPU

    n_valid = int(valid_mask.sum())
    if verbose and n_valid < n_scouts:
        print(f"⇢ Filtered out {n_scouts - n_valid} NaN scouts.")

    k = min(max(1, int(keep_frac * n_scouts)), n_valid)

    if k > 0:
        top_vals, top_idx = torch.topk(-valid_chi2, k)   # ascending
        # Move survivors to device for LM
        survivors = valid_scouts[top_idx].to(dtype=dtype, device=device)
    else:
        survivors = torch.empty((0, scouts_cpu.shape[1]), dtype=dtype, device=device)

    if verbose:
        if k > 0:
            best_raw = (-top_vals).min().item()
            print(f"⇢ Scouted {n_scouts} points, {n_valid} valid, "
                  f"best raw χ²/n = {best_raw / n_chi:.4g}")
            print(f"⇢ Keeping {k} survivors for LM refinement …")
            print((-top_vals).cpu().numpy() / n_chi)
        else:
            print(f"⇢ Scouted {n_scouts} points — no valid (non-NaN) scouts found.")

    # ------------------------------------------------------------------
    # Memory flush before LM
    # ------------------------------------------------------------------
    del scouts_cpu, scouts_u, chi2_scouts, valid_chi2, valid_mask, valid_scouts
    del chi2_list, Y_dev, Cinv_dev, Cinv
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if verbose:
        print("⇢ GPU cache cleared — starting LM refinement.")

    # ------------------------------------------------------------------
    # Stage 2 – serial LM refinement on survivors
    # ------------------------------------------------------------------
    history  = []
    best_val = torch.tensor(float("inf"), dtype=dtype, device=device)
    best_X   = None
    best_L   = None

    for i, x0 in enumerate(survivors, 1):
        res = lm_fn(x0.clone(), **lm_kwargs)
        X_opt, L_opt, chi2_opt = res

        history.append(dict(
            idx  = i,
            X    = X_opt.detach().cpu(),
            chi2 = float(chi2_opt),
            L    = float(L_opt),
        ))

        if chi2_opt < best_val:
            best_val = chi2_opt.detach()
            best_X   = X_opt.detach().clone()
            best_L   = L_opt

        if verbose:
            print(f"[{i:>3}/{k}] χ²/n={chi2_opt.item()/n_chi:.4g}   "
                  f"best={best_val.item()/n_chi:.4g}")

        # Clear LM temporaries between runs to reduce fragmentation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return best_X, best_val, best_L, history