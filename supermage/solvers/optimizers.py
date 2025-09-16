import torch
import numpy as np
from torch.func import jvp, vjp
from torch.func import jacrev, jacfwd  # (PyTorch ≥2.0; for older versions import from functorch)
from torch.autograd import grad as torch_grad

def cg_solve(hvp, b, x0=None, tol=1e-6, maxiter=20):
    if x0 is None:
        x = torch.zeros_like(b)
    else:
        x = x0.clone()
    r = b - hvp(x)
    p = r.clone()
    rs_old = torch.dot(r, r)
    for i in range(maxiter):
        Ap = hvp(p)
        alpha = rs_old / torch.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.dot(r, r)
        if torch.sqrt(rs_new) < tol:
            break
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new
    return x


def lm_cg_autograd_stable(
    X, Y, f, n_chi,
    C=None,
    epsilon=1e-1, L=1.0, L_dn=11.0, L_up=9.0,
    max_iter=50, cg_maxiter=20, cg_tol=1e-6,
    L_min=1e-9, L_max=1e9,
    stopping=1e-4,
    verbose=True,
):
    if C is None:
        Cinv = torch.ones_like(Y)
    elif C.ndim == 1:
        Cinv = 1.0 / C
    else:
        Cinv = torch.linalg.inv(C)

    # Use torch.autograd.grad for explicit gradient calculation
    from torch.autograd import grad as torch_grad

    def get_loss_and_grad(x_in):
        x_ = x_in.detach().requires_grad_(True)
        fX = f(x_)
        dY = Y - fX
        chi2 = (dY**2 * Cinv).sum()
        loss = 0.5 * chi2
        grad = torch_grad(loss, x_, allow_unused=True)[0]
        return chi2.detach(), loss.detach(), grad

    def hvp(v, current_x, current_L):
        x_ = current_x.detach().requires_grad_(True)
        fX = f(x_)
        dY = Y - fX
        loss = 0.5 * (dY**2 * Cinv).sum()
        grad_L = torch_grad(loss, x_, create_graph=True, allow_unused=True)[0]
        hvp_L = torch_grad((grad_L * v).sum(), x_, retain_graph=True, allow_unused=True)[0]
        return hvp_L + current_L * v

    chi2, _, grad = get_loss_and_grad(X)

    if verbose:
        print(f"{'Iter':>4} | {'chi2/n':>12} | {'chi2_new/n':>12} | {'λ':>8} | {'ρ':>6} | {'acc':>5}")
        print("-"*65)

    for it in range(max_iter):
        hvp_for_cg = lambda v: hvp(v, X, L)
        h = cg_solve(hvp_for_cg, -grad, maxiter=cg_maxiter, tol=cg_tol)

        chi2_new, _, _ = get_loss_and_grad(X + h)

        actual_reduction = chi2 - chi2_new
        pred_reduction = -torch.dot(h, grad) - 0.5 * torch.dot(h, hvp(h, X, 0.0))
        
        # --- START: THIS IS THE FIX ---
        # A valid step should result in a positive predicted reduction.
        # If not, the quadratic approximation is poor, and the step should be rejected.
        if pred_reduction > 0:
            rho = actual_reduction / pred_reduction
        else:
            # Force rejection of the step if the model predicts an increase in chi-squared
            rho = torch.tensor(-1.0) 
        # --- END: THIS IS THE FIX ---

        accepted = (rho >= epsilon)
        if accepted:
            X, chi2 = X + h, chi2_new
            L = max(L / L_dn, L_min)
            _, _, grad = get_loss_and_grad(X) # Recalculate gradient at the new point
        else:
            L = min(L * L_up, L_max)

        if verbose:
            # Use chi2/n_chi for the current accepted value in the printout
            current_chi2_val = chi2 if accepted else get_loss_and_grad(X)[0]
            print(f"{it:4d} | {current_chi2_val.item()/n_chi:12.4e} | {chi2_new.item()/n_chi:12.4e} | {L:8.2e} | {rho.item():6.3f} | {str(bool(accepted)):>5}")

        if torch.norm(h) < stopping:
            if verbose: print("Stopping: Step size below threshold.")
            break
            
        if torch.isnan(rho) or rho.item() < -100: # break on divergence
            if verbose: print("Stopping: Rho is NaN or diverging.")
            break

    return X, L, chi2

def lm_direct(
    X, Y, f,
    n_chi,
    C=None,
    epsilon=1e-1, L=1.0, L_dn=11.0, L_up=9.0,
    max_iter=50, cg_maxiter=20, cg_tol=1e-6,   # cg_* kept for signature compatibility (unused)
    L_min=1e-9, L_max=1e9,
    stopping=1e-8,
    verbose=True,
):
    """
    Dense (direct solve) Levenberg–Marquardt matching lm_cg signature but without CG.

    Parameters are identical to lm_cg; cg_maxiter & cg_tol are ignored.

    Returns
    -------
    X      : optimised parameter vector
    L      : final damping value
    chi2   : final chi^2 (scalar tensor)
    """

    # Clone to avoid in-place modification of caller's tensor
    X = X.clone()

    # ------------------------------------------------------------------
    # Prepare inverse covariance / weights
    # ------------------------------------------------------------------
    if C is None:
        # Diagonal weights = 1
        Cinv = torch.ones_like(Y)
        is_diag = True
    elif C.ndim == 1:
        Cinv = 1.0 / C
        is_diag = True
    else:
        Cinv = torch.linalg.inv(C)
        is_diag = False

    def forward_residual(x):
        fY = f(x)
        dY = Y - fY
        return fY, dY

    def chi2_from_residual(dY):
        if is_diag:
            return (dY**2 * Cinv).sum()
        else:
            return (dY @ Cinv @ dY)

    # Initial χ²
    fY, dY = forward_residual(X)
    chi2 = chi2_from_residual(dY)

    if verbose:
        print(f"{'Iter':>4} | {'chi2':>12} | {'chi2_new':>12} | {'λ':>8} | {'ρ':>6} | {'acc':>4}")
        print("-"*60)

    Din = X.numel()
    eye = torch.eye(Din, device=X.device, dtype=X.dtype)

    for it in range(max_iter):
        torch.cuda.empty_cache()
        # --------------------------------------------------------------
        # Jacobian J : (Dout, Din)
        # --------------------------------------------------------------
        # jacfwd returns shape of output + shape of input -> (Dout, Din)
        J = jacfwd(f)(X)
        if J.ndim != 2:
            J = J.reshape(-1, Din)  # flatten any structured output just in case
        Dout = J.shape[0]

        # --------------------------------------------------------------
        # Build RHS (called 'grad' in your code) = J^T W dY
        # --------------------------------------------------------------
        if is_diag:
            w_dY = Cinv * dY           # (Dout,)
            grad = J.T @ w_dY          # (Din,)
            # Hessian (Gauss–Newton) H = J^T diag(Cinv) J
            # (Multiply each row of J by sqrt weights, or by weights then J^T)
            # Use broadcasting for efficiency:
            H = J.T @ (J * Cinv.view(-1, 1))
        else:
            w_dY = Cinv @ dY           # (Dout,)
            grad = J.T @ w_dY
            H = J.T @ Cinv @ J         # (Din, Din)

        # Damped system: (H + L I) h = grad
        H_damped = H + L * eye

        # Solve
        h = torch.linalg.solve(H_damped, grad)

        if h.ndim > 1:  # ensure vector
            h = h.squeeze(-1)

        # --------------------------------------------------------------
        # Candidate update
        # --------------------------------------------------------------
        fY_new, dY_new = forward_residual(X + h)
        chi2_new = chi2_from_residual(dY_new)

        # Expected improvement (match cg version): h^T[(H + L I)h + grad]
        expected = h @ (H_damped @ h + grad)
        if expected.abs() < 1e-32:
            rho = torch.tensor(0.0, device=X.device, dtype=X.dtype)
        else:
            rho = (chi2 - chi2_new) / expected.abs()

        accepted = (rho >= epsilon)
        if accepted:
            X = X + h
            chi2 = chi2_new
            L = max(L / L_dn, L_min)
            # Recompute residual for next iteration (lazy update okay)
            fY, dY = fY_new, dY_new
        else:
            L = min(L * L_up, L_max)

        if verbose:
            i = 2 if Din > 2 else 0
            H_ii = H[i, i].item()
            #print(f"param[{i}]: grad={grad[i].item():.3e}, H_ii={H_ii:.3e}, L={L:.3e}")
            print(f"{it:4d} | {chi2.item()/n_chi:12.4e} | {chi2_new.item()/n_chi:12.4e} | {L:8.2e} | {rho.item():6.3f} | {str(bool(accepted)):>4} ")

        # Stopping criterion
        if torch.norm(h) < stopping:
            break
        if L >= L_max:
            break

    return X, L, chi2


def adam_optimizer(
    X_init, Y, f,
    C=None,
    max_iter=300,
    lr=1e-2,
    T_start=None,
    T_end=None,
    n_chi=None,          # ignored for now, kept for compatibility
    verbose=True,
):
    X = X_init.clone().detach().requires_grad_(True)

    # Prepare inverse covariance (assume diagonal or full)
    if C is None:
        Cinv = torch.ones_like(Y)
    elif C.ndim == 1:
        Cinv = 1.0 / C
    else:
        Cinv = torch.linalg.inv(C)

    # Gaussian χ² function
    def chi2_fn(X):
        fY = f(X)
        dY = Y - fY
        return (dY**2 * Cinv).sum()

    # Adam optimizer setup
    optimizer = torch.optim.Adam([X], lr=lr)

    # Cosine annealing temperature schedule (for optional noise injection)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter, eta_min=lr * 0.1)

    for it in range(max_iter):
        optimizer.zero_grad()
        chi2 = chi2_fn(X)
        chi2.backward()

        if torch.isnan(X.grad).any() or torch.isnan(chi2):
            print("NaNs encountered, stopping early.")
            break

        optimizer.step()
        scheduler.step()

        # Optional: temperature annealing for simulated annealing behavior
        if T_start is not None and T_end is not None:
            T = T_end + 0.5 * (T_start - T_end) * (1 + torch.cos(torch.tensor(it / max_iter * 3.14159)))
            noise = torch.randn_like(X) * T.sqrt() * 1e-4
            with torch.no_grad():
                X.add_(noise)

        if verbose and it%10==0:
            print(f"{it:4d} | chi² = {chi2.item()/n_chi:.4e} | M_bh : {X[2].item():.4f}")
            print(X.detach())

    return X.detach(), None, chi2_fn(X.detach())