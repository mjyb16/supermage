import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Literal, Optional, Tuple
from torch import pi, sqrt
from torch.special import modified_bessel_i0, modified_bessel_i1, modified_bessel_k0, modified_bessel_k1
from caskade import Module, forward, Param
import numpy as np
from numpy.polynomial.legendre import leggauss
from torch.nn.functional import conv2d, avg_pool2d
from functools import lru_cache
import math
import joblib


@lru_cache(maxsize=None)
def _leggauss_const(n, dtype, device):
    x_np, w_np = np.polynomial.legendre.leggauss(n)
    return (torch.as_tensor(x_np, dtype=dtype, device = device),
            torch.as_tensor(w_np, dtype=dtype, device = device))

# 2.  Pure-Torch mapping keeps autograd alive and avoids graph breaks.
def leggauss_interval(n, t_low, t_high, device=None, dtype=None):
    x0, w0 = _leggauss_const(n, dtype, device)
    half_const = torch.tensor(0.5, dtype=dtype, device=device)

    half = half_const * (t_high - t_low)
    mid  = half_const * (t_high + t_low)

    # allow t_low / t_high to be batched – add a dim for broadcasting
    x = half.unsqueeze(-1) * x0 + mid.unsqueeze(-1)
    w = half.unsqueeze(-1) * w0
    return x, w

def _h_poly(t):
    """Helper function to compute the 'h' polynomial matrix used in the
    cubic spline.

    Parameters
    ----------
    t: Tensor
        A 1D tensor representing the normalized x values.

    Returns
    -------
    Tensor
        A 2D tensor of size (4, len(t)) representing the 'h' polynomial matrix.

    """

    tt = t[None, :] ** (torch.arange(4, device=t.device)[:, None])
    A = torch.tensor(
        [[1, 0, -3, 2], [0, 1, -2, 1], [0, 0, 3, -2], [0, 0, -1, 1]],
        dtype=t.dtype,
        device=t.device,
    )
    return A @ tt

### A modified version of the caustics interpolation
def interp1d(
    x: Tensor,
    y: Tensor,
    xs: Tensor,
    extend: Literal["extrapolate", "const", "linear"] = "extrapolate",
) -> Tensor:
    """Compute the 1D cubic spline interpolation for the given data points
    using PyTorch.

    Parameters
    ----------
    x: Tensor
        A 1D tensor representing the x-coordinates of the known data points.
    y: Tensor
        A 1D tensor representing the y-coordinates of the known data points.
    xs: Tensor
        A 1D tensor representing the x-coordinates of the positions where
        the cubic spline function should be evaluated.
    extend: (str, optional)
        The method for handling extrapolation, either "const", "extrapolate", or "linear".
        Default is "extrapolate".
        "const": Use the value of the last known data point for extrapolation.
        "linear": Use linear extrapolation based on the last two known data points.
        "extrapolate": Use cubic extrapolation of data.

    Returns
    -------
    Tensor
        A 1D tensor representing the interpolated values at the specified positions (xs).

    """
    m = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    m = torch.cat([m[[0]], (m[1:] + m[:-1]) / 2, m[[-1]]])
    idxs = torch.searchsorted(x[:-1].contiguous(), xs) - 1
    dx = x[idxs + 1] - x[idxs]
    hh = _h_poly((xs - x[idxs]) / dx)
    ret = hh[0] * y[idxs] + hh[1] * m[idxs] * dx + hh[2] * y[idxs + 1] + hh[3] * m[idxs + 1] * dx  # fmt: skip
    if extend == "const":
        ret[xs > x[-1]] = y[-1]
    elif extend == "linear":
        indices = xs > x[-1]
        ret[indices] = y[-1] + (xs[indices] - x[-1]) * (y[-1] - y[-2]) / (x[-1] - x[-2])
    return ret

def transform_DE(t):
    """
    Double-exponential transform:
      u = exp((π/2) * sinh(t)),
      du/dt = (π/2)*cosh(t)*u.
    """
    u = torch.exp((np.pi/2.0)*torch.sinh(t))
    du_dt = (np.pi/2.0)*torch.cosh(t)*u
    return u, du_dt
    

def interpolate_velocity(R_grid: torch.Tensor,
                         R_map : torch.Tensor,
                         v_grid: torch.Tensor) -> torch.Tensor:
    """
    1-D linear interpolation on an arbitrary monotonic grid.
    Any value outside [R_grid[0], R_grid[-1]] is clamped to the edges.
    Works on CUDA tensors, keeps gradients, avoids out-of-bounds.
    """
    # 1. Clamp the query points to the grid range
    R_clamp = R_map.clamp(min=R_grid[0], max=R_grid[-1])

    # 2. Locate the interval: first index such that R_grid[idx_hi] ≥ R_clamp
    idx_hi = torch.searchsorted(R_grid, R_clamp, right=False)

    #   For values equal to R_grid[-1] we still get idx_hi == len(R_grid)
    idx_hi = idx_hi.clamp(max=R_grid.numel() - 1)

    # 3. Lower neighbour
    idx_lo = (idx_hi - 1).clamp(min=0)

    # 4. Gather the two bracketing points
    R_lo, R_hi = R_grid[idx_lo], R_grid[idx_hi]
    v_lo, v_hi = v_grid[idx_lo], v_grid[idx_hi]

    # 5. Linear weight (when R_lo == R_hi, weight → 0)
    w = torch.where(
        R_hi == R_lo,
        torch.zeros_like(R_lo),
        (R_clamp - R_lo) / (R_hi - R_lo)
    )

    return v_lo + w * (v_hi - v_lo)


class MGEVelocityIntr(Module):
    """
    MGE but uses the intrinsic q directly.
    """
    def __init__(self, N_components: int, device, dtype, quad_points=128, radius_res = 4096, variable_M_to_L = False, soft = 0.0, G=0.004301):
        """
        Soft: softening length in parsecs
        """
        super().__init__("MGEVelocityIntr")
        self.device = device
        self.dtype  = dtype
        
        self.N_components = N_components
        
        # Same parameter definitions
        self.surf   = Param("surf",   shape=(N_components,))
        self.sigma  = Param("sigma",  shape=(N_components,))
        self.qintr   = Param("qintr",   shape=(N_components,))
        if variable_M_to_L:
            self.M_to_L = Param("M_to_L", shape=(N_components,))
        else:
            self.M_to_L = Param("M_to_L", shape=())
        
        self.m_bh  = Param("m_bh",  shape=())
        self.quad_points = quad_points
        self.radius_res = radius_res

        self.soft = torch.tensor(soft, device=device, dtype=dtype)
        self.G = torch.tensor(G, device=device, dtype=dtype)
        self.inc = Param("inc",   shape=())

    def radial_velocity(self, R_flat,
                 surf, sigma, qintr, M_to_L,
                 inc, m_bh):
        """
        Compute the rotational velocity at radii R_flat, but use a
        double-exponential transform from [0,1] -> (0,∞).
        """
        # --- Type-Safe Constants Definition ---
        # Define EVERY float literal as a tensor to prevent silent promotion.
        _p5 = torch.tensor(0.5, device=self.device, dtype=self.dtype)
        _1 = torch.tensor(1.0, device=self.device, dtype=self.dtype)
        _2 = torch.tensor(2.0, device=self.device, dtype=self.dtype)
        _10 = torch.tensor(10.0, device=self.device, dtype=self.dtype)
        _pi = torch.tensor(np.pi, device=self.device, dtype=self.dtype)
        _1e_7 = torch.tensor(1e-7, device=self.device, dtype=self.dtype)
        _1e3 = torch.tensor(1e3, device=self.device, dtype=self.dtype)
        _neg_1p5 = torch.tensor(-1.5, device=self.device, dtype=self.dtype)
        # --- End Constants Definition ---

        sqrt_2pi = torch.sqrt(_2 * _pi)
        qobs = torch.sqrt(qintr**2 * (torch.sin(inc))**2 + (torch.cos(inc))**2)
        mass_density = surf * M_to_L * qobs / (qintr * sigma * sqrt_2pi)

        N_points = R_flat.shape[0]

        # Scale by median sigma
        scale = sigma.quantile(q=0.5)
        sigma_sc = sigma / scale
        R_sc = R_flat / scale
        soft_sc = self.soft / scale

        mds = sigma_sc.quantile(q=0.5)
        mxs = torch.max(sigma_sc)

        xlim = (torch.arcsinh(torch.log(_1e_7 * mds) * _2 / _pi),
                torch.arcsinh(torch.log(_1e3 * mxs) * _2 / _pi))

        # --- Gauss–Legendre on [0,1] ---
        lo, hi = xlim
        t_1d, w_1d = leggauss_interval(self.quad_points, lo, hi, device=self.device, dtype=self.dtype)

        # --- Double-exponential transform t->u in (0,∞) ---
        u_1d, du_1d = transform_DE(t_1d)

        R_i = R_sc.view(-1, 1, 1)                     # (N,1,1)
        u_j = u_1d.view(1, -1, 1)                    # (1,Q,1)
        w_j = w_1d.view(1, -1, 1)                    # (1,Q,1)
        du_j = du_1d.view(1, -1, 1)                    # (1,Q,1)

        sigma_mat = sigma_sc.view(1, 1, -1)         # (1,1,C)
        qintr_mat = qintr.view(1, 1, -1)           # (1,1,C)
        mass_den_mat = mass_density.view(1, 1, -1)     # (1,1,C)

        # ---- kernel -----------------------------------------------------------------
        one_plus = _1 + u_j
        exp_val = torch.exp(-_p5 * R_i.pow(_2) /
                             (sigma_mat.pow(_2) * one_plus))

        denom = one_plus.pow(_2) * torch.sqrt(qintr_mat.pow(_2) + u_j)

        term = (qintr_mat * mass_den_mat * exp_val) / denom
        weighted = term * du_j * w_j

        # ---- quadrature & component sums -------------------------------------------
        integral_val = weighted.sum(dim=1).sum(dim=1)

        # ---- finish exactly as before ----------------------------------------------
        vc2_mge_factor = _2 * _pi * self.G * (scale**_2)
        vc2_mge = vc2_mge_factor * integral_val

        vc2_bh = self.G * _10**m_bh / scale * (R_sc**_2 + soft_sc**_2).pow(_neg_1p5)

        v_rot_flat = R_sc * torch.sqrt(vc2_mge + vc2_bh)

        return v_rot_flat
        
    @forward
    def velocity(
        self,
        R_map,                           # 2-D tensor [H,W]  (pc)
        surf=None, sigma=None, qintr=None, M_to_L=None, inc = None, m_bh=None
    ):
        """
        Returns v_rot(R) for every pixel in the sky plane.
        """
        Rmin = torch.as_tensor(self.soft, dtype=self.dtype, device=self.device)
        # Detach Rmax from the AD graph: the grid is a numerical lookup table whose
        # *spacing* does not need to carry gradients.  Differentiating through
        # torch.logspace w.r.t. its start/end args is not supported in forward AD
        # (jacfwd), and the effect of slightly shifted grid bounds on the
        # interpolated velocity is negligible for any dense grid.
        Rmax = R_map.max().detach()

        # 1-D lookup table (same as before)
        R_grid = torch.logspace(
            torch.log10(Rmin),
            torch.log10(Rmax),
            self.radius_res,
            device=self.device,
            dtype=self.dtype,
        )
        v_grid = self.radial_velocity(
            R_grid, surf, sigma, qintr, M_to_L, inc, m_bh
        )

        # interpolate onto the pixel-by-pixel radii
        v_abs = interpolate_velocity(R_grid, R_map, v_grid)      # (H,W)

        return v_abs


                
# -----------------------------
# Helper: precomputed ridge projector for MGE coefficients
# -----------------------------

class PrecomputedMGEProjector(torch.nn.Module):
    """
    Fixed basis A_ij = exp(-R_i^2 / (2 sigma_j^2))
    Precompute ridge projector:
        P = (A^T A + lam I)^-1 A^T
    so surf = P @ y
    """
    def __init__(
        self,
        *,
        r_grid_pc: torch.Tensor,
        sigma_grid_pc: torch.Tensor,
        lam_base: float = 1e-6,
        max_jitter_tries: int = 12,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.register_buffer("r_grid", r_grid_pc.to(device=device, dtype=dtype))
        self.register_buffer("sigma_grid", sigma_grid_pc.to(device=device, dtype=dtype))

        r = self.r_grid
        sig = self.sigma_grid
        A = torch.exp(-(r[:, None] ** 2) / (2.0 * sig[None, :] ** 2))  # (R,K)
        self.register_buffer("A", A)

        A64 = A.to(torch.float64)
        _, K = A64.shape
        AtA64 = A64.T @ A64
        I64 = torch.eye(K, device=device, dtype=torch.float64)

        lam = float(lam_base)
        L = None
        for _ in range(max_jitter_tries):
            try:
                M64 = AtA64 + lam * I64
                L = torch.linalg.cholesky(M64)
                break
            except Exception:
                lam *= 10.0

        if L is None:
            raise RuntimeError(
                "Failed to Cholesky-factor (A^T A + λI). "
                "Try increasing lam_base or reducing K."
            )

        P64 = torch.cholesky_solve(A64.T, L)  # (K,R)
        self.register_buffer("P", P64.to(dtype=dtype))
        self.lam_used = lam

    def surf_from_profile(self, y: torch.Tensor) -> torch.Tensor:
        if y.ndim == 1:
            return self.P @ y
        elif y.ndim == 2:
            return y @ self.P.T
        raise ValueError(f"y must be 1D or 2D, got {tuple(y.shape)}")

    def profile_from_surf(self, surf: torch.Tensor) -> torch.Tensor:
        if surf.ndim == 1:
            return self.A @ surf
        elif surf.ndim == 2:
            return surf @ self.A.T
        raise ValueError(f"surf must be 1D or 2D, got {tuple(surf.shape)}")



# -----------------------------
# Shared base
# -----------------------------
class _PreLSMGEBase(Module):
    def __init__(
        self,
        name: str,
        N_MGE_components: int,
        *,
        distance_Mpc: float,
        soft: float,
        device,
        dtype,
        n_radii_data: int = 100,
        r_min_pc: float = 1.0,
        r_max_pc: float = 1e4,
        lam_base: float = 1e-6,
        max_jitter_tries: int = 12,
        quad_points: int = 128,
        radius_res: int = 4096,
        G: float = 0.004301,
    ):
        super().__init__(name)
        self.device = device
        self.dtype = dtype
        self.N_MGE_components = N_MGE_components

        pi_t = torch.tensor(math.pi, device=device, dtype=dtype)
        c_t = torch.tensor(0.648, device=device, dtype=dtype)  # pi/0.648 ≈ 4.848
        self.distance_Mpc = torch.tensor(distance_Mpc, device=device, dtype=dtype)
        self.pc_per_arcsec = self.distance_Mpc * (pi_t / c_t)

        self._eps = torch.tensor(1e-20, device=device, dtype=dtype)
        self.G = torch.tensor(G, device=device, dtype=dtype)

        r_grid = torch.tensor(
            np.geomspace(r_min_pc, r_max_pc, n_radii_data),
            device=device,
            dtype=dtype,
        )

        low_G = np.log10(np.min(r_grid.detach().cpu().numpy()) / np.sqrt(3.0))
        high_G = np.log10(np.max(r_grid.detach().cpu().numpy()) / np.sqrt(3.0))
        dx = (high_G - low_G) / N_MGE_components
        sigma_grid_np = 10 ** (low_G + (0.5 + np.arange(N_MGE_components)) * dx)
        sigma_grid = torch.tensor(sigma_grid_np, device=device, dtype=dtype)

        self.projector = PrecomputedMGEProjector(
            r_grid_pc=r_grid,
            sigma_grid_pc=sigma_grid,
            lam_base=lam_base,
            max_jitter_tries=max_jitter_tries,
            dtype=dtype,
            device=device,
        )

        self.MGE = MGEVelocityIntr(
            N_components=N_MGE_components,
            device=device,
            dtype=dtype,
            quad_points=quad_points,
            radius_res=radius_res,
            soft=soft,
            G=G,
        )

        # galaxy params
        self.inc = Param("inc", shape=())
        self.qintr = Param("qintr", shape=())
        self.m_bh = Param("m_bh", shape=())

        self.MGE.inc = self.inc
        self.MGE.m_bh = self.m_bh

        # sigma basis is fixed
        self.MGE.sigma.to_static(self.projector.sigma_grid.clone())

        #Mass to Light is now controlled by intensity normalization of profile
        self.MGE.M_to_L.to_static(torch.ones((1,), device=self.device, dtype=self.dtype))

        # qintr becomes a functional pointer: scalar -> per-Gaussian vector
        self.MGE.qintr = self._qintr_pointer
        self.MGE.qintr.link([self.qintr])

    def _ten(self):
        return torch.tensor(10.0, device=self.device, dtype=self.dtype)

    def _qintr_pointer(self, p):
        return p.qintr.value * torch.ones(
            (self.N_MGE_components,),
            device=self.device,
            dtype=self.dtype,
        )


# -----------------------------
# Profile helpers
# -----------------------------
def sersic_profile_torch_1d(R, I_e, R_e, n, eps=1e-30):
    R   = torch.clamp(R,   min=eps)
    R_e = torch.clamp(R_e, min=eps)
    n   = torch.clamp(n,   min=eps)

    b_n = 2.0 * n - (1.0 / 3.0) + 4.0 / (405.0 * n) + 46.0 / (25515.0 * n * n)
    return I_e * torch.exp(-b_n * (torch.pow(R / R_e, 1.0 / n) - 1.0))


def nuker_profile_torch_1d(R, I_b, r_b, alpha, beta, gamma, eps=1e-30):
    R = torch.clamp(R, min=eps)
    r_b = torch.clamp(r_b, min=eps)
    alpha = torch.clamp(alpha, min=eps)

    x = torch.clamp(R / r_b, min=eps)
    two = torch.tensor(2.0, device=R.device, dtype=R.dtype)
    pref = I_b * torch.pow(two, (beta - gamma) / alpha)
    core = torch.pow(x, -gamma)
    outer = torch.pow(1.0 + torch.pow(x, alpha), (gamma - beta) / alpha)
    return pref * core * outer


def core_sersic_torch_1d(R, I_b, R_b, R_e, alpha, gamma, n, eps=1e-30):
    R   = torch.clamp(R,   min=eps)
    R_b = torch.clamp(R_b, min=eps)
    R_e = torch.clamp(R_e, min=eps)
    alpha = torch.clamp(alpha, min=eps)
    n     = torch.clamp(n, min=eps)

    b_n = 2.0 * n - (1.0 / 3.0) + 4.0 / (405.0 * n) + 46.0 / (25515.0 * n * n)

    term1 = torch.pow(1.0 + torch.pow(R_b / R, alpha), gamma / alpha)
    inside = (torch.pow(R, alpha) + torch.pow(R_b, alpha)) / torch.pow(R_e, alpha)
    term2 = torch.exp(-b_n * (torch.pow(inside, 1.0 / (alpha * n)) - 1.0))
    return I_b * term1 * term2


# -----------------------------
# Sérsic
# -----------------------------
class SersicMGE(_PreLSMGEBase):
    """
    Sérsic params -> profile -> functional-pointer surf -> MGE velocity
    """
    def __init__(
        self,
        N_MGE_components: int,
        *,
        distance_Mpc: float,
        soft: float,
        device,
        dtype,
        n_radii_data: int = 100,
        r_min_pc: float = 1.0,
        r_max_pc: float = 1e4,
        lam_base: float = 1e-6,
        max_jitter_tries: int = 12,
        quad_points: int = 128,
        radius_res: int = 4096,
        G: float = 0.004301,
    ):
        super().__init__(
            "SersicMGE",
            N_MGE_components,
            distance_Mpc=distance_Mpc,
            soft=soft,
            device=device,
            dtype=dtype,
            n_radii_data=n_radii_data,
            r_min_pc=r_min_pc,
            r_max_pc=r_max_pc,
            lam_base=lam_base,
            max_jitter_tries=max_jitter_tries,
            quad_points=quad_points,
            radius_res=radius_res,
            G=G,
        )

        self.n = Param("n", shape=(1,))
        self.r_e = Param("r_e", shape=())           # arcsec
        self.I_e = Param("intensity_r_e", shape=()) # log10(I_e)

        self.MGE.surf = self._surf_pointer
        self.MGE.surf.link([self.n, self.r_e, self.I_e])

    def _surf_pointer(self, p):
        r_e_pc = torch.clamp(p.r_e.value * self.pc_per_arcsec, min=self._eps)
        I_e_lin = torch.pow(self._ten(), p.intensity_r_e.value)

        y = sersic_profile_torch_1d(
            self.projector.r_grid,
            I_e_lin,
            r_e_pc.squeeze() if r_e_pc.ndim > 0 else r_e_pc,
            p.n.value.squeeze(),
            eps=float(self._eps.item()),
        )
        return self.projector.surf_from_profile(y)

    @forward
    def velocity(self, R_flat, inc=None, qintr=None, m_bh=None, n=None, r_e=None, I_e=None):
        return self.MGE.velocity(R_map=R_flat)


# -----------------------------
# Nuker
# -----------------------------
class NukerMGE(_PreLSMGEBase):
    """
    Nuker params -> profile -> functional-pointer surf -> MGE velocity
    """
    def __init__(
        self,
        N_MGE_components: int,
        *,
        distance_Mpc: float,
        soft: float,
        device,
        dtype,
        n_radii_data: int = 100,
        r_min_pc: float = 1.0,
        r_max_pc: float = 1e4,
        lam_base: float = 1e-6,
        max_jitter_tries: int = 12,
        quad_points: int = 128,
        radius_res: int = 4096,
        G: float = 0.004301,
    ):
        super().__init__(
            "NukerMGE",
            N_MGE_components,
            distance_Mpc=distance_Mpc,
            soft=soft,
            device=device,
            dtype=dtype,
            n_radii_data=n_radii_data,
            r_min_pc=r_min_pc,
            r_max_pc=r_max_pc,
            lam_base=lam_base,
            max_jitter_tries=max_jitter_tries,
            quad_points=quad_points,
            radius_res=radius_res,
            G=G,
        )

        self.alpha = Param("alpha", shape=(1,))
        self.gmb = Param("gamma_minus_beta", shape=(1,))
        self.gamma = Param("gamma", shape=(1,))
        self.r_b = Param("r_b", shape=(1,))              # arcsec
        self.I_b = Param("intensity_r_b", shape=())      # log10(I_b)

        self.MGE.surf = self._surf_pointer
        self.MGE.surf.link([self.alpha, self.gmb, self.gamma, self.r_b, self.I_b])

    def _surf_pointer(self, p):
        gamma = p.gamma.value.squeeze()
        gmb = p.gamma_minus_beta.value.squeeze()
        beta = gamma - gmb
    
        r_b_pc = torch.clamp(p.r_b.value * self.pc_per_arcsec, min=self._eps)
        I_b_lin = torch.pow(self._ten(), p.intensity_r_b.value)
    
        y = nuker_profile_torch_1d(
            self.projector.r_grid,
            I_b_lin,
            r_b_pc.squeeze(),
            p.alpha.value.squeeze(),
            beta,
            gamma,
            eps=float(self._eps.item()),
        )
        return self.projector.surf_from_profile(y)

    @forward
    def velocity(
        self,
        R_flat,
        inc=None,
        qintr=None,
        m_bh=None,
        alpha=None,
        gmb=None,
        gamma=None,
        r_b=None,
        I_b=None,
    ):
        return self.MGE.velocity(R_map=R_flat)


# -----------------------------
# Core-Sérsic
# -----------------------------
class CoreSersicMGE(_PreLSMGEBase):
    """
    Core-Sérsic params -> profile -> functional-pointer surf -> MGE velocity
    """
    def __init__(
        self,
        N_MGE_components: int,
        *,
        distance_Mpc: float,
        soft: float,
        device,
        dtype,
        n_radii_data: int = 100,
        r_min_pc: float = 1.0,
        r_max_pc: float = 1e4,
        lam_base: float = 1e-6,
        max_jitter_tries: int = 12,
        quad_points: int = 128,
        radius_res: int = 4096,
        G: float = 0.004301,
    ):
        super().__init__(
            "CoreSersicMGE",
            N_MGE_components,
            distance_Mpc=distance_Mpc,
            soft=soft,
            device=device,
            dtype=dtype,
            n_radii_data=n_radii_data,
            r_min_pc=r_min_pc,
            r_max_pc=r_max_pc,
            lam_base=lam_base,
            max_jitter_tries=max_jitter_tries,
            quad_points=quad_points,
            radius_res=radius_res,
            G=G,
        )

        self.I_b = Param("I_b", shape=())       # log10(I_b)
        self.R_b = Param("R_b", shape=(1,))     # arcsec
        self.R_e = Param("R_e", shape=(1,))     # arcsec
        self.alpha = Param("alpha", shape=(1,))
        self.gamma = Param("gamma", shape=(1,))
        self.n = Param("n", shape=(1,))

        self.MGE.surf = self._surf_pointer
        self.MGE.surf.link([self.I_b, self.R_b, self.R_e, self.alpha, self.gamma, self.n])

    def _surf_pointer(self, p):
        R_b_pc = torch.clamp(p.R_b.value * self.pc_per_arcsec, min=self._eps)
        R_e_pc = torch.clamp(p.R_e.value * self.pc_per_arcsec, min=self._eps)
        I_b_lin = torch.pow(self._ten(), p.I_b.value)

        y = core_sersic_torch_1d(
            self.projector.r_grid,
            I_b_lin,
            R_b_pc.squeeze(),
            R_e_pc.squeeze(),
            p.alpha.value.squeeze(),
            p.gamma.value.squeeze(),
            p.n.value.squeeze(),
            eps=float(self._eps.item()),
        )
        return self.projector.surf_from_profile(y)

    @forward
    def velocity(
        self,
        R_flat,
        inc=None,
        qintr=None,
        m_bh=None,
        I_b=None,
        R_b=None,
        R_e=None,
        alpha=None,
        gamma=None,
        n=None,
    ):
        return self.MGE.velocity(R_map=R_flat)


# ------------------------------------------
# Gas mass profile (exponential disk model)
# ------------------------------------------

class GasSelfGrav(Module):
    def __init__(self,intensity_model,device,dtype):
        super().__init__("GasSelfGrav")
        self.scale = intensity_model.scale
        self.m_gas = Param("m_gas",  shape=())

    @forward
    def velocity(self, R_flat, m_gas = None, scale = None, G=0.004301):
        x=R_flat/scale
        ## based on eqn 8.74 in "Dynamics and Astrophysics of Galaxies" by Bovy 
        prefac=((G*(10**m_gas))/(2*scale))*(x**2)
        endfac=modified_bessel_i0(x/2)*modified_bessel_k0(x/2) - modified_bessel_i1(x/2)*modified_bessel_k1(x/2)
        vcsqr=prefac*endfac
        return torch.sqrt(vcsqr)


# --------------------------
# Composable mass models
# --------------------------

class QuadratureVelocitySum(Module):
    def __init__(self, models: Tuple[Module, ...], name: Optional[str] = None):
        super().__init__(name or "MultiComponentMass")

        self.n_models = len(models)
        self._model_names = []

        for i, model in enumerate(models):
            attr = f"model_{i}"
            setattr(self, attr, model)
            self._model_names.append(attr)

        # Look for inclination-like params on child models.
        # Support either `.inc` or `.inclination`.
        inclination_attrs = []
        for name in self._model_names:
            model = getattr(self, name)

            if hasattr(model, "inc") and isinstance(getattr(model, "inc"), Param):
                inclination_attrs.append((name, "inc"))
            elif hasattr(model, "inclination") and isinstance(getattr(model, "inclination"), Param):
                inclination_attrs.append((name, "inclination"))

        # If any child has an inclination param, create one shared top-level param
        # and point all such children to it.
        if len(inclination_attrs) > 0:
            self.inc = Param("inclination", shape=())

            for model_name, attr_name in inclination_attrs:
                model = getattr(self, model_name)
                setattr(model, attr_name, self.inc)

    @forward
    def velocity(self, R_flat, inclination=None):
        v2_total = torch.zeros_like(R_flat)

        for name in self._model_names:
            model = getattr(self, name)
            v_cur = model.velocity(R_flat)
            v2_total = v2_total + v_cur**2

        return torch.sqrt(v2_total)


