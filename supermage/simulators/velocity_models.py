import torch
from torch import pi, sqrt
from caskade import Module, forward, Param
import numpy as np
from numpy.polynomial.legendre import leggauss
from torch.nn.functional import conv2d, avg_pool2d
from functools import lru_cache
import math


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
        Rmax = R_map.max()

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

class Nuker_MGE(Module):
    def __init__(self, N_MGE_components: int, Nuker_NN, NN_dtype, distance, r_min, r_max, soft, device, dtype, quad_points=128):
        super().__init__("NukerMGE")
        self.N_components = N_MGE_components
        self.soft = soft
        self.MGE = MGEVelocityIntr(self.N_components, soft = soft, quad_points = quad_points, dtype = dtype, device = device)
        self.MGE.surf = torch.ones((self.N_components), device = device).to(dtype = dtype)
        self.MGE.sigma = torch.ones((self.N_components), device = device).to(dtype = dtype)
        self.MGE.qintr = torch.ones((self.N_components), device = device).to(dtype = dtype)
        self.MGE.M_to_L = torch.tensor([1.0], dtype = dtype, device = device)
        self.NN = Nuker_NN

        inner_slope=torch.tensor([3.0], device = device, dtype = dtype)
        outer_slope=torch.tensor([3.0], device = device, dtype = dtype)
        low_Gauss=torch.log10(r_min/torch.sqrt(inner_slope))
        high_Gauss=torch.log10(r_max/torch.sqrt(outer_slope))
        dx=(high_Gauss-low_Gauss)/self.N_components
        
        # --- SOLUTION ---
        # Ensure all scalars are tensors of the correct dtype before the calculation
        distance_t = torch.tensor(distance, device=device, dtype=dtype)
        pi_t = torch.tensor(np.pi, device=device, dtype=dtype)
        
        self.sigma = (distance_t * (pi_t / 0.648)) * 10**(low_Gauss + (0.5 + torch.arange(self.N_components, device=device, dtype=dtype)) * dx)
        
        self.inc   = Param("inc",   shape=())
        self.qintr = Param("qintr", shape=())
        self.qintr_shaper = torch.ones((self.N_components), device = device).to(dtype = dtype)
        self.m_bh  = Param("m_bh",  shape=())
        self.MGE.inc = self.inc
        self.MGE.m_bh = self.m_bh

        self.alpha = Param("alpha", shape=(1, ))
        self.gmb = Param("gamma_minus_beta", shape=(1, ))
        self.gamma = Param("gamma", shape=(1, ))
        self.r_b = Param("break_r", shape = ())
        self.I_b = Param("intensity_r_b", shape = ())
        self.dtype = dtype
        self.NN_dtype = NN_dtype
    
    def symexp(self, y, linthresh=1e-12, base=10.0):
        # --- SOLUTION ---
        # Create tensor constants that match the input tensor's properties
        linthresh_t = torch.tensor(linthresh, device=y.device, dtype=y.dtype)
        base_t = torch.tensor(base, device=y.device, dtype=y.dtype)
        one_t = torch.tensor(1.0, device=y.device, dtype=y.dtype)
    
        return torch.sign(y) * linthresh_t * (base_t**torch.abs(y) - one_t)

    @forward
    def velocity(self, R_flat,
                 inc=None, qintr=None, m_bh=None,
                 alpha = None, gmb = None, gamma = None, r_b = None, I_b = None,
                 G=0.004301):
        device = R_flat.device
        dtype  = R_flat.dtype
        beta = gamma - gmb

        NN_input = torch.cat([alpha, beta, gamma])#.to(self.NN_dtype)
        NN_output_transformed = self.NN.forward(NN_input)#.to(self.dtype)
        NN_output = self.symexp(NN_output_transformed)
        
        surf = NN_output*10**I_b
        MGE_sigma = self.sigma*r_b
        v_rot = self.MGE.velocity(R_map = R_flat, surf = surf, sigma = MGE_sigma, qintr = qintr*self.qintr_shaper)
        return v_rot