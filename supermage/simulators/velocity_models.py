import torch
import torch.nn as nn
from torch import Tensor
from typing import Literal
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

class Sersic_MGE(Module):
    def __init__(self, N_MGE_components: int, n_grid, surf_grid, distance, r_min, r_max, soft, device, dtype, quad_points=128):
        super().__init__("SersicMGE")
        self.N_components = N_MGE_components
        self.soft = soft
        self.MGE = MGEVelocityIntr(self.N_components, soft = soft, quad_points = quad_points, dtype = dtype, device = device)
        self.MGE.surf = torch.ones((self.N_components), device = device).to(dtype = dtype)
        self.MGE.sigma = torch.ones((self.N_components), device = device).to(dtype = dtype)
        self.MGE.qintr = torch.ones((self.N_components), device = device).to(dtype = dtype)
        self.MGE.M_to_L = torch.tensor([1.0], dtype = dtype, device = device)
        self.n_grid = n_grid
        self.surf_grid = surf_grid
        self.surf=torch.zeros(self.N_components, device=device, dtype=dtype)

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

        self.n = Param("n", shape=(1, ))
        self.r_e = Param("r_e", shape = ())
        self.I_e = Param("intensity_r_e", shape = ())
        self.dtype = dtype

    @forward
    def velocity(self, R_flat,
                 inc=None, qintr=None, m_bh=None,
                 n = None, r_e = None, I_e = None,
                 G=0.004301):
        device = R_flat.device
        dtype  = R_flat.dtype

        for i in range(self.N_components):
            self.surf[i]=interp1d(self.n_grid,self.surf_grid[:,i],n,extend="extrapolate")
        
        MGE_surf = self.surf*10**I_e
        MGE_sigma = self.sigma*r_e
        v_rot = self.MGE.velocity(R_map = R_flat, surf = MGE_surf, sigma = MGE_sigma, qintr = qintr*self.qintr_shaper)
        return v_rot
                     
class Sersic_Gas(Module):
    def __init__(self, gas_grav_model, N_MGE_components: int, n_grid, surf_grid, distance, r_min, r_max, soft, device, dtype, quad_points=128):
        super().__init__("Sersic_Gas")
        self.gas_grav = gas_grav_model
        self.scale = gas_grav_model.scale
        self.m_gas = gas_grav_model.m_gas
        
        self.N_components = N_MGE_components
        self.soft = soft
        self.MGE = MGEVelocityIntr(self.N_components, soft = soft, quad_points = quad_points, dtype = dtype, device = device)
        self.MGE.surf = torch.ones((self.N_components), device = device).to(dtype = dtype)
        self.MGE.sigma = torch.ones((self.N_components), device = device).to(dtype = dtype)
        self.MGE.qintr = torch.ones((self.N_components), device = device).to(dtype = dtype)
        self.MGE.M_to_L = torch.tensor([1.0], dtype = dtype, device = device)
        self.n_grid = n_grid
        self.surf_grid = surf_grid
        self.surf=torch.zeros(self.N_components, device=device, dtype=dtype)

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

        self.n = Param("n", shape=(1, ))
        self.r_e = Param("r_e", shape = ())
        self.I_e = Param("intensity_r_e", shape = ())
        self.dtype = dtype

    @forward
    def velocity(self, R_flat,
                 m_gas = None, scale = None,
                 inc=None, qintr=None, m_bh=None,
                 n = None, r_e = None, I_e = None,
                 G=0.004301):
        device = R_flat.device
        dtype  = R_flat.dtype

        for i in range(self.N_components):
            self.surf[i]=interp1d(self.n_grid,self.surf_grid[:,i],n,extend="extrapolate")
        
        MGE_surf = self.surf*10**I_e
        MGE_sigma = self.sigma*r_e
        v_stars_BH = self.MGE.velocity(R_map = R_flat, surf = MGE_surf, sigma = MGE_sigma, qintr = qintr*self.qintr_shaper)
        
        v_gas = self.gas_grav.velocity(R_flat, m_gas = m_gas, scale = scale)
        
        v_rot = torch.sqrt(v_stars_BH**2 + v_gas**2) 
        return v_rot

class Nuker_Gas(Module):
    def __init__(self, gas_grav_model, N_MGE_components: int, Nuker_NN, NN_dtype, distance, r_min, r_max, soft, device, dtype, quad_points=128):
        super().__init__("Nuker_Gas")
        self.gas_grav = gas_grav_model
        self.scale = gas_grav_model.scale
        self.m_gas = gas_grav_model.m_gas
        
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
                 m_gas = None, scale = None, 
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
        v_stars_BH = self.MGE.velocity(R_map = R_flat, surf = surf, sigma = MGE_sigma, qintr = qintr*self.qintr_shaper)

        v_gas = self.gas_grav.velocity(R_flat, m_gas = m_gas, scale = scale)

        v_rot = torch.sqrt(v_stars_BH**2 + v_gas**2) 
        return v_rot


class NukerMGEFull(Module):
    def __init__(self, N_MGE_components: int, Nuker_NN, NN_dtype, distance, soft, device, dtype, scaler_path, quad_points=128):
        """
        A velocity model that uses a trained neural network to map Nuker
        parameters to an MGE representation. This class takes a trained NukerMGEProfileModel as input.

        Args:
            trained_nn_model (MGEProfileModel): The fully trained neural network model
                that predicts MGE profiles.
            scaler_path (str): The file path to the saved 'StandardScaler'
                (.joblib file) used during training.
            device: The PyTorch device (e.g., 'cuda' or 'cpu').
            dtype: The PyTorch data type (e.g., torch.float32).
            mge_velocity_calculator (MGEVelocityIntr): A pre-initialized module
                that calculates velocities from MGE components.
        """
        super().__init__("NukerMGEFull")

        self.N_components = N_MGE_components
        self.soft = soft
        # pc / arcsec conversion (distance in Mpc)
        pi_t   = torch.tensor(math.pi, device=device, dtype=dtype)
        c_t    = torch.tensor(0.648, device=device, dtype=dtype)  # so that pi/0.648 ≈ 4.848
        self.distance_Mpc   = torch.tensor(distance, device=device, dtype=dtype)
        self.pc_per_arcsec  = self.distance_Mpc * (pi_t / c_t)    # ≈ 4.848 * D_Mpc
        # Small epsilon for safe logs
        self._eps = torch.tensor(1e-20, device=device, dtype=dtype)
        
        self.MGE = MGEVelocityIntr(self.N_components, soft = soft, quad_points = quad_points, dtype = dtype, device = device)
        self.MGE.surf = torch.ones((self.N_components), device = device).to(dtype = dtype)
        self.MGE.sigma = torch.ones((self.N_components), device = device).to(dtype = dtype)
        self.MGE.qintr = torch.ones((self.N_components), device = device).to(dtype = dtype)
        self.MGE.M_to_L = torch.tensor([1.0], dtype = dtype, device = device)
        self.NN = Nuker_NN
        
        # --- 2. Load the Scaler and create autodifferentiable buffers ---
        try:
            scaler = joblib.load(scaler_path)
            # Convert scaler's mean and scale to PyTorch tensors
            self.scaler_mean = torch.tensor(scaler.mean_, device=device, dtype=dtype)
            self.scaler_scale = torch.tensor(scaler.scale_, device=device, dtype=dtype)
            
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Scaler file not found at '{scaler_path}'. "
                "This file is required to transform inputs for the neural network."
            )
        
        # --- 4. Define the fittable parameters using Caskade's syntax ---
        # These are the physical parameters of the galaxy model
        self.inc   = Param("inc",   shape=())
        self.qintr = Param("qintr", shape=())
        self.m_bh  = Param("m_bh",  shape=())
        
        # These are the four Nuker parameters that will be fed into the NN
        self.alpha   = Param("alpha",   shape=(1, ))
        self.gmb = Param("gamma_minus_beta", shape=(1, ))
        self.gamma   = Param("gamma",   shape=(1, ))
        self.r_b = Param("r_b", shape=(1, ))
        self.I_b = Param("intensity_r_b", shape = ())

        # --- 5. Set up other necessary attributes ---
        # Get the number of components from the trained NN model
        self.N_components = self.NN.n_gauss_model
        self.qintr_shaper = torch.ones((self.N_components), device=device, dtype=dtype)
        
        # Link parameters to the MGE velocity calculator
        self.MGE.inc = self.inc
        self.MGE.m_bh = self.m_bh

    @forward
    def velocity(self, R_flat,
                 inc=None, qintr=None, m_bh=None,
                 alpha = None, gmb = None, gamma = None, r_b = None, I_b = None,
                 G=0.004301):
        """
        Calculates the velocity curve.
        """
        # --- Step 1: Assemble the Nuker parameters for the NN ---
        # The NN expects a tensor of shape (batch_size, 4).
        # We unsqueeze to create a batch dimension of 1.
        beta = gamma - gmb
        r_b_pc = torch.clamp(r_b * self.pc_per_arcsec, min=self._eps)
        log10_r_b_pc = torch.log10(r_b_pc)
        NN_input = torch.cat([
            alpha,
            beta,
            gamma,
            log10_r_b_pc
        ]).unsqueeze(0) # Transpose to get shape (1, 4)

        # --- Step 2: Apply the scaler transformation (autodifferentiable) ---
        # This operation is now part of the PyTorch computation graph.
        NN_input_scaled = (NN_input - self.scaler_mean) / self.scaler_scale

        # --- Step 3: Get MGE parameters from the trained neural network ---
        # The NN_model's internal get_mge_params handles the symexp transform.
        # It returns surf and sigma with shape (1, n_gauss). We squeeze them.
        surf, sigma = self.NN.get_mge_params(NN_input_scaled)
        surf = surf.squeeze(0)
        sigma = sigma.squeeze(0)
        
        # --- Step 4: Calculate the velocity using the predicted MGE ---
        v_rot = self.MGE.velocity(
            R_map=R_flat, 
            surf=surf*10**I_b, 
            sigma=sigma, 
            qintr=qintr * self.qintr_shaper
        )
        
        return v_rot

class NukerMGEProfileModel(nn.Module):
    """
    A neural network-based model that predicts the MGE mass profile from Nuker parameters. Use this class to train the neural network and load it into NukerMGEFull!
    """
    def __init__(self, n_gauss_model, n_radii_data=100, r_min=1, r_max=10000, 
                 linthresh=1e-5, base=10.0):
        """
        Initializes the model and the fixed sigma grid.
        Args:
            n_gauss_model (int): The number of Gaussian components.
            n_radii_data (int): The number of radial bins for the profile.
            r_min (float): The minimum radius (in pc) for the sigma grid calculation.
            r_max (float): The maximum radius (in pc) for the sigma grid calculation.
            linthresh (float): The threshold for the linear region of symexp/symlog.
            base (float): The base for the exponential/logarithmic part of the transform.
        """
        super(NukerMGEProfileModel, self).__init__()
        self.n_gauss_model = n_gauss_model
        self.nuker_to_mge_net = NukerToMGE_NN(n_gauss=self.n_gauss_model)
        
        # Store symexp parameters
        self.linthresh = linthresh # <--- NEW
        self.base = base         # <--- NEW

        r_space_basis = np.geomspace(r_min, r_max, n_radii_data)
        low_Gauss = np.log10(np.min(r_space_basis) / np.sqrt(3.0))
        high_Gauss = np.log10(np.max(r_space_basis) / np.sqrt(3.0))
        dx = (high_Gauss - low_Gauss) / self.n_gauss_model
        sigma_grid_np = 10**(low_Gauss + (0.5 + np.arange(self.n_gauss_model)) * dx)
        
        self.register_buffer('sigma_grid', torch.tensor(sigma_grid_np, dtype=torch.float32))

    def symexp(self, y): # <--- NEW: Symexp transformation method
        """ Symmetrical exponential function. Inverse of symlog. """
        linthresh_t = torch.tensor(self.linthresh, device=y.device, dtype=y.dtype)
        base_t = torch.tensor(self.base, device=y.device, dtype=y.dtype)
        one_t = torch.tensor(1.0, device=y.device, dtype=y.dtype)
    
        return torch.sign(y) * linthresh_t * (base_t**torch.abs(y) - one_t)

    def symlog(self, x): # <--- NEW: Symlog for analysis (not used in training)
        """ Symmetrical logarithmic function. Inverse of symexp. """
        linthresh_t = torch.tensor(self.linthresh, device=x.device, dtype=x.dtype)
        base_t = torch.tensor(self.base, device=x.device, dtype=x.dtype)
        one_t = torch.tensor(1.0, device=x.device, dtype=x.dtype)

        return torch.sign(x) * torch.log10(torch.abs(x) / linthresh_t + one_t)

    def get_mge_params(self, nuker_params):
        """
        Exposes the MGE parameters after applying the symexp transform.
        """
        # The core NN now predicts the *compressed* surf values
        compressed_surfs = self.nuker_to_mge_net(nuker_params) # <--- CHANGED
        
        # Apply the symexp transform to get the true, high-dynamic-range surf values
        predicted_surfs = self.symexp(compressed_surfs) # <--- CHANGED
        
        return predicted_surfs, self.sigma_grid

    def forward(self, nuker_params, r_space_eval):
        """
        Predicts the MGE mass profile. This function does not need to change.
        """
        # This method automatically uses the new get_mge_params method
        predicted_surfs, sigmas = self.get_mge_params(nuker_params)
        
        r_squared = r_space_eval.view(1, 1, -1)**2
        sigmas_squared = (2 * sigmas.view(1, -1, 1)**2)
        
        gaussians = torch.exp(-r_squared / sigmas_squared)
        mass_profile = torch.sum(predicted_surfs.unsqueeze(2) * gaussians, dim=1)
        return mass_profile

class NukerToMGE_NN(nn.Module):
    """
    A neural network that maps Nuker parameters to MGE surface brightness normalizations.
    """
    def __init__(self, n_gauss=64):
        """
        Initializes the network architecture.
        Args:
            n_gauss (int): The number of Gaussian components in the MGE model.
        """
        super(NukerToMGE_NN, self).__init__()
        
        # This sequential model forms the core of the network, learning the
        # complex relationship between Nuker parameters and MGE normalizations.
        self.layers = nn.Sequential(
            # Input layer expects 4 features: alpha, beta, gamma, log_r_b
            nn.Linear(4, 128),
            nn.SELU(),
            nn.Linear(128, 256),
            nn.SELU(),
            nn.Linear(256, 512),
            nn.SELU(),
            nn.Linear(512, 256),
            nn.SELU(),
            nn.Linear(256, 128),
            nn.SELU(),
            # Output layer produces n_gauss values, corresponding to the MGE surfs.
            nn.Linear(128, n_gauss)
        )

    def forward(self, x):
        """
        Performs the forward pass.
        Args:
            x (torch.Tensor): A tensor of shape (batch_size, 4) containing the
                              Nuker parameters.
        Returns:
            torch.Tensor: A tensor of shape (batch_size, n_gauss) representing
                          the predicted MGE surf values.
        """
        return self.layers(x)



########################################################## DEPRECATED MODELS BELOW THIS LINE #####################################################################

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
