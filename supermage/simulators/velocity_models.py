"""Circular-velocity (mass) models for gas-kinematic modeling.

The central engine is :class:`MGEVelocityIntr`, which computes the circular
velocity of an axisymmetric Multi-Gaussian Expansion (MGE) mass distribution
plus a central black hole (Cappellari 2002, eq. 10-11 style quadrature,
evaluated with a double-exponential transform so the improper integral is
accurate in float32/float64).

On top of it, :class:`SersicMGE`, :class:`NukerMGE` and
:class:`CoreSersicMGE` parameterize the stellar surface-brightness profile
analytically (Sersic / Nuker / Core-Sersic) and convert it on the fly into
MGE amplitudes with a precomputed ridge-regression projector
(:class:`PrecomputedMGEProjector`), so the sampler sees the profile's natural
parameters instead of raw per-Gaussian amplitudes.

:class:`GasSelfGrav` adds the self-gravity of an exponential gas disk and
:class:`QuadratureVelocitySum` composes several mass components by adding
their circular velocities in quadrature.

Unit conventions
----------------
* radii passed to ``velocity`` / ``radial_velocity`` are in **parsec**;
* MGE ``sigma`` dispersions are in parsec, ``surf`` in surface density
  (mass or light per pc^2 -- see ``M_to_L``);
* profile break/effective radii of the Sersic/Nuker/Core-Sersic wrappers are
  in **arcsec** (converted internally with the model distance);
* intensities/amplitudes of those wrappers are ``log10`` values;
* ``m_bh`` is ``log10(M_BH / M_sun)``;
* ``inc`` (inclination) is in **radians**;
* with the default ``G = 0.004301`` (pc (km/s)^2 / M_sun), velocities come
  out in **km/s**.

All models are `caskade <https://github.com/Ciela-Institute/caskade>`_
Modules: constructor arguments are static configuration, while
:class:`caskade.Param` attributes are (potentially trainable) simulation
parameters supplied at call time.
"""
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
    """Cached Gauss-Legendre nodes/weights on ``[-1, 1]`` as torch tensors.

    Parameters
    ----------
    n : int
        Number of quadrature points.
    dtype : torch.dtype
        Tensor dtype of the returned nodes and weights.
    device : torch.device or str
        Device of the returned tensors.

    Returns
    -------
    (Tensor, Tensor)
        Nodes ``x`` and weights ``w``, each of shape ``(n,)``.  Cached with
        ``lru_cache`` so repeated likelihood evaluations reuse the same
        constants.
    """
    x_np, w_np = np.polynomial.legendre.leggauss(n)
    return (torch.as_tensor(x_np, dtype=dtype, device = device),
            torch.as_tensor(w_np, dtype=dtype, device = device))

# 2.  Pure-Torch mapping keeps autograd alive and avoids graph breaks.
def leggauss_interval(n, t_low, t_high, device=None, dtype=None):
    """Gauss-Legendre nodes/weights mapped to the interval ``[t_low, t_high]``.

    The affine map from ``[-1, 1]`` is done in pure Torch so autograd flows
    through ``t_low`` / ``t_high`` and ``torch.compile`` / ``vmap`` see no
    graph breaks.

    Parameters
    ----------
    n : int
        Number of quadrature points.
    t_low, t_high : Tensor
        Interval endpoints.  May be scalars or batched tensors; batch
        dimensions broadcast against the node axis.
    device, dtype : optional
        Device/dtype of the cached ``[-1, 1]`` constants.

    Returns
    -------
    (Tensor, Tensor)
        Nodes and weights of shape ``(*batch, n)``.
    """
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
    """Double-exponential (tanh-sinh style) change of variables.

    Maps quadrature nodes ``t`` on a finite interval to
    ``u = exp((pi/2) * sinh(t))`` on ``(0, inf)``, returning both ``u`` and
    the Jacobian ``du/dt = (pi/2) * cosh(t) * u``.  Used to evaluate the
    improper MGE velocity integral with rapidly (double-exponentially)
    decaying truncation error.

    Parameters
    ----------
    t : Tensor
        Quadrature nodes (any shape).

    Returns
    -------
    (Tensor, Tensor)
        ``(u, du_dt)`` with the same shape as ``t``.
    """
    u = torch.exp((np.pi/2.0)*torch.sinh(t))
    du_dt = (np.pi/2.0)*torch.cosh(t)*u
    return u, du_dt
    

def interpolate_velocity(R_grid: torch.Tensor,
                         R_map : torch.Tensor,
                         v_grid: torch.Tensor) -> torch.Tensor:
    """1-D linear interpolation of a rotation curve onto arbitrary radii.

    Any query outside ``[R_grid[0], R_grid[-1]]`` is clamped to the edge
    values.  Works on CUDA tensors, keeps gradients (w.r.t. ``v_grid`` and
    ``R_map``), and never indexes out of bounds, so it is safe inside
    ``torch.func.vmap``.

    Parameters
    ----------
    R_grid : Tensor
        Monotonically increasing 1-D grid of radii (the lookup table axis).
    R_map : Tensor
        Radii to interpolate at; any shape (e.g. a 2-D sky-plane radius map).
    v_grid : Tensor
        Velocities tabulated at ``R_grid`` (same length as ``R_grid``).

    Returns
    -------
    Tensor
        Interpolated velocities with the same shape as ``R_map``.
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
    """Circular velocity of an axisymmetric MGE mass model + black hole.

    Implements the classic MGE circular-velocity quadrature (Cappellari 2002)
    parameterized directly by the **intrinsic** axial ratios ``qintr`` of the
    Gaussians (the observed ``qobs`` is derived internally from ``qintr`` and
    the inclination).  A Keplerian black-hole term with a Plummer softening is
    added in quadrature.

    Parameters (caskade ``Param``\\ s, supplied at call time)
    --------------------------------------------------------
    surf : (N_components,)
        Peak surface density of each Gaussian (e.g. L_sun/pc^2; multiplied by
        ``M_to_L`` to obtain mass surface density).
    sigma : (N_components,)
        Dispersion of each Gaussian [pc].
    qintr : (N_components,)
        Intrinsic (deprojected) axial ratio of each Gaussian.
    M_to_L : scalar or (N_components,)
        Mass-to-light ratio (per-Gaussian if ``variable_M_to_L=True``).
    m_bh : scalar
        ``log10`` of the black-hole mass [M_sun].
    inc : scalar
        Inclination [rad] (used to convert ``qintr`` to the observed
        flattening entering the surface-density normalization).

    Parameters
    ----------
    N_components : int
        Number of Gaussian components.
    device, dtype :
        Torch device / dtype used for all internal constants.
    quad_points : int, optional
        Number of Gauss-Legendre nodes for the (double-exponentially
        transformed) velocity integral.
    radius_res : int, optional
        Length of the logarithmic radius lookup table built by
        :meth:`velocity`; the rotation curve is evaluated on this 1-D grid
        and linearly interpolated onto the requested radii.
    variable_M_to_L : bool, optional
        If True, ``M_to_L`` is a per-Gaussian vector Param instead of a
        scalar.
    soft : float, optional
        Plummer softening length of the black-hole potential [pc].
    G : float, optional
        Gravitational constant; the default ``0.004301``
        [pc (km/s)^2 / M_sun] yields velocities in km/s for radii in pc.
    """
    def __init__(self, N_components: int, device, dtype, quad_points=128, radius_res = 4096, variable_M_to_L = False, soft = 0.0, G=0.004301):
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
        """Evaluate the circular velocity at a 1-D set of radii.

        The MGE integrand is evaluated with Gauss-Legendre quadrature after a
        double-exponential transform of the integration variable from a
        finite interval onto ``(0, inf)``; radii are internally rescaled by
        the median ``sigma`` for numerical conditioning.  The black-hole term
        ``G * 10**m_bh * R^2 / (R^2 + soft^2)^{3/2}`` is added in quadrature.

        Parameters
        ----------
        R_flat : Tensor, shape (N,)
            Radii in the galaxy midplane [pc].
        surf, sigma, qintr, M_to_L, inc, m_bh :
            MGE parameters as described in the class docstring.  This method
            is *not* a caskade forward -- values must be passed explicitly
            (it is called internally by :meth:`velocity`).

        Returns
        -------
        Tensor, shape (N,)
            Circular velocity at each radius [km/s for the default ``G``].
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
        """Circular velocity for every pixel of a radius map (caskade forward).

        Builds a ``radius_res``-point logarithmic lookup table spanning
        ``[soft, R_map.max()]``, evaluates :meth:`radial_velocity` on it once,
        and linearly interpolates onto ``R_map`` — far cheaper than evaluating
        the quadrature per pixel and accurate for any reasonably dense table.

        Parameters
        ----------
        R_map : Tensor
            Radii in the galaxy midplane [pc]; any shape (typically ``(H, W)``).
        surf, sigma, qintr, M_to_L, inc, m_bh : optional
            caskade Params; automatically filled from the module graph when
            the model is called through caskade.

        Returns
        -------
        Tensor
            ``v_rot(R)`` with the same shape as ``R_map``.
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
    """Ridge-regression projector from a 1-D radial profile to MGE amplitudes.

    With a fixed Gaussian basis ``A_ij = exp(-R_i^2 / (2 sigma_j^2))``
    evaluated on radii ``r_grid_pc`` and dispersions ``sigma_grid_pc``, the
    least-squares MGE amplitudes of any profile ``y(R_i)`` are the linear map

    .. math:: \\mathrm{surf} = P\\,y,\\qquad P = (A^T A + \\lambda I)^{-1} A^T

    ``P`` is factored **once** in float64 at construction (Cholesky, with
    automatic jitter escalation ``lam *= 10`` until the factorization
    succeeds), so per-sample profile-to-MGE conversion is a single matmul —
    differentiable and vmap-friendly.

    Parameters
    ----------
    r_grid_pc : Tensor, shape (R,)
        Radii where profiles will be sampled [pc].
    sigma_grid_pc : Tensor, shape (K,)
        Fixed logarithmically spaced dispersions of the Gaussian basis [pc].
    lam_base : float, optional
        Initial ridge regularization strength.
    max_jitter_tries : int, optional
        Maximum number of tenfold ``lam`` increases before giving up.
    dtype, device : optional
        Storage dtype/device of the cached ``A`` and ``P`` buffers.

    Raises
    ------
    RuntimeError
        If ``A^T A + lam I`` cannot be Cholesky-factored even after
        ``max_jitter_tries`` jitter escalations.

    Attributes
    ----------
    A : Tensor, shape (R, K)
        The Gaussian design matrix (forward map ``surf -> profile``).
    P : Tensor, shape (K, R)
        The precomputed ridge projector (``profile -> surf``).
    lam_used : float
        The ridge strength that was actually used.
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
        """Project a radial profile onto MGE amplitudes (``surf = P @ y``).

        Parameters
        ----------
        y : Tensor, shape (R,) or (B, R)
            Profile values sampled at ``r_grid``; a leading batch dimension
            is supported.

        Returns
        -------
        Tensor, shape (K,) or (B, K)
            Least-squares MGE amplitudes (may be negative where the basis
            overshoots; the velocity integral tolerates this).
        """
        if y.ndim == 1:
            return self.P @ y
        elif y.ndim == 2:
            return y @ self.P.T
        raise ValueError(f"y must be 1D or 2D, got {tuple(y.shape)}")

    def profile_from_surf(self, surf: torch.Tensor) -> torch.Tensor:
        """Reconstruct the radial profile from MGE amplitudes (``A @ surf``).

        Parameters
        ----------
        surf : Tensor, shape (K,) or (B, K)
            MGE amplitudes.

        Returns
        -------
        Tensor, shape (R,) or (B, R)
            The profile evaluated on ``r_grid`` — useful to inspect the
            quality of the MGE fit.
        """
        if surf.ndim == 1:
            return self.A @ surf
        elif surf.ndim == 2:
            return surf @ self.A.T
        raise ValueError(f"surf must be 1D or 2D, got {tuple(surf.shape)}")



# -----------------------------
# Shared base
# -----------------------------
class _PreLSMGEBase(Module):
    """Shared machinery for analytic-profile -> MGE velocity models.

    Builds (i) a log-spaced radius grid ``[r_min_pc, r_max_pc]``, (ii) a
    matching fixed sigma basis and :class:`PrecomputedMGEProjector`, and
    (iii) an internal :class:`MGEVelocityIntr` whose ``surf`` Param is
    re-pointed (by the subclass) to a function that evaluates the analytic
    profile and projects it to MGE amplitudes on the fly.  ``sigma`` is
    static, ``M_to_L`` is fixed to 1 (the profile normalization carries the
    mass scale), and the scalar ``qintr`` Param is broadcast to all
    Gaussians.

    Top-level caskade Params: ``inc`` [rad], ``qintr`` (scalar) and ``m_bh``
    (``log10 M_sun``); subclasses add the profile-shape Params.

    Parameters
    ----------
    name : str
        caskade module name.
    N_MGE_components : int
        Number of Gaussians in the fixed sigma basis.
    distance_Mpc : float
        Angular-diameter distance used to convert the profile's arcsec radii
        to pc (``pc_per_arcsec = distance_Mpc * pi / 0.648``).
    soft : float
        Black-hole softening length [pc], forwarded to
        :class:`MGEVelocityIntr`.
    device, dtype :
        Torch device/dtype.
    n_radii_data : int, optional
        Number of geometric radius samples the profile is evaluated on.
    r_min_pc, r_max_pc : float, optional
        Radial range of the profile sampling [pc].
    lam_base, max_jitter_tries : optional
        Ridge settings of :class:`PrecomputedMGEProjector`.
    quad_points, radius_res, G : optional
        Forwarded to :class:`MGEVelocityIntr`.
    """
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
        """Constant ``10`` on the model's device/dtype (for ``10**log10x``)."""
        return torch.tensor(10.0, device=self.device, dtype=self.dtype)

    def _qintr_pointer(self, p):
        """Broadcast the scalar ``qintr`` Param to one value per Gaussian."""
        return p.qintr.value * torch.ones(
            (self.N_MGE_components,),
            device=self.device,
            dtype=self.dtype,
        )


# -----------------------------
# Profile helpers
# -----------------------------
def sersic_profile_torch_1d(R, I_e, R_e, n, eps=1e-30):
    """Sersic (1968) surface-brightness profile, elementwise in Torch.

    ``I(R) = I_e * exp(-b_n * ((R/R_e)^(1/n) - 1))`` with the Ciotti &
    Bertin (1999) asymptotic expansion for ``b_n``.

    Parameters
    ----------
    R : Tensor
        Radii (same units as ``R_e``).
    I_e : Tensor or float
        Intensity at the effective radius (linear units).
    R_e : Tensor or float
        Effective (half-light) radius.
    n : Tensor or float
        Sersic index.
    eps : float, optional
        Lower clamp applied to ``R``, ``R_e`` and ``n`` to avoid 0^negative
        and division by zero.

    Returns
    -------
    Tensor
        ``I(R)`` with the same shape as ``R``.
    """
    R   = torch.clamp(R,   min=eps)
    R_e = torch.clamp(R_e, min=eps)
    n   = torch.clamp(n,   min=eps)

    b_n = 2.0 * n - (1.0 / 3.0) + 4.0 / (405.0 * n) + 46.0 / (25515.0 * n * n)
    return I_e * torch.exp(-b_n * (torch.pow(R / R_e, 1.0 / n) - 1.0))


def nuker_profile_torch_1d(R, I_b, r_b, alpha, beta, gamma, eps=1e-30):
    """Nuker (Lauer et al. 1995) double power-law profile, elementwise.

    ``I(R) = I_b * 2^((beta-gamma)/alpha) * (R/r_b)^(-gamma)
    * (1 + (R/r_b)^alpha)^((gamma-beta)/alpha)``: an inner cusp of slope
    ``-gamma`` breaking to an outer slope ``-beta`` at ``r_b`` with
    sharpness ``alpha``.

    Parameters
    ----------
    R : Tensor
        Radii (same units as ``r_b``).
    I_b : Tensor or float
        Intensity at the break radius (linear units).
    r_b : Tensor or float
        Break radius.
    alpha : Tensor or float
        Break sharpness (larger = sharper).
    beta : Tensor or float
        Outer logarithmic slope.
    gamma : Tensor or float
        Inner logarithmic slope.
    eps : float, optional
        Lower clamp on ``R``, ``r_b``, ``alpha`` and ``R/r_b``.

    Returns
    -------
    Tensor
        ``I(R)`` with the same shape as ``R``.
    """
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
    """Core-Sersic profile (Graham et al. 2003), overflow-safe in float32.

    ``I(R) = I_b * (1 + (R_b/R)^alpha)^(gamma/alpha)
    * exp(-b_n * ((R^alpha + R_b^alpha)/R_e^alpha)^(1/(alpha*n)))
    * exp(b_n)`` — a power-law core of slope ``-gamma`` inside the break
    radius ``R_b`` joined (with sharpness ``alpha``) to an outer Sersic
    profile of index ``n`` and effective radius ``R_e``.

    Both bracketed terms are evaluated in log-space with ``logaddexp`` so no
    intermediate ever forms ``R_e**alpha`` explicitly (which overflows
    float32 for large ``R_e`` and used to manufacture a spurious
    "extended large-R_e" likelihood mode; see the inline comment).

    Parameters
    ----------
    R : Tensor
        Radii (same units as ``R_b``/``R_e``).
    I_b : Tensor or float
        Intensity scale at the break radius (linear units).
    R_b : Tensor or float
        Break (core) radius.
    R_e : Tensor or float
        Effective radius of the outer Sersic component.
    alpha : Tensor or float
        Transition sharpness.
    gamma : Tensor or float
        Inner (core) logarithmic slope.
    n : Tensor or float
        Outer Sersic index.
    eps : float, optional
        Lower clamp on radii, ``alpha`` and ``n``.

    Returns
    -------
    Tensor
        ``I(R)`` with the same shape as ``R``.
    """
    R   = torch.clamp(R,   min=eps)
    R_b = torch.clamp(R_b, min=eps)
    R_e = torch.clamp(R_e, min=eps)
    alpha = torch.clamp(alpha, min=eps)
    n     = torch.clamp(n, min=eps)

    b_n = 2.0 * n - (1.0 / 3.0) + 4.0 / (405.0 * n) + 46.0 / (25515.0 * n * n)

    # Evaluate the inner cusp (term1) and the outer Sérsic roll-off (term2) in LOG-SPACE.
    # The direct algebraic form builds R_e**alpha (also R**alpha, R_b**alpha) explicitly, which
    # OVERFLOWS float32 for large R_e: e.g. R_e~1e4 pc with alpha~9.6 gives R_e**alpha ~ 4e38 >
    # 3.4e38 (float32 max) -> +inf. The overflow sends the roll-off argument to 0, collapsing
    # term2 to the radius-independent constant exp(b_n) and inflating the whole profile by
    # hundreds-to-thousands x — which manufactures a spurious "extended large-R_e" likelihood
    # mode in float32 nested sampling. logaddexp keeps every intermediate finite and is
    # mathematically identical: verified float32==float64 to ~1e-6 and reproducing the original
    # float64 result to ~1e-15.
    log_R, log_Rb, log_Re = torch.log(R), torch.log(R_b), torch.log(R_e)
    # term1 = (1 + (R_b/R)**alpha)**(gamma/alpha)
    log_term1 = (gamma / alpha) * torch.logaddexp(
        torch.zeros_like(log_R), alpha * (log_Rb - log_R))
    # inside = (R**alpha + R_b**alpha) / R_e**alpha   (never formed directly)
    log_inside = torch.logaddexp(alpha * log_R, alpha * log_Rb) - alpha * log_Re
    term2 = torch.exp(-b_n * (torch.exp(log_inside / (alpha * n)) - 1.0))
    return I_b * torch.exp(log_term1) * term2


# -----------------------------
# Sérsic
# -----------------------------
class SersicMGE(_PreLSMGEBase):
    """Circular velocity of a Sersic stellar profile via on-the-fly MGE.

    Each evaluation renders the Sersic profile on the fixed radius grid,
    projects it to MGE amplitudes with the precomputed ridge projector and
    feeds those into :class:`MGEVelocityIntr` — so samplers work directly in
    the profile's natural parameters.

    caskade Params (in addition to the base ``inc``, ``qintr``, ``m_bh``):

    - ``n`` : Sersic index;
    - ``r_e`` : effective radius [arcsec];
    - ``intensity_r_e`` (attribute ``I_e``) : ``log10`` of the intensity at
      ``r_e``.  Because ``M_to_L`` is fixed to 1, this normalization sets
      the stellar-mass scale directly.

    See :class:`_PreLSMGEBase` for the constructor arguments.
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
        """Render the Sersic profile and project it to MGE amplitudes."""
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
        """Circular velocity at radii ``R_flat`` [pc] (delegates to the MGE).

        The profile Params appear in the signature only so caskade routes
        them into the graph; the actual profile evaluation happens inside
        the linked ``surf`` pointer of the internal :class:`MGEVelocityIntr`.
        """
        return self.MGE.velocity(R_map=R_flat)


# -----------------------------
# Nuker
# -----------------------------
class NukerMGE(_PreLSMGEBase):
    """Circular velocity of a Nuker stellar profile via on-the-fly MGE.

    Same construction as :class:`SersicMGE` but with the Nuker double
    power-law of :func:`nuker_profile_torch_1d`.

    caskade Params (in addition to the base ``inc``, ``qintr``, ``m_bh``):

    - ``alpha`` : break sharpness;
    - ``gamma`` : inner slope;
    - ``gamma_minus_beta`` (attribute ``gmb``) : the *difference*
      ``gamma - beta`` — sampling this instead of ``beta`` keeps the
      outer slope steeper than the inner one by construction;
    - ``r_b`` : break radius [arcsec];
    - ``intensity_r_b`` (attribute ``I_b``) : ``log10`` intensity at ``r_b``.

    See :class:`_PreLSMGEBase` for the constructor arguments.
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
        """Render the Nuker profile and project it to MGE amplitudes."""
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
        """Circular velocity at radii ``R_flat`` [pc] (delegates to the MGE)."""
        return self.MGE.velocity(R_map=R_flat)


# -----------------------------
# Core-Sérsic
# -----------------------------
class CoreSersicMGE(_PreLSMGEBase):
    """Circular velocity of a Core-Sersic stellar profile via on-the-fly MGE.

    Same construction as :class:`SersicMGE` but with the (overflow-safe)
    Core-Sersic profile of :func:`core_sersic_torch_1d`.

    caskade Params (in addition to the base ``inc``, ``qintr``, ``m_bh``):

    - ``I_b`` : ``log10`` intensity scale at the break radius;
    - ``R_b`` : break (core) radius [arcsec];
    - ``R_e`` : effective radius of the outer Sersic part [arcsec];
    - ``alpha`` : transition sharpness;
    - ``gamma`` : inner (core) slope;
    - ``n`` : outer Sersic index.

    See :class:`_PreLSMGEBase` for the constructor arguments.
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
        """Render the Core-Sersic profile and project it to MGE amplitudes."""
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
        """Circular velocity at radii ``R_flat`` [pc] (delegates to the MGE)."""
        return self.MGE.velocity(R_map=R_flat)


# ------------------------------------------
# Gas mass profile (exponential disk model)
# ------------------------------------------

class GasSelfGrav(Module):
    """Self-gravity of a razor-thin exponential gas disk.

    Implements the classic Freeman (1970) rotation curve of an infinitely
    thin exponential disk (eq. 8.74 of Bovy, "Dynamics and Astrophysics of
    Galaxies"), written with modified Bessel functions.  The disk scale
    length is *shared* with the intensity model passed to the constructor
    (the same caskade Param object), so the surface-brightness profile and
    the gas mass profile stay consistent during fitting.

    caskade Params: ``m_gas`` (``log10`` of the total gas mass [M_sun]) and
    the shared ``scale`` [pc].

    Parameters
    ----------
    intensity_model : Module
        Model exposing a ``scale`` Param (e.g.
        :class:`supermage.simulators.intensity_models.ExponentialDisk2D`);
        its ``scale`` is reused as this module's scale length.
    device, dtype :
        Unused directly (kept for interface symmetry with other velocity
        models).
    """
    def __init__(self,intensity_model,device,dtype):
        super().__init__("GasSelfGrav")
        self.scale = intensity_model.scale
        self.m_gas = Param("m_gas",  shape=())

    @forward
    def velocity(self, R_flat, m_gas = None, scale = None, G=0.004301):
        """Circular velocity of the gas disk at radii ``R_flat`` [pc].

        Parameters
        ----------
        R_flat : Tensor
            Radii [pc]; any shape.
        m_gas, scale : optional
            caskade Params (``log10`` gas mass, scale length in pc).
        G : float, optional
            Gravitational constant [pc (km/s)^2 / M_sun].

        Returns
        -------
        Tensor
            ``v_circ(R)`` [km/s], same shape as ``R_flat``.
        """
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
    """Compose mass components by summing circular velocities in quadrature.

    ``v_total(R) = sqrt(sum_i v_i(R)^2)`` — the correct composition rule for
    independent axisymmetric mass components (stars + gas + ...).

    If any child model exposes an inclination Param (attribute ``inc`` or
    ``inclination``), a single shared top-level ``inclination`` Param is
    created and pointed into all such children, so the composite model has
    one inclination.

    Parameters
    ----------
    models : tuple of Module
        Velocity models, each exposing ``velocity(R_flat) -> Tensor`` (e.g.
        :class:`SersicMGE`, :class:`GasSelfGrav`).  They are registered as
        child modules ``model_0``, ``model_1``, ...
    name : str, optional
        caskade module name (default ``"MultiComponentMass"``).
    """
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
        """Total circular velocity at radii ``R_flat`` [pc].

        Parameters
        ----------
        R_flat : Tensor
            Radii [pc]; any shape.
        inclination : optional
            The shared inclination Param (only present when at least one
            child model has an inclination).

        Returns
        -------
        Tensor
            ``sqrt(sum_i v_i^2)`` with the same shape as ``R_flat``.
        """
        v2_total = torch.zeros_like(R_flat)

        for name in self._model_names:
            model = getattr(self, name)
            v_cur = model.velocity(R_flat)
            v2_total = v2_total + v_cur**2

        return torch.sqrt(v2_total)


