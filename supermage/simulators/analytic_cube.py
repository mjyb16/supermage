"""Unlensed spectral-cube simulators (image-plane renderers).

Two rendering strategies produce a ``(N_chan, N_pix, N_pix)`` cube from an
intensity model plus a velocity model:

* :class:`AnalyticInverse` — deterministic, grid-based: builds an oversampled
  sky-plane grid, inverts the disk projection pixel-by-pixel, evaluates the
  analytic intensity/velocity fields, broadens each pixel into ``K_vel``
  equal-flux Gaussian quantile sub-channels, and box-filters down to the
  requested resolution.  Recommended default.
* :class:`CloudCatalog` + :class:`CloudRasterizerOversample` — Monte-Carlo,
  KinMS-style: a static low-discrepancy catalogue of "clouds" in the disk
  plane is projected to the sky and rasterized with trilinear
  (position-position-velocity) weighting onto an oversampled cube.

Conventions shared by both:

* the disk lies in the ``(x_gal, y_gal)`` plane [pc]; ``+x_gal`` is the
  receding side;
* ``inclination`` and ``sky_rot`` are in **radians**; the position angle used
  internally is ``pa = sky_rot + pi/2``;
* ``x0``/``y0`` are sky offsets in **arcsec** (east/north positive);
* velocities are in km/s; the velocity axis is derived from the frequency
  axis with :func:`supermage.utils.doppler_velocities.create_velocity_grid_stable`
  using the rest-frame ``line`` frequency — ``freq_axis`` and ``line`` must
  be in the **same units** (the conversion is ratio-based).
"""
import math, torch
from caskade import Module, Param, forward            # same import style you use
import torch.nn.functional as F
from supermage.utils.doppler_velocities import create_velocity_grid_stable
from supermage.simulators.velocity_scatter import scatter_quantiles_along_v
# ----------------------------------------------------------------------
# Helper: equal-probability Gaussian abscissae -------------------------
# ----------------------------------------------------------------------
def gaussian_quantile_offsets_flex(sigma, K, *, device, dtype):
    """Deterministic mid-quantile velocity offsets for ``N(0, sigma^2)``.

    Splits a Gaussian line profile into ``K`` equal-probability sub-channels
    by taking the inverse CDF at the mid-quantiles ``(k + 0.5)/K`` — a
    deterministic, differentiable alternative to Monte-Carlo velocity
    jitter.

    Parameters
    ----------
    sigma : Tensor
        Line-broadening dispersion [km/s]; scalar or per-pixel ``(H, W)``.
    K : int
        Number of quantile sub-channels.
    device, dtype :
        Placement of the quantile constants.

    Returns
    -------
    Tensor
        Offsets of shape ``(K, 1, 1)`` for scalar ``sigma`` or ``(K, H, W)``
        for per-pixel ``sigma``, broadcastable against a velocity map.
    """
    p_mid = (torch.arange(K, device=device, dtype=dtype) + 0.5) / K
    unit = math.sqrt(2.0) * torch.erfinv(2.0 * p_mid - 1.0)  # (K,)
    if sigma.ndim == 0:
        return (sigma * unit).view(K, 1, 1)
    else:
        return unit.view(K, 1, 1) * sigma.view(1, *sigma.shape)

def make_dv_table(N_clouds, K_vel, *, seed, device, dtype):
    """Reproducible unit-Gaussian velocity-jitter table for cloud catalogues.

    Draws an ``(N_clouds, K_vel)`` matrix from a scrambled Sobol
    low-discrepancy sequence mapped through the inverse normal CDF, so each
    cloud's ``K_vel`` jitters are stratified but rows differ from each other.
    Used by :class:`CloudCatalog` when ``gaussian_quantile=False``.

    Parameters
    ----------
    N_clouds : int
        Number of clouds (rows).
    K_vel : int
        Sub-samples per cloud (columns / Sobol dimensions).
    seed : int
        Sobol scramble seed (reproducibility).
    device, dtype :
        Placement of the returned table.

    Returns
    -------
    Tensor, shape (N_clouds, K_vel)
        Standard-normal jitter values; multiply by a sigma to scale.
    """
    # 1. reproducible uniform [0,1) matrix
    sobol = torch.quasirandom.SobolEngine(
        dimension=K_vel, scramble=True, seed=seed
    )
    u = sobol.draw(int(N_clouds)).to(device=device, dtype=dtype)   # (N,K)

    # 2. map uniform → standard normal N(0,1)
    return math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)       # (N,K)


# ----------------------------------------------------------------------
# Inverse-mapped analytic renderer
# ----------------------------------------------------------------------
class AnalyticInverse(Module):
    """Deterministic inverse-mapped spectral-cube renderer.

    For every pixel of an oversampled sky-plane grid this renderer

    1. subtracts the source offsets ``(x0, y0)`` [arcsec],
    2. inverts the disk projection (rotation by the position angle,
       de-foreshortening by ``cos i``) to intrinsic coordinates
       ``(x_gal, y_gal)`` [pc],
    3. evaluates the analytic ``intensity_model.brightness(R)`` and
       ``velocity_model.velocity(R)`` fields,
    4. projects the circular velocity to the line of sight
       (``v_los = v_circ * sin(i) * cos(theta) + velocity_shift``),
    5. spreads each pixel's flux over ``K_vel`` deterministic Gaussian
       quantile sub-channels of width ``line_broadening``, and
    6. box-filters the hi-res ``(Nv*ov_v, N*ov_xy, N*ov_xy)`` cube down to
       the requested ``(Nv, N_pix, N_pix)``.

    Being fully deterministic and built from differentiable ops, it is safe
    for gradients and ``torch.func.vmap`` batching.

    caskade Params: ``inclination`` [rad], ``sky_rot`` [rad],
    ``line_broadening`` [km/s], ``velocity_shift`` [km/s], ``x0``/``y0``
    [arcsec], ``distance_pc`` [pc], plus everything owned by the intensity
    and velocity sub-models.

    Parameters
    ----------
    intensity_model : Module
        Analytic brightness model, ``brightness(R) -> (H, W)``.
    velocity_model : Module
        Analytic circular-velocity model, ``velocity(R) -> (H, W)``; its
        ``inc`` Param is pointed at this module's ``inclination``.
    freq_axis : sequence, shape (Nv,)
        Uniform frequency axis of the output cube.  Must be in the same
        units as ``line`` (the frequency-to-velocity conversion is
        ratio-based).
    pixel_scale_arcsec : float
        Output pixel scale [arcsec / pixel].
    N_pix_x : int
        Output cube side length [pixels].
    K_vel : int, optional
        Number of Gaussian quantile sub-channels per pixel.
    oversamp_xy : int, optional
        Spatial oversampling factor of the internal hi-res grid.
    oversamp_v : int, optional
        Velocity oversampling factor of the internal hi-res grid.
    chunk_v : int, optional
        Reserved for chunked processing (currently unused here).
    device, dtype :
        Torch placement of grids and the rendered cube.
    line : float, optional
        Rest-frame line frequency, in the same units as ``freq_axis``
        (default 230.538, i.e. CO(2-1) in GHz).
    name : str, optional
        caskade module name.
    """

    def __init__(
        self,
        intensity_model,           # analytic model: brightness(R)
        velocity_model,            # analytic model: velocity(R)
        freq_axis,                 # (Nv,) uniform freqs for output cube
        pixel_scale_arcsec,        # arcsec / pixel on image plane
        N_pix_x,                   # output pixels side (square)
        *,
        K_vel: int = 8,
        oversamp_xy: int = 4,
        oversamp_v: int = 4,
        chunk_v: int | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        line: float = 230.538,
        name: str = "analytic_inverse",
    ):
        super().__init__(name)
        self.device, self.dtype = device, dtype
        self.intensity_model = intensity_model
        self.velocity_model  = velocity_model

        # User-facing parameters
        self.inclination     = Param("inclination",     None)   # [rad]
        self.velocity_model.inc = self.inclination
        self.sky_rot         = Param("sky_rot",         None)   # [rad]
        self.line_broadening = Param("line_broadening", None)   # [km/s]
        self.velocity_shift  = Param("velocity_shift",  None)   # [km/s]
        self.x0              = Param("x0",              None)   # [arcsec]
        self.y0              = Param("y0",              None)   # [arcsec]
        self.distance_pc     = Param("distance_pc",     None)   # [pc]

        # Velocity grid
        from supermage.utils.doppler_velocities import create_velocity_grid_stable
        vel_axis, _ = create_velocity_grid_stable(
            f_start=freq_axis[0],
            f_end=freq_axis[-1],
            num_points=len(freq_axis),
            target_dtype=dtype,
            line=line,
        )
        self.vel0_lo = vel_axis[0].to(dtype)
        self.dv_lo   = float((vel_axis[1] - vel_axis[0]).item())
        self.Nv_lo   = int(vel_axis.numel())

        self.oversamp_v = int(oversamp_v)
        self.dv_hi      = self.dv_lo / self.oversamp_v
        delta           = 0.5 * (self.dv_lo - self.dv_hi)
        self.vel0_hi    = self.vel0_lo - delta
        self.Nv_hi      = self.Nv_lo * self.oversamp_v

        self.K_vel   = int(K_vel)
        self.chunk_v = int(chunk_v) if (chunk_v is not None) else None

        # Spatial grids
        self.pixscale_lo = float(pixel_scale_arcsec)
        self.N_pix_lo    = int(N_pix_x)
        self.N_pix       = self.N_pix_lo
        self.fov_half_lo = 0.5 * (self.N_pix_lo - 1) * self.pixscale_lo

        self.oversamp_xy = int(oversamp_xy)
        self.pixscale_hi = self.pixscale_lo / self.oversamp_xy
        self.N_pix_hi    = self.N_pix_lo * self.oversamp_xy
        self.fov_half_hi = 0.5 * (self.N_pix_hi - 1) * self.pixscale_hi

        # Build sky-plane grid directly in arcsec
        xs = (-self.fov_half_hi) + self.pixscale_hi * torch.arange(
            self.N_pix_hi, device=device, dtype=dtype
        )
        ys = (-self.fov_half_hi) + self.pixscale_hi * torch.arange(
            self.N_pix_hi, device=device, dtype=dtype
        )
        self.xsky = xs.view(1, -1).expand(self.N_pix_hi, -1)   # (H,W), east
        self.ysky = ys.view(-1, 1).expand(-1, self.N_pix_hi)   # (H,W), north

        # Precompute flat spatial indices
        yy = torch.arange(self.N_pix_hi, device=device)
        xx = torch.arange(self.N_pix_hi, device=device)
        Y, X = torch.meshgrid(yy, xx, indexing="ij")
        self.Y_flat = Y.reshape(1, -1)
        self.X_flat = X.reshape(1, -1)
        self.hw     = int(self.N_pix_hi * self.N_pix_hi)

    def _sky_to_intrinsic(self, x_sky_arcsec, y_sky_arcsec, *, x0, y0, pa, cos_i, arcsec_per_pc):
        """Invert the sky projection to intrinsic disk coordinates.

        The forward projection is::

           x_sky =  cos(pa) x_gal - sin(pa) (y_gal cos i)
           y_sky =  sin(pa) x_gal + cos(pa) (y_gal cos i)

        This subtracts the source offsets, converts arcsec -> pc, and applies
        the inverse rotation + de-foreshortening.

        Returns
        -------
        (Tensor, Tensor, Tensor)
            ``(x_gal, y_gal, R)`` in pc, each shaped like the input maps.
        """
        bx = x_sky_arcsec - x0
        by = y_sky_arcsec - y0

        X = bx / arcsec_per_pc
        Y = by / arcsec_per_pc

        cos_pa, sin_pa = torch.cos(pa), torch.sin(pa)
        x_gal =  cos_pa * X + sin_pa * Y
        y_gal = (-sin_pa * X + cos_pa * Y) / (cos_i + 1e-12)
        R = torch.hypot(x_gal, y_gal)
        return x_gal, y_gal, R

    def _bin_quantiles_along_v_(self, cube_hi, v_los, I_map, sigma):
        """Scatter K equal-flux quantile sub-channels of every pixel into ``cube_hi``.

        Thin wrapper around
        :func:`supermage.simulators.velocity_scatter.scatter_quantiles_along_v`
        (which drops out-of-band flux instead of folding it into the edge
        channels).  ``sigma`` may be scalar or per-pixel ``(H, W)``.
        """
        K = self.K_vel
        Δv = gaussian_quantile_offsets_flex(
            sigma.abs() + 1e-12, K, device=self.device, dtype=self.dtype
        )

        v_sub = v_los.view(1, *v_los.shape) + Δv
        return scatter_quantiles_along_v(
            cube_hi, v_sub, I_map,
            vel0_hi=self.vel0_hi, dv_hi=self.dv_hi, Nv_hi=self.Nv_hi,
            Y_flat=self.Y_flat, X_flat=self.X_flat,
            N_pix_hi=self.N_pix_hi, hw=self.hw,
        )

    @forward
    def forward(
        self,
        inclination=None,
        sky_rot=None,
        line_broadening=None,
        velocity_shift=None,
        x0=None,
        y0=None,
        distance_pc=None,
        return_intermediates: bool = False,
    ):
        """Render the spectral cube (caskade forward).

        Parameters
        ----------
        inclination, sky_rot, line_broadening, velocity_shift, x0, y0, distance_pc : optional
            caskade Params (see the class docstring for units).
        return_intermediates : bool, optional
            If True, also return a dict with the intermediate maps
            (``I_map``, ``v_los``, ``R``, ``x_gal``, ``y_gal``, ``x_sky``,
            ``y_sky``) on the hi-res grid — useful for debugging geometry.

        Returns
        -------
        Tensor, shape (Nv, N_pix, N_pix)
            The rendered cube (relative flux units), or
            ``(cube, intermediates)`` when ``return_intermediates=True``.
        """
        cos_i = torch.cos(inclination)
        pa    = sky_rot + math.pi / 2.0
        arcsec_per_pc = 206265.0 / distance_pc

        # Direct sky-plane -> intrinsic
        x_gal, y_gal, R = self._sky_to_intrinsic(
            self.xsky, self.ysky,
            x0=x0, y0=y0, pa=pa, cos_i=cos_i, arcsec_per_pc=arcsec_per_pc
        )

        # Analytic fields
        I_map  = self.intensity_model.brightness(R)
        v_circ = self.velocity_model.velocity(R)
        cos_theta = x_gal / (R + 1e-12)
        v_los = v_circ * torch.sin(inclination) * cos_theta + velocity_shift

        # Allocate hi-res cube
        cube_hi = torch.zeros(
            self.Nv_hi, self.N_pix_hi, self.N_pix_hi,
            device=self.device, dtype=self.dtype
        )

        # Velocity broadening
        cube_hi = self._bin_quantiles_along_v_(cube_hi, v_los, I_map, line_broadening)

        # Box-filter to low-res
        cube_hi = cube_hi.view(
            self.Nv_lo, self.oversamp_v,
            self.N_pix_lo, self.oversamp_xy,
            self.N_pix_lo, self.oversamp_xy
        )
        cube_lo = cube_hi.mean((1, 3, 5))

        if return_intermediates:
            return cube_lo, {
                "I_map": I_map,
                "v_los": v_los,
                "R": R,
                "x_gal": x_gal,
                "y_gal": y_gal,
                "x_sky": self.xsky,
                "y_sky": self.ysky,
            }
        return cube_lo


# ----------------------------------------------------------------------
#  MC cloud catalogue           (only forward() was modified)
# ----------------------------------------------------------------------

def gaussian_quantile_offsets(sigma, K, *, device, dtype):
    """Deterministic mid-quantile offsets for a scalar ``N(0, sigma^2)``.

    1-D version of :func:`gaussian_quantile_offsets_flex`: returns the ``K``
    inverse-CDF values at mid-quantiles, scaled by ``sigma``, shape ``(K,)``.
    """
    p_mid = (torch.arange(K, device=device, dtype=dtype) + 0.5) / K
    return sigma * math.sqrt(2.0) * torch.erfinv(2.0 * p_mid - 1.0)


class CloudCatalog(Module):
    """Monte-Carlo "cloud" catalogue for KinMS-style cube simulation.

    A static set of ``N_clouds`` positions is drawn once (uniform or
    scrambled-Sobol) over the square disk-plane field of view; at call time
    each cloud is assigned a flux from the intensity model, a line-of-sight
    velocity from the velocity model, projected to the sky, and split into
    ``K_vel`` velocity sub-samples (deterministic Gaussian quantiles by
    default, or the precomputed Sobol jitter table).  Feed the result to
    :class:`CloudRasterizerOversample` to obtain a cube.

    Because the catalogue positions are fixed, all randomness is frozen at
    construction — repeated forward calls are deterministic and
    differentiable w.r.t. the physical parameters.

    caskade Params: ``inclination`` [rad], ``sky_rot`` [rad],
    ``line_broadening`` [km/s], ``velocity_shift`` [km/s], ``x0``/``y0``
    [arcsec] (east/north positive), plus the sub-models' Params.

    Parameters
    ----------
    intensity_model : Module
        ``brightness(R) -> flux per cloud`` (relative units).
    velocity_model : Module
        ``velocity(R) -> circular velocity`` [km/s]; its ``inc`` Param is
        pointed at this module's ``inclination``.
    fov_half_pc : float
        Half-width of the square field of view the clouds populate [pc].
    N_clouds : int
        Number of clouds.
    K_vel : int
        Velocity sub-samples per cloud.
    brightness_init :
        Unused legacy argument (kept for call-site compatibility).
    distance_pc : float
        Distance used for the pc -> arcsec conversion.
    sampling_method : {"sobol_uniform", "uniform"}, optional
        Low-discrepancy Sobol (default) or pseudo-random uniform positions.
    seed : int, optional
        Seed for the position draw and the jitter table.
    device, dtype :
        Torch placement of the catalogue.
    name : str, optional
        caskade module name.
    """
    def __init__(
        self,
        intensity_model,
        velocity_model,
        fov_half_pc,
        N_clouds,
        K_vel,
        brightness_init,
        distance_pc,
        sampling_method = "sobol_uniform",
        seed=42,
        device="cuda",
        dtype=torch.float64,
        name="clouds",
    ):
        super().__init__(name)
        self.device, self.dtype = device, dtype
        self.intensity_model, self.velocity_model = intensity_model, velocity_model
        self.K_vel, self.D_pc = K_vel, float(distance_pc)

        # ---------- static MC catalogue ---------------------------------
        if sampling_method == "sobol_uniform":
            # --- low‑discrepancy Sobol points over the square FoV ------
            sobol = torch.quasirandom.SobolEngine(
                dimension=2, scramble=True, seed=seed
            )
            # draw in [-1,1]^2 then scale to pc
            self.pos_gal0 = ((sobol.draw(int(N_clouds), dtype = dtype) * 2.0 - 1.0) * fov_half_pc)
            self.pos_gal0 = self.pos_gal0.to(device = device)

        elif sampling_method == "uniform":
            # existing pure‑uniform sampler (unchanged) -----------------
            gen = torch.Generator(device).manual_seed(seed)
            self.pos_gal0 = (
                torch.rand((N_clouds, 2), generator=gen,
                           device=device, dtype=dtype) * 2.0 - 1.0
            ) * fov_half_pc

        else:
            raise ValueError(f"Unknown sampling_method '{sampling_method}'.")

        # ---------- velocity‑broadening template ------------------------
        self.dv_template = gaussian_quantile_offsets(
            torch.ones((), device=device, dtype=dtype),
            K_vel, device=device, dtype=dtype,
        )

        self.dv_unit = make_dv_table(N_clouds, K_vel,
                          seed=seed, device=device, dtype=dtype)

        # ---------- global fit parameters -------------------------------
        self.inclination     = Param("inclination", None)   # rad
        self.sky_rot         = Param("sky_rot", None)       # rad
        self.line_broadening = Param("line_broadening", None)
        self.velocity_shift  = Param("velocity_shift", None)
        self.x0              = Param("x0", None)            # ″  (–ΔRA)
        self.y0              = Param("y0", None)            # ″  (+ΔDec)

        # pass inclination to nested model for autograd
        self.velocity_model.inc = self.inclination

    # ------------------------------------------------------------------
    @forward
    def forward(
        self,
        inclination=None,
        sky_rot=None,
        line_broadening=None,
        velocity_shift=None,
        x0=None,
        y0=None,
        return_subsamples: bool = False,
        gaussian_quantile = True
    ):
        """Project the cloud catalogue to sky coordinates and velocities.

        Parameters
        ----------
        inclination, sky_rot, line_broadening, velocity_shift, x0, y0 : optional
            caskade Params (units in the class docstring).
        return_subsamples : bool, optional
            If True return a tuple ``(pos_img, vel_chan, flux)`` (the layout
            :class:`CloudRasterizerOversample` consumes); otherwise return
            the same three tensors in a dict.
        gaussian_quantile : bool, optional
            If True (default) use deterministic Gaussian quantile offsets
            for the ``K_vel`` velocity sub-samples; if False use the
            precomputed per-cloud Sobol jitter table (stochastic-looking but
            frozen at construction).

        Returns
        -------
        pos_img : Tensor, shape (N_clouds, K_vel, 2)
            Sky positions [arcsec]; last axis is (RA-east, Dec-north).
        vel_chan : Tensor, shape (N_clouds, K_vel)
            Line-of-sight velocities of each sub-sample [km/s].
        flux : Tensor, shape (N_clouds, K_vel)
            Flux carried by each sub-sample (cloud flux / K_vel).
        """
        # -------- aliases & trig ---------------------------------------
        x_gal, y_gal = self.pos_gal0.T                        # pc
        cos_i, sin_i = torch.cos(inclination), torch.sin(inclination)
        pa      = sky_rot + math.pi / 2.0                  # keep your variable name
        cos_pa  = torch.cos(pa)
        sin_pa  = torch.sin(pa)

        # -------- intrinsic radius & dynamics --------------------------
        R = torch.hypot(x_gal, y_gal)
        flux_cloud = self.intensity_model.brightness(R)
        v_circ     = self.velocity_model.velocity(R)
        cos_theta =  x_gal / (R + 1e-12)            #  +x_gal = receding
        v_los     =  v_circ * sin_i * cos_theta + velocity_shift

        # ------------------------------------------------------------------
        # 2) sky‑plane projection  (inverse of grid simulator)
        # ------------------------------------------------------------------
        x_sky_pc =  cos_pa * x_gal - sin_pa * (y_gal * cos_i)   #  east  (+)
        y_sky_pc =  sin_pa * x_gal + cos_pa * (y_gal * cos_i)   #  north (+)

        # ------------------------------------------------------------------
        # 3) pc → arcsec  + global offsets  (+x0 = shift to the **right**)
        # ------------------------------------------------------------------
        arcsec_per_pc = 206265.0 / self.D_pc
        ra_east   =  x_sky_pc * arcsec_per_pc + x0      # +x0  = shift right
        dec_north =  y_sky_pc * arcsec_per_pc + y0      # +y0  = shift up

        # ------------------------------------------------------------------
        # 4) velocity broadening  (flip sign so red = receding = north)
        # ------------------------------------------------------------------
        if gaussian_quantile:
            Δv_k     = gaussian_quantile_offsets(
                  line_broadening, self.K_vel, device=self.device, dtype=self.dtype)
            vel_chan =  v_los.unsqueeze(-1) + Δv_k
            flux_sub = flux_cloud.unsqueeze(-1).expand(-1, self.K_vel) / self.K_vel
        else:
            Δv_k = line_broadening.unsqueeze(-1) * self.dv_unit       # broadcast σ
            vel_chan = v_los.unsqueeze(-1) + Δv_k                     # (N,K)
        
            flux_sub = (flux_cloud / self.K_vel)[:, None].expand(-1, self.K_vel)

        # ------------------------------------------------------------------
        # 5) broadcast spatial coordinates   **horizontal = RA**
        # ------------------------------------------------------------------
        pos_img = torch.stack([ra_east, dec_north], dim=-1) \
                      .unsqueeze(1).expand(-1, self.K_vel, -1).clone()

        return (pos_img, vel_chan, flux_sub) if return_subsamples else {
            "pos_img": pos_img, "vel_chan": vel_chan, "flux": flux_sub
        }


class CloudRasterizerOversample(Module):
    """Trilinear PPV rasterizer for :class:`CloudCatalog` output.

    Recommended rasterizer for cloud-based models.  Cloud sub-samples are
    deposited with trilinear (cloud-in-cell) weights onto a spatially and
    spectrally oversampled position-position-velocity grid, which is then
    box-filtered down to the requested resolution — ensuring Nyquist
    sampling of steep spatio-spectral variations (e.g. near a black hole).
    Sub-samples falling outside the hi-res cube are dropped (their weights
    are zeroed, not clamped into the edge voxels).

    Parameters
    ----------
    cloudcatalog : CloudCatalog
        The cloud generator whose forward output is rasterized.
    freq_axis : sequence, shape (Nv,)
        Uniform frequency axis of the output cube; same units as ``line``.
    pixel_scale_arcsec : float
        Output pixel scale [arcsec / pixel].
    N_pix_x : int
        Output cube side length [pixels].
    oversamp_xy : int, optional
        Spatial oversampling factor of the internal hi-res grid.
    oversamp_v : int, optional
        Velocity oversampling factor of the internal hi-res grid.
    device, dtype :
        Torch placement.
    line : float, optional
        Rest-frame line frequency, same units as ``freq_axis`` (default
        230.538, i.e. CO(2-1) in GHz).
    name : str, optional
        caskade module name.
    """
    # ──────────────────────────────────────────────────────────────────
    def __init__(self,
                 cloudcatalog,
                 freq_axis,              # (Nv,) uniform
                 pixel_scale_arcsec,
                 N_pix_x,
                 oversamp_xy: int = 4,
                 oversamp_v : int = 4,   # NEW: velocity oversampling
                 device: str = "cuda",
                 dtype : torch.dtype = torch.float32,
                 line = 230.538,
                 name  : str = "raster"):
        super().__init__()
        self.device, self.dtype = device, dtype
        self.clouds      = cloudcatalog
        self.oversamp_xy = int(oversamp_xy)
        self.oversamp_v  = int(oversamp_v)

        # ── low‑res velocity grid ────────────────────────────
        # velocity axis -------------------------------------------------
        vel_axis, dv = create_velocity_grid_stable(f_start = freq_axis[0], f_end = freq_axis[-1], num_points = len(freq_axis), target_dtype = dtype, line = line)
        self.vel0_lo = vel_axis[0].to(dtype)
        self.dv_lo   = float((vel_axis[1] - vel_axis[0]).item())
        self.Nv_lo   = vel_axis.numel()

        # ── high‑res velocity grid (offset by δ) ─────────────────────
        self.dv_hi   = self.dv_lo / self.oversamp_v
        delta        = 0.5 * (self.dv_lo - self.dv_hi)      # centre‑align shift
        self.vel0_hi = self.vel0_lo - delta                 # **key change**
        self.Nv_hi   = self.Nv_lo * self.oversamp_v

        # ── low‑res spatial grid ─────────────────────────────────────
        self.pixscale_lo = float(pixel_scale_arcsec)
        self.N_pix_lo    = int(N_pix_x)
        self.N_pix = self.N_pix_lo # Makes API compatible with the other rasterizers
        self.fov_half_lo = 0.5 * (self.N_pix_lo - 1) * self.pixscale_lo

        # ── high‑res spatial grid ────────────────────────────────────
        self.pixscale_hi = self.pixscale_lo / self.oversamp_xy
        self.N_pix_hi    = self.N_pix_lo * self.oversamp_xy
        self.fov_half_hi = 0.5 * (self.N_pix_hi - 1) * self.pixscale_hi
        self.cube_flat = torch.zeros(
            self.Nv_hi * self.N_pix_hi * self.N_pix_hi,
            device=device,
            dtype=dtype
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _index_and_frac(x: torch.Tensor):
        """Split continuous grid coordinates into ``(floor index, fraction)``."""
        i0 = torch.floor(x).to(torch.long)
        return i0, x - i0.to(x.dtype)

    # ------------------------------------------------------------------
    def _rasterise_hi(self, ra, dec, vel, flux):
        """Trilinearly scatter flattened sub-samples onto the hi-res PPV cube.

        Parameters
        ----------
        ra, dec : Tensor, shape (M,)
            Sky positions [arcsec].
        vel : Tensor, shape (M,)
            Line-of-sight velocities [km/s].
        flux : Tensor, shape (M,)
            Flux per sub-sample.

        Returns
        -------
        Tensor, shape (Nv_hi, N_pix_hi, N_pix_hi)
            The hi-res cube; out-of-bounds sub-samples contribute nothing.
        """
        # --- 1. continuous indices and fractional parts -------------------------
        ix0_f, fx = self._index_and_frac((ra  + self.fov_half_hi) / self.pixscale_hi)
        iy0_f, fy = self._index_and_frac((dec + self.fov_half_hi) / self.pixscale_hi)
        iv0_f, fv = self._index_and_frac((vel - self.vel0_hi)     / self.dv_hi)
    
        # neighbour indices before clamping
        ix1_f, iy1_f, iv1_f = ix0_f + 1, iy0_f + 1, iv0_f + 1
    
        # --- 2. “is this point inside the cube?” --------------------------------
        valid = (
            (ix0_f >= 0) & (ix0_f < self.N_pix_hi - 1) &
            (iy0_f >= 0) & (iy0_f < self.N_pix_hi - 1) &
            (iv0_f >= 0) & (iv0_f < self.Nv_hi   - 1)
        )
    
        # --- 3. clamp indices so they’re always legal (static shape!) -----------
        ix0 = ix0_f.clamp(0, self.N_pix_hi - 1).long()
        iy0 = iy0_f.clamp(0, self.N_pix_hi - 1).long()
        iv0 = iv0_f.clamp(0, self.Nv_hi   - 1).long()
        ix1 = ix1_f.clamp(0, self.N_pix_hi - 1).long()
        iy1 = iy1_f.clamp(0, self.N_pix_hi - 1).long()
        iv1 = iv1_f.clamp(0, self.Nv_hi   - 1).long()
    
        # --- 4. weights; 0 out the invalid ones ---------------------------------
        w_valid = valid.to(flux.dtype)              # (M,)
        wx0, wy0, wv0 = (1 - fx) * w_valid, (1 - fy) * w_valid, (1 - fv) * w_valid
        wx1, wy1, wv1 =      fx  * w_valid,      fy  * w_valid,      fv  * w_valid
    
        # stack neighbours exactly as before (shape = (M, 8))
        ix = torch.stack([ix0, ix0, ix0, ix0, ix1, ix1, ix1, ix1], dim=1)
        iy = torch.stack([iy0, iy0, iy1, iy1, iy0, iy0, iy1, iy1], dim=1)
        iv = torch.stack([iv0, iv1, iv0, iv1, iv0, iv1, iv0, iv1], dim=1)
        wx = torch.stack([wx0, wx0, wx0, wx0, wx1, wx1, wx1, wx1], dim=1)
        wy = torch.stack([wy0, wy1, wy0, wy1, wy0, wy1, wy0, wy1], dim=1)
        wv = torch.stack([wv0, wv1, wv0, wv1, wv0, wv1, wv0, wv1], dim=1)
    
        f_w = flux.unsqueeze(1) * (wx * wy * wv)          # still (M, 8)
    
        # --- 5. scatter‑add – now indices are always valid ----------------------
        idx_flat = (iv * self.N_pix_hi + iy) * self.N_pix_hi + ix
        cube_scattered = torch.scatter_add(self.cube_flat, 0, idx_flat.reshape(-1), f_w.reshape(-1))
    
        return cube_scattered.view(self.Nv_hi, self.N_pix_hi, self.N_pix_hi)

    # ------------------------------------------------------------------
    @forward
    def forward(self):
        """Generate clouds, rasterize hi-res, box-filter to the output grid.

        Returns
        -------
        Tensor, shape (Nv_lo, N_pix_lo, N_pix_lo)
            The rendered spectral cube (relative flux units).
        """
        pos_img, vel_chan, flux = self.clouds.forward(return_subsamples=True)
        M = pos_img.numel() // 2

        ra  = pos_img[..., 0].reshape(M)
        dec = pos_img[..., 1].reshape(M)
        vel = vel_chan.reshape(M)
        flx = flux.reshape(M)

        # ---- high‑res raster ----------------------------------------
        cube_hi = self._rasterise_hi(ra, dec, vel, flx)
        #     shape: (Nv_hi, N_pix_hi, N_pix_hi)

        # ---- box‑filter / down‑sample in v and (x,y) ----------------
        cube_hi = cube_hi.view(
            self.Nv_lo,  self.oversamp_v,
            self.N_pix_lo, self.oversamp_xy,
            self.N_pix_lo, self.oversamp_xy
        )
        cube_lo = cube_hi.mean((1, 3, 5))           # average over v,x,y sub‑cells

        return cube_lo