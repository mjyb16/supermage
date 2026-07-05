"""Gravitationally lensed spectral-cube simulators (caustics integration).

Two ways to lens a rotating-disk cube through a
`caustics <https://github.com/Ciela-Institute/caustics>`_ lens model:

* :class:`CubeLens` — lens an *already rendered* source cube channel by
  channel through a ``Pixelated`` caustics source (interpolation in the
  source plane).  Works with any source cube simulator.
* :class:`AnalyticLens` — inverse-mapped analytic renderer: raytrace the
  image-plane grid to the source plane once, then evaluate the analytic
  intensity/velocity fields *directly at the raytraced positions*.  No
  source-plane pixelization, so it preserves the steep central velocity
  gradients that matter for black-hole work.  Recommended for lensed
  kinematics.

Both expect image-plane coordinates in arcsec and share the disk-projection
conventions of :mod:`supermage.simulators.analytic_cube`.
"""
import torch
import math
from caskade import Module, forward, Param
from torch import vmap
import caustics
from caustics.light import Pixelated
from torch.nn.functional import avg_pool2d, conv2d

from supermage.simulators.velocity_scatter import scatter_quantiles_along_v


class CubeLens(Module):
    """Lens a rendered source cube channel-by-channel through a caustics lens.

    The source cube produced by ``source_cube.forward()`` is wrapped in a
    caustics ``Pixelated`` source; each channel is raytraced on an
    oversampled image-plane grid (``vmap`` over channels) and average-pooled
    down to the requested lens-plane resolution.

    Parameters
    ----------
    lens : caustics lens
        Any lens exposing ``raytrace(thx, thy) -> (bx, by)`` [arcsec].
    source_cube : Module
        Cube simulator whose forward returns ``(N_chan, S, S)``.
    pixelscale_source : float
        Source-plane pixel scale of the ``Pixelated`` interpolator [arcsec].
    pixelscale_lens : float
        Output image-plane pixel scale [arcsec].
    pixels_x_source : int
        Source-plane cube side [pixels].
    pixels_x_lens : int
        Output image-plane side [pixels].
    upsample_factor : int
        Image-plane oversampling before average-pooling.
    name : str, optional
        caskade module name.
    """
    def __init__(
        self,
        lens,
        source_cube,
        pixelscale_source,
        pixelscale_lens,
        pixels_x_source,
        pixels_x_lens,
        upsample_factor,
        name: str = "sim",
    ):
        super().__init__(name)

        self.lens = lens
        self.source_cube = source_cube
        self.device = source_cube.device
        self.dtype = source_cube.dtype
        self.upsample_factor = upsample_factor
        self.src = Pixelated(name="source", shape=(pixels_x_source, pixels_x_source), pixelscale=pixelscale_source, image = torch.zeros((pixels_x_source, pixels_x_source)))

        # Create the high-resolution grid
        thx, thy = caustics.utils.meshgrid(
            pixelscale_lens / upsample_factor,
            upsample_factor * pixels_x_lens,
            device = source_cube.device, dtype = source_cube.dtype
        )

        self.thx = thx
        self.thy = thy

    @forward
    def forward(self, lens_source = True):
        """Render the lensed cube.

        Parameters
        ----------
        lens_source : bool, optional
            If False, skip the deflection and simply resample the source
            cube on the image-plane grid (useful to compare lensed vs
            unlensed appearance).

        Returns
        -------
        Tensor, shape (N_chan, pixels_x_lens, pixels_x_lens)
            The lensed (or resampled) cube.
        """
        cube = self.source_cube.forward()
        bx, by = self.lens.raytrace(self.thx, self.thy)

        def lens_channel(image):
            if lens_source:
                return self.src.brightness(bx, by, image = image)
            else:
                return self.src.brightness(self.thx, self.thy, image = image)
        
        # Ray-trace to get the lensed positions
        lensed_cube = vmap(lens_channel)(cube)
        del cube

        # Downsample to the desired resolution
        lensed_cube = avg_pool2d(lensed_cube[:, None], self.upsample_factor)[:, 0]
        torch.cuda.empty_cache()
        return lensed_cube


# ────────────────────────────────────────────────────────────────────────────
# Inverse-mapped analytic renderer (no CloudCatalog, no Pixelated source)
# ────────────────────────────────────────────────────────────────────────────

def gaussian_quantile_offsets(sigma, K, *, device, dtype):
    """Deterministic mid-quantile offsets for ``N(0, sigma^2)``.

    Same as
    :func:`supermage.simulators.analytic_cube.gaussian_quantile_offsets_flex`:
    splits a Gaussian line profile into ``K`` equal-probability sub-channels
    via the inverse CDF at mid-quantiles.

    Parameters
    ----------
    sigma : Tensor
        Dispersion [km/s]; scalar or per-pixel ``(H, W)``.
    K : int
        Number of quantile sub-channels.
    device, dtype :
        Placement of the quantile constants.

    Returns
    -------
    Tensor
        ``(K, 1, 1)`` for scalar ``sigma``, ``(K, H, W)`` for per-pixel.
    """
    p_mid = (torch.arange(K, device=device, dtype=dtype) + 0.5) / K
    unit = math.sqrt(2.0) * torch.erfinv(2.0 * p_mid - 1.0)    # (K,)
    if sigma.ndim == 0:
        return (sigma * unit).view(K, 1, 1)
    else:
        return unit.view(K, 1, 1) * sigma.view(1, *sigma.shape)
        

class AnalyticLens(Module):
    """Inverse-mapped analytic renderer for a lensed rotating disk.

    The lensed analogue of
    :class:`supermage.simulators.analytic_cube.AnalyticInverse`:

    1. build an oversampled image-plane grid ``(thx, thy)`` [arcsec],
    2. raytrace through the caustics lens to source-plane ``beta(theta)``,
    3. subtract source offsets and invert the disk projection to intrinsic
       ``(x_gal, y_gal)`` [pc],
    4. evaluate the analytic intensity/velocity fields at those positions,
    5. spread each image-plane pixel over ``K_vel`` deterministic Gaussian
       quantile sub-channels, and
    6. box-filter the hi-res cube down to ``(Nv, N_pix, N_pix)``.

    Because the source is evaluated analytically at the raytraced positions
    (no ``Pixelated`` interpolation), magnified regions are sampled at full
    intrinsic resolution.  Pixels raytraced to (numerically) infinite source
    radius near the lens centre are zeroed rather than allowed to poison the
    cube with NaNs (see the inline comment in :meth:`forward`).

    caskade Params: ``inclination`` [rad], ``sky_rot`` [rad],
    ``line_broadening`` [km/s], ``velocity_shift`` [km/s], ``x0``/``y0``
    [arcsec, source-plane offsets], ``distance_pc`` [pc], plus the lens's
    and sub-models' Params.

    Parameters
    ----------
    lens : caustics lens
        Lens exposing ``raytrace(thx, thy) -> (bx, by)`` [arcsec].
    intensity_model : Module
        Analytic brightness model, ``brightness(R) -> (H, W)``.
    velocity_model : Module
        Analytic velocity model, ``velocity(R) -> (H, W)``; its ``inc``
        Param is pointed at this module's ``inclination``.
    freq_axis : sequence, shape (Nv,)
        Uniform frequency axis; same units as ``line``.
    pixel_scale_arcsec : float
        Output image-plane pixel scale [arcsec / pixel].
    N_pix_x : int
        Output cube side [pixels].
    K_vel : int, optional
        Gaussian quantile sub-channels per pixel.
    oversamp_xy, oversamp_v : int, optional
        Spatial / velocity oversampling of the internal hi-res grid.
    chunk_v : int, optional
        If set, process the hi-res grid in spatial tiles (reduces peak
        memory; rarely needed).
    device, dtype :
        Torch placement.
    line : float
        Rest-frame line frequency, in the same units as ``freq_axis``
        (required; e.g. ``230.538e9`` for CO(2-1) with a Hz axis).
    name : str, optional
        caskade module name.
    """

    def __init__(
        self,
        lens,                      # caustics lens with .raytrace(θx, θy) -> (βx, βy) in arcsec
        intensity_model,           # analytic model: brightness(R) -> (H,W)
        velocity_model,            # analytic model: velocity(R)   -> (H,W)
        freq_axis,                 # (Nv,) uniform freqs for output cube
        pixel_scale_arcsec,        # arcsec / pixel on image plane
        N_pix_x,                   # output pixels side (square)
        *,
        K_vel: int = 8,            # number of quantile sub-channels per pixel
        oversamp_xy: int = 4,      # spatial oversampling for box-filtering
        oversamp_v : int = 4,      # velocity oversampling for box-filtering
        chunk_v: int | None = None,# optional: process velocity axis in chunks of this many hi-res planes
        device: str = "cuda",
        dtype : torch.dtype = torch.float32,
        line  : float,
        name  : str = "analytic_cloudless_lens_inverse",
    ):
        super().__init__(name)
        self.device, self.dtype = device, dtype
        self.lens = lens
        self.intensity_model = intensity_model
        self.velocity_model  = velocity_model

        # User-facing physical / geometric parameters (match CloudCatalog semantics)
        self.inclination     = Param("inclination",     None)     # [rad]
        self.velocity_model.inc = self.inclination
        self.sky_rot         = Param("sky_rot",         None)     # [rad]; position angle - 90°
        self.line_broadening = Param("line_broadening", None)     # [km/s] (or your velocity units)
        self.velocity_shift  = Param("velocity_shift",  None)     # [km/s]
        self.x0              = Param("x0",              None)     # [arcsec] source offset east (+)
        self.y0              = Param("y0",              None)     # [arcsec] source offset north (+)
        self.distance_pc     = Param("distance_pc",     None)     # [pc], for arcsec↔pc conversion

        # Velocity grid (low & high)
        from supermage.utils.doppler_velocities import create_velocity_grid_stable
        vel_axis, _ = create_velocity_grid_stable(
            f_start=freq_axis[0], f_end=freq_axis[-1],
            num_points=len(freq_axis), target_dtype=dtype, line=line
        )
        self.vel0_lo = vel_axis[0].to(dtype)
        self.dv_lo   = float((vel_axis[1] - vel_axis[0]).item())
        self.Nv_lo   = int(vel_axis.numel())

        self.oversamp_v  = int(oversamp_v)
        self.dv_hi   = self.dv_lo / self.oversamp_v
        delta        = 0.5 * (self.dv_lo - self.dv_hi)   # center-align
        self.vel0_hi = self.vel0_lo - delta
        self.Nv_hi   = self.Nv_lo * self.oversamp_v

        self.K_vel = int(K_vel)
        self.chunk_v = int(chunk_v) if (chunk_v is not None) else None

        # Spatial grids (image plane), high-res
        self.pixscale_lo = float(pixel_scale_arcsec)
        self.N_pix_lo    = int(N_pix_x)
        self.N_pix       = self.N_pix_lo
        self.fov_half_lo = 0.5 * (self.N_pix_lo - 1) * self.pixscale_lo

        self.oversamp_xy = int(oversamp_xy)
        self.pixscale_hi = self.pixscale_lo / self.oversamp_xy
        self.N_pix_hi    = self.N_pix_lo * self.oversamp_xy
        self.fov_half_hi = 0.5 * (self.N_pix_hi - 1) * self.pixscale_hi

        # Build θ-grid (arcsec), centered
        xs = (-self.fov_half_hi) + self.pixscale_hi * torch.arange(
            self.N_pix_hi, device=device, dtype=dtype
        )
        ys = (-self.fov_half_hi) + self.pixscale_hi * torch.arange(
            self.N_pix_hi, device=device, dtype=dtype
        )
        self.thx = xs.view(1, -1).expand(self.N_pix_hi, -1)  # (H,W)
        self.thy = ys.view(-1, 1).expand(-1, self.N_pix_hi)  # (H,W)

        # Precompute spatial indices for velocity-only scattering
        yy = torch.arange(self.N_pix_hi, device=device)
        xx = torch.arange(self.N_pix_hi, device=device)
        Y, X = torch.meshgrid(yy, xx, indexing="ij")
        self.Y_flat = Y.reshape(1, -1)   # (1, H*W)
        self.X_flat = X.reshape(1, -1)   # (1, H*W)
        self.hw     = int(self.N_pix_hi * self.N_pix_hi)

    # Inverse of your sky-projection (undo rotation + foreshortening)
    def _beta_to_intrinsic(self, beta_x, beta_y, *, x0, y0, pa, cos_i, arcsec_per_pc):
        """Source-plane coordinates -> intrinsic disk coordinates.

        Given ``beta_x, beta_y`` [arcsec], subtract offsets, convert to pc,
        and invert the disk projection::

           x_sky =  cos(pa) x_gal - sin(pa) (y_gal cos i)
           y_sky =  sin(pa) x_gal + cos(pa) (y_gal cos i)
           =>
           x_gal =  cos(pa) X + sin(pa) Y
           y_gal = (-sin(pa) X + cos(pa) Y) / cos i

        Returns
        -------
        (Tensor, Tensor, Tensor)
            ``(x_gal, y_gal, R)`` [pc], shaped like the input maps.
        """
        bx = beta_x - x0
        by = beta_y - y0
        X = bx / arcsec_per_pc  # pc
        Y = by / arcsec_per_pc  # pc

        cos_pa, sin_pa = torch.cos(pa), torch.sin(pa)
        x_gal =  cos_pa * X + sin_pa * Y
        y_gal = (-sin_pa * X + cos_pa * Y) / (cos_i + 1e-12)
        R = torch.hypot(x_gal, y_gal)
        return x_gal, y_gal, R

    # 1D linear binning along velocity axis for K quantiles per pixel
    def _bin_quantiles_along_v_(self, cube_hi, v_los, I_map, sigma):
        """Scatter K equal-flux quantile sub-channels per pixel into ``cube_hi``.

        Parameters
        ----------
        cube_hi : Tensor, shape (V, H, W)
            Pre-zeroed hi-res output cube.
        v_los : Tensor, shape (H, W)
            Line-of-sight velocity map [km/s].
        I_map : Tensor, shape (H, W)
            Per-pixel intensity (split equally over the K sub-channels).
        sigma : Tensor
            Line broadening [km/s]; scalar or per-pixel ``(H, W)``.

        Returns
        -------
        Tensor, shape (V, H, W)
            ``cube_hi`` with the flux deposited (out-of-band flux dropped,
            see :func:`supermage.simulators.velocity_scatter.scatter_quantiles_along_v`).
        """
        K = self.K_vel
        Δv = gaussian_quantile_offsets(sigma.abs() + 1e-12, K, device=self.device, dtype=self.dtype)  # (K,1,1) or (K,H,W)

        v_sub = v_los.view(1, *v_los.shape) + Δv              # (K,H,W)
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
        """Render the lensed spectral cube (caskade forward).

        Parameters
        ----------
        inclination, sky_rot, line_broadening, velocity_shift, x0, y0, distance_pc : optional
            caskade Params (units in the class docstring).
        return_intermediates : bool, optional
            If True, also return ``{"I_map": ..., "v_los": ...}`` on the
            hi-res image-plane grid.

        Returns
        -------
        Tensor, shape (Nv_lo, N_pix_lo, N_pix_lo)
            The lensed cube (relative flux units), or
            ``(cube, intermediates)`` when ``return_intermediates=True``.
        """
        # Aliases
        cos_i = torch.cos(inclination)
        pa    = sky_rot + math.pi / 2.0
        arcsec_per_pc = 206265.0 / distance_pc

        # θ → β(θ) in arcsec
        bx, by = self.lens.raytrace(self.thx, self.thy)   # (H,W)

        # β → intrinsic coords and R
        x_gal, y_gal, R = self._beta_to_intrinsic(
            bx, by, x0=x0, y0=y0, pa=pa, cos_i=cos_i, arcsec_per_pc=arcsec_per_pc
        )

        # Analytic fields
        I_map  = self.intensity_model.brightness(R)               # (H,W)
        v_circ = self.velocity_model.velocity(R)                  # (H,W)
        cos_theta = x_gal / (R + 1e-12)
        v_los = v_circ * torch.sin(inclination) * cos_theta + velocity_shift

        # Lens-center singularity guard: image-plane pixels within ~a hi-res pixel of
        # the lens center get a (near-)divergent EPL deflection -> beta overflows in
        # float32 -> R ~ inf, cos_theta = inf/inf = NaN -> v_los NaN, which would poison
        # the WHOLE cube through the flux normalization (a single bad voxel rejects an
        # otherwise-valid sample). Those pixels map to effectively infinite source
        # radius, so their physical surface brightness is zero: zero both fields there.
        # A genuinely broken draw (non-finite everywhere) still yields an all-zero cube
        # -> non-finite flux normalization -> the sampler's invalid-logL sentinel.
        _bad = ~(torch.isfinite(I_map) & torch.isfinite(v_los))
        I_map = torch.where(_bad, torch.zeros_like(I_map), I_map)
        v_los = torch.where(_bad, torch.zeros_like(v_los), v_los)

        # Allocate hi-res cube
        cube_hi = torch.zeros(self.Nv_hi, self.N_pix_hi, self.N_pix_hi,
                              device=self.device, dtype=self.dtype)

        # Quantile broadening along v
        if self.chunk_v is None:
            # Single pass: bin all K quantiles
            cube_hi = self._bin_quantiles_along_v_(cube_hi, v_los, I_map, line_broadening)
        else:
            # Optional: process spatial tiles to reduce peak memory (rarely needed)
            # Here, we chunk *velocity planes* post-binning would not help (binning is 1D).
            # Instead we tile spatial dims.
            tile = int(max(64, self.N_pix_hi // 4))  # heuristic tile size
            for y0i in range(0, self.N_pix_hi, tile):
                y1i = min(self.N_pix_hi, y0i + tile)
                for x0i in range(0, self.N_pix_hi, tile):
                    x1i = min(self.N_pix_hi, x0i + tile)
                    cube_hi[:, y0i:y1i, x0i:x1i] = self._bin_quantiles_along_v_(
                        cube_hi[:, y0i:y1i, x0i:x1i].clone(),
                        v_los[y0i:y1i, x0i:x1i],
                        I_map[y0i:y1i, x0i:x1i],
                        line_broadening if line_broadening.ndim == 0
                        else line_broadening[y0i:y1i, x0i:x1i]
                    )

        # Box-filter to low-res
        cube_hi = cube_hi.view(
            self.Nv_lo,  self.oversamp_v,
            self.N_pix_lo, self.oversamp_xy,
            self.N_pix_lo, self.oversamp_xy
        )
        cube_lo = cube_hi.mean((1, 3, 5))

        if return_intermediates:
            return cube_lo, {"I_map": I_map, "v_los": v_los}
        return cube_lo
