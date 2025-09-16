import torch
import math
from caskade import Module, forward, Param
from torch import vmap
import caustics
from caustics.light import Pixelated
from torch.nn.functional import avg_pool2d, conv2d


class CubeLens(Module):
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
    """
    Deterministic mid-quantile offsets for N(0, sigma^2).
    Works with scalar or per-pixel sigma:
      - if sigma is scalar: returns (K,1,1)
      - if sigma is (H,W): returns (K,H,W)
    """
    p_mid = (torch.arange(K, device=device, dtype=dtype) + 0.5) / K
    unit = math.sqrt(2.0) * torch.erfinv(2.0 * p_mid - 1.0)    # (K,)
    if sigma.ndim == 0:
        return (sigma * unit).view(K, 1, 1)
    else:
        return unit.view(K, 1, 1) * sigma.view(1, *sigma.shape)
        

class AnalyticLens(Module):
    """
    Lensing renderer that:
      - builds an image-plane grid (θx, θy),
      - raytraces to source plane β(θ),
      - inverts the sky projection to intrinsic (x_gal, y_gal),
      - evaluates analytic intensity/velocity fields,
      - applies deterministic Gaussian-quantile broadening in velocity,
      - bins into a hi-res (V,H,W) cube and box-filters to low-res.
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
        line  : str = "co21",
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
        from supermage.utils.cube_tools import create_velocity_grid_stable
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
        """
        Given βx, βy [arcsec], subtract offsets, convert to pc, and invert:
           x_sky =  cos(pa) x_gal - sin(pa) (y_gal cos i)
           y_sky =  sin(pa) x_gal + cos(pa) (y_gal cos i)
        Inverse:
           x_gal =  cos(pa) X + sin(pa) Y
           y_gal = (-sin(pa) X + cos(pa) Y) / cos i
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
        """
        cube_hi: (V,H,W) output (pre-zeroed)
        v_los : (H,W)
        I_map : (H,W)
        sigma : scalar or (H,W)
        Places K_vel equal-flux subchannels at v_los + Δv_k and bins to {iv0, iv1}.
        """
        K = self.K_vel
        Δv = gaussian_quantile_offsets(sigma.abs() + 1e-12, K, device=self.device, dtype=self.dtype)  # (K,1,1) or (K,H,W)

        v_sub = v_los.view(1, *v_los.shape) + Δv              # (K,H,W)
        iv_f  = (v_sub - self.vel0_hi) / self.dv_hi           # (K,H,W)
        iv0   = torch.floor(iv_f).to(torch.long).clamp(0, self.Nv_hi - 1)
        iv1   = (iv0 + 1).clamp(0, self.Nv_hi - 1)
        fv    = (iv_f - iv0.to(iv_f.dtype)).clamp(0, 1)       # (K,H,W)

        # Equal split across quantiles
        fsub = (I_map / float(K)).view(1, -1)                 # (1, H*W)
        # Flatten spatial dims once
        iv0 = iv0.view(K, -1)                                 # (K, H*W)
        iv1 = iv1.view(K, -1)
        w0  = (1 - fv).view(K, -1)
        w1  = fv.view(K, -1)

        # Build flat indices for (v,y,x)
        baseY = self.Y_flat.expand(K, -1)                     # (K, H*W)
        baseX = self.X_flat.expand(K, -1)
        stride_xy = self.hw
        idx0 = iv0 * stride_xy + baseY * self.N_pix_hi + baseX
        idx1 = iv1 * stride_xy + baseY * self.N_pix_hi + baseX

        flat = cube_hi.view(-1)
        flat.scatter_add_(0, idx0.reshape(-1), (fsub * w0).reshape(-1))
        flat.scatter_add_(0, idx1.reshape(-1), (fsub * w1).reshape(-1))

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
        """
        Returns
        -------
        cube_lo : Tensor  (Nv_lo, N_pix_lo, N_pix_lo)
        Optionally returns intermediates (I_map, v_los) if return_intermediates=True.
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

        # Allocate hi-res cube
        cube_hi = torch.zeros(self.Nv_hi, self.N_pix_hi, self.N_pix_hi,
                              device=self.device, dtype=self.dtype)

        # Quantile broadening along v
        if self.chunk_v is None:
            # Single pass: bin all K quantiles
            self._bin_quantiles_along_v_(cube_hi, v_los, I_map, line_broadening)
        else:
            # Optional: process spatial tiles to reduce peak memory (rarely needed)
            # Here, we chunk *velocity planes* post-binning would not help (binning is 1D).
            # Instead we tile spatial dims.
            tile = int(max(64, self.N_pix_hi // 4))  # heuristic tile size
            for y0i in range(0, self.N_pix_hi, tile):
                y1i = min(self.N_pix_hi, y0i + tile)
                for x0i in range(0, self.N_pix_hi, tile):
                    x1i = min(self.N_pix_hi, x0i + tile)
                    self._bin_quantiles_along_v_(
                        cube_hi[:, y0i:y1i, x0i:x1i],
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