import torch, math, torch.nn.functional as F
from caskade import Module, forward, Param
import numpy as np
from supermage.utils.primary_beams import gaussian_pb
from supermage.utils.doppler_velocities import create_velocity_grid_stable

class VisibilityCubePadded(Module):
    """
    Identical public API except that padding is done *after* PB‑weighting.
    """
    def __init__(
        self,
        cube_simulator,
        mask,
        freqs,
        npix,                 # final grid side
        pixelscale,           # ″ / pix on the final grid
        dish_diameter: float = 12.0,
        line = "co21"
    ):
        super().__init__()
        self.cube_simulator = cube_simulator
        self.mask           = mask
        self.freqs          = freqs
        self.npix           = npix
        self.pixelscale     = pixelscale
        self.dish_diameter  = dish_diameter

        self.device = cube_simulator.device
        self.dtype  = cube_simulator.dtype
        self.flux   = Param("flux", None)

        # ── size of the *small* cube returned by cube_simulator ─────────
        self.small_side = cube_simulator.N_pix      # uses the provided attribute
        if self.small_side > self.npix:
            raise ValueError(
                f"cube_simulator.N_pix ({self.small_side}) > npix ({self.npix})."
            )
        if (self.small_side%2) != (self.npix%2):
            raise ValueError(
                "Parity mismatch between cubes! Make shapes the same parity"
            )
        # symmetric padding widths: (left,right,top,bottom)
        pad_tot   = self.npix - self.small_side
        self.pad  = (pad_tot // 2, pad_tot - pad_tot // 2) * 2  # (L,R,T,B)

        # ── primary beams ON THE SMALL GRID ─────────────────────────────
        #   (no need to generate values we’ll pad with zeros later)
        self.pb_small = torch.stack(
            [
                gaussian_pb(
                    diameter=self.dish_diameter,
                    freq=f,
                    shape=(self.small_side, self.small_side),
                    deltal=self.pixelscale,          # same pixel scale
                    device=self.device,
                    dtype=self.dtype,
                )[0]                                # gaussian_pb returns (pb, _)
                for f in freqs
            ],
            dim=0,                                  # (N_chan, S, S)
        )
        
        vel_axis, dv = create_velocity_grid_stable(f_start = freqs[0], f_end = freqs[-1], num_points = len(freqs), target_dtype = self.dtype, line = line)

        self.dv = dv[0]

    # ---------------------------------------------------------------------
    @forward
    def forward(self, plot = True, flux = None):
        """
        If `plot` is True → full UV cube masked by `mask`.
        If `plot` is False → only values where mask == True.
        """
        cube_small = self.cube_simulator.forward()          # (N_chan, S, S)

        # 1. multiply by primary beam on the same small grid
        cube_pb = cube_small * self.pb_small                # (N_chan, S, S)
        del cube_small

        # 2. scale to requested total flux *before* padding
        if flux is None:
            flux = self.flux if self.flux is not None else 1.0
        cube_pb = cube_pb * flux / self.dv / cube_pb.sum() #Integrated flux in Jy km/s

        # 3. pad *both* spatial axes to (npix, npix)
        cube_pad = F.pad(cube_pb, self.pad, mode="constant", value=0.0)
        del cube_pb

        # 4. FFT  (channel‑wise 2‑D)
        fft_cube = torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(cube_pad, dim=(-2, -1)),
                           norm="backward"),
            dim=(-2, -1),
        )
        del cube_pad                                           # memory

        # 5. deliver either the full UV plane or only the sampled points
        if plot:
            return fft_cube * self.mask.float()                # (N_chan, npix, npix)

        def gather(fft_slc, mask_slc):                         # per‑channel
            return fft_slc[mask_slc]
        return torch.vmap(gather)(fft_cube, self.mask)         # (N_chan, N_points)