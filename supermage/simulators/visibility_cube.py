import torch, math, torch.nn.functional as F
from caskade import Module, forward, Param
import numpy as np
from supermage.utils.primary_beams import gaussian_pb
from supermage.utils.doppler_velocities import create_velocity_grid_stable

class VisibilityCubePadded(Module):
    """
    Identical public API except that padding is done *after* PB-weighting.

    Optional feature:
      - accepts an external image-plane apodization map (e.g. generated in VisCube)
      - applies its stabilized inverse to the padded model cube before the FFT
        so the UV-space forward model matches gridded data conventions.
    """
    def __init__(
        self,
        cube_simulator,
        mask,
        freqs,
        npix,                 # final grid side
        pixelscale,           # ″ / pix on the final grid
        dish_diameter: float = 12.0,
        line = 230.538,
        apodization_map = None,
        deapod_eps_fraction = 1e-3,
        deapod_clamp_max = 1e3,
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

        self.deapod_eps_fraction = float(deapod_eps_fraction)
        self.deapod_clamp_max    = deapod_clamp_max

        # ── size of the *small* cube returned by cube_simulator ─────────
        self.small_side = cube_simulator.N_pix
        if self.small_side > self.npix:
            raise ValueError(
                f"cube_simulator.N_pix ({self.small_side}) > npix ({self.npix})."
            )
        if (self.small_side % 2) != (self.npix % 2):
            raise ValueError(
                "Parity mismatch between cubes! Make shapes the same parity"
            )

        # symmetric padding widths: (left,right,top,bottom)
        pad_tot  = self.npix - self.small_side
        self.pad = (pad_tot // 2, pad_tot - pad_tot // 2) * 2  # (L,R,T,B)

        # ── primary beams ON THE SMALL GRID ─────────────────────────────
        #   (no need to generate values we’ll pad with zeros later)
        self.pb_small = torch.stack(
            [
                gaussian_pb(
                    diameter=self.dish_diameter,
                    freq=f,
                    shape=(self.small_side, self.small_side),
                    deltal=self.pixelscale,
                    device=self.device,
                    dtype=self.dtype,
                )[0]                                # gaussian_pb returns (pb, _)
                for f in freqs
            ],
            dim=0,                                  # (N_chan, S, S)
        )

        vel_axis, dv = create_velocity_grid_stable(
            f_start=freqs[0],
            f_end=freqs[-1],
            num_points=len(freqs),
            target_dtype=self.dtype,
            device=self.device,
            line=line,
        )
        self.dv = dv[0]

        # ── optional external apodization map (defined on FINAL FFT grid) ───────
        self.apodization_map     = None
        self.inv_apodization_map = None

        if apodization_map is not None:
            apo = torch.as_tensor(apodization_map, device=self.device, dtype=self.dtype)

            if apo.shape != (self.npix, self.npix):
                raise ValueError(
                    "apodization_map must have shape "
                    f"({self.npix}, {self.npix}), got {tuple(apo.shape)}."
                )

            self.apodization_map = apo

            thresh  = self.deapod_eps_fraction * torch.max(torch.abs(apo))
            inv_map = torch.where(
                torch.abs(apo) >= thresh,
                1.0 / apo,
                torch.zeros_like(apo),
            )

            if self.deapod_clamp_max is not None:
                inv_map = torch.clamp(
                    inv_map,
                    min=-float(self.deapod_clamp_max),
                    max= float(self.deapod_clamp_max),
                )

            self.inv_apodization_map = inv_map

    # ---------------------------------------------------------------------
    @forward
    def forward(self, flux=None):
        """
        Returns the padded FFT cube in UV space, masked by self.mask.
        """
        cube_small = self.cube_simulator.forward()          # (N_chan, S, S)

        # 1. multiply by primary beam on the small grid
        cube_pb = cube_small * self.pb_small                # (N_chan, S, S)
        del cube_small

        # 2. scale to requested total flux *before* padding
        if flux is None:
            flux = self.flux if self.flux is not None else 1.0
        cube_pb = cube_pb * flux / torch.abs(self.dv) / cube_pb.sum()  # Integrated flux in Jy km/s

        # 3. pad *both* spatial axes to (npix, npix)
        cube_pad = F.pad(cube_pb, self.pad, mode="constant", value=0.0)
        del cube_pb

        # 4. optional de-apodization before FFT
        if self.inv_apodization_map is not None:
            cube_pad = cube_pad * self.inv_apodization_map[None, :, :]

        # 5. FFT (channel-wise 2D)
        fft_cube = torch.fft.fftshift(
            torch.fft.fft2(
                torch.fft.ifftshift(cube_pad, dim=(-2, -1)),
                norm="backward",
            ),
            dim=(-2, -1),
        )
        del cube_pad                                        # memory

        return fft_cube * self.mask.float()                 # (N_chan, npix, npix)