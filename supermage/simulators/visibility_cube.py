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


class JointVisibilityCube(Module):
    """Joint, multi-dataset drop-in replacement for ``VisibilityCubePadded``.

    Runs **one** shared ``cube_simulator`` (built at the *finest* dataset's image grid), then for each
    dataset:
      1. down-samples the shared cube to that dataset's image grid (flux/area-conserving),
      2. multiplies by that dataset's primary beam,
      3. scales to that dataset's own ``flux`` (a separate Param per dataset),
      4. pads to that dataset's uv-grid, de-apodizes, FFTs, and masks.

    ``forward()`` returns a list of per-dataset model visibility cubes (same shape/convention as
    ``VisibilityCubePadded.forward()`` would return for each dataset), so it plugs straight into a
    per-dataset Gaussian visibility likelihood that is summed across datasets.

    The shared source (image cube, velocity model, geometry, ...) is computed **once**; only the flux
    and uv-sampling/beam differ per dataset.  The module is independent of the underlying layers:
    ``cube_simulator`` may be any image-cube ``Module`` exposing ``.forward() -> (N_chan, S, S)``,
    ``.N_pix`` (int side of that cube), ``.device`` and ``.dtype``.

    Parameters
    ----------
    cube_simulator : Module
        Shared source cube simulator, built at the finest dataset's image grid.
    datasets : list[dict]
        One dict per dataset, each with keys:
          ``name`` (str), ``mask`` (bool array, (N_chan, npix, npix)), ``freqs`` (seq, len N_chan),
          ``npix`` (int, final uv-grid side), ``pixelscale`` (float, ''/pix on the final grid),
          ``N_px`` (int, image-cube side for this dataset; must be <= cube_simulator.N_pix),
          and optional ``apodization_map`` ((npix, npix)).
    dish_diameter, line, deapod_eps_fraction, deapod_clamp_max, downsample_mode :
        Same conventions as ``VisibilityCubePadded`` (``downsample_mode`` is the ``F.interpolate``
        mode used to degrade the shared cube to a coarser grid; ``'area'`` is flux/area conserving).
    """

    def __init__(
        self,
        cube_simulator,
        datasets,
        *,
        dish_diameter: float = 12.0,
        line: float = 230.538,
        deapod_eps_fraction: float = 1e-3,
        deapod_clamp_max: float = 1e3,
        downsample_mode: str = "area",
        name: str = "joint_visibility",
    ):
        super().__init__(name)
        self.cube_simulator = cube_simulator
        self.device = cube_simulator.device
        self.dtype = cube_simulator.dtype
        self.fine_side = int(cube_simulator.N_pix)
        self.deapod_eps_fraction = float(deapod_eps_fraction)
        self.deapod_clamp_max = deapod_clamp_max
        self.downsample_mode = downsample_mode

        self._heads = []           # per-dataset precomputed (non-trainable) buffers
        self.fluxes = []           # per-dataset flux Params (registered as children)

        for i, ds in enumerate(datasets):
            nm = ds.get("name", f"d{i}")
            S = int(ds["N_px"])            # this dataset's image (small-cube) side
            npix = int(ds["npix"])         # this dataset's final uv-grid side
            pixelscale = float(ds["pixelscale"])
            freqs = ds["freqs"]

            if S > self.fine_side:
                raise ValueError(
                    f"dataset '{nm}': N_px ({S}) > cube_simulator.N_pix ({self.fine_side}); "
                    "the shared cube must be the FINEST grid."
                )
            if (S % 2) != (npix % 2):
                raise ValueError(f"dataset '{nm}': parity mismatch S={S} npix={npix}.")

            pb = torch.stack(
                [gaussian_pb(diameter=dish_diameter, freq=f, shape=(S, S),
                             deltal=pixelscale, device=self.device, dtype=self.dtype)[0]
                 for f in freqs],
                dim=0,
            )
            _, dv = create_velocity_grid_stable(
                f_start=freqs[0], f_end=freqs[-1], num_points=len(freqs),
                target_dtype=self.dtype, device=self.device, line=line,
            )
            pad_tot = npix - S
            pad = (pad_tot // 2, pad_tot - pad_tot // 2) * 2     # (L, R, T, B)
            mask = torch.as_tensor(ds["mask"], device=self.device, dtype=torch.bool)

            inv_apod = None
            apo_in = ds.get("apodization_map", None)
            if apo_in is not None:
                apo = torch.as_tensor(apo_in, device=self.device, dtype=self.dtype)
                if apo.shape != (npix, npix):
                    raise ValueError(f"dataset '{nm}': apodization_map must be ({npix},{npix}).")
                thresh = self.deapod_eps_fraction * torch.max(torch.abs(apo))
                inv_apod = torch.where(torch.abs(apo) >= thresh, 1.0 / apo, torch.zeros_like(apo))
                if self.deapod_clamp_max is not None:
                    inv_apod = torch.clamp(inv_apod, -float(self.deapod_clamp_max),
                                           float(self.deapod_clamp_max))

            flux = Param(f"flux_{nm}", None)
            setattr(self, f"flux_{nm}", flux)                   # register as a child Param
            self.fluxes.append(flux)
            self._heads.append(dict(name=nm, S=S, npix=npix, pb=pb, dv=dv[0],
                                    pad=pad, mask=mask, inv_apod=inv_apod))

    # ------------------------------------------------------------------
    def _visibility_from_cube(self, cube, head, flux):
        """One dataset's cube -> masked UV visibilities (mirrors VisibilityCubePadded)."""
        cube_pb = cube * head["pb"]
        cube_pb = cube_pb * flux / torch.abs(head["dv"]) / cube_pb.sum()   # Jy km/s total
        cube_pad = F.pad(cube_pb, head["pad"], mode="constant", value=0.0)
        if head["inv_apod"] is not None:
            cube_pad = cube_pad * head["inv_apod"][None, :, :]
        fft = torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(cube_pad, dim=(-2, -1)), norm="backward"),
            dim=(-2, -1),
        )
        return fft * head["mask"].float()

    @forward
    def forward(self):
        """Returns a list of per-dataset masked UV cubes, in the order ``datasets`` was given."""
        cube_fine = self.cube_simulator.forward()              # (N_chan, fine, fine) -- ONCE
        outs = []
        for head, fluxp in zip(self._heads, self.fluxes):
            S = head["S"]
            if S == self.fine_side:
                cube = cube_fine
            else:                                              # degrade shared cube to this grid
                cube = F.interpolate(cube_fine.unsqueeze(0), size=(S, S),
                                     mode=self.downsample_mode).squeeze(0)
            flux = fluxp.value if fluxp.value is not None else 1.0
            outs.append(self._visibility_from_cube(cube, head, flux))
        return outs