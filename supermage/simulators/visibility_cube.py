"""Image-cube -> model-visibility simulators (uv-plane forward models).

Wraps a cube simulator (any Module whose ``forward()`` returns
``(N_chan, S, S)``) into the interferometric measurement chain:
primary-beam attenuation -> total-flux normalization -> zero-padding to the
uv grid -> image-plane taper multiplication -> centered FFT -> (optional
Hermitian half-plane slab) -> uv mask.  The output matches gridded
kernel-weighted-mean visibilities produced by ``viscube``.

Conventions (2026-07 corrected pipeline):

* all frequencies (``freqs``, ``line``) are in **Hz** and validated loudly;
* the image-plane taper (the FT of the gridding kernel) is **multiplied**
  into the padded image (the legacy divide-by-apodization convention is
  reproducible explicitly, see :class:`VisibilityCubePadded`);
* ``output_half_plane=True`` returns only the non-redundant Hermitian half
  of the uv plane, which avoids double-counting cells in the likelihood.

:class:`JointVisibilityCube` runs one shared source cube through several
per-dataset measurement chains (multi-configuration / multi-band fits).
"""
import torch, math, torch.nn.functional as F
from caskade import Module, forward, Param
import numpy as np
from supermage.utils.primary_beams import gaussian_pb
from supermage.utils.doppler_velocities import create_velocity_grid_stable


def _check_hz(x, name):
    """Frequencies are Hz everywhere in this module (ratio-based velocity math
    is unit-agnostic, but the primary beam is not — a GHz value here used to
    make the PB silently flat)."""
    v = float(x)
    if not (1e8 < v < 1e13):
        raise ValueError(
            f"{name} must be in Hz (1e8 < f < 1e13); got {v!r}. "
            "A value of a few hundred suggests GHz — multiply by 1e9."
        )


def _fft_visibilities(cube_pb, pad, taper_map, mask, half_plane, npix):
    """
    Shared image-cube -> masked model-visibility path:
      zero-pad -> optional MULTIPLY by the image-plane taper (the FT of the
      gridding kernel, so the model matches per-cell kernel-weighted-mean
      gridded data; the legacy code DIVIDED here, which is the inverted
      convention) -> centered FFT -> optional Hermitian half-plane slab
      (rows [npix//2:], valid for a real image on an odd grid) -> mask.
    """
    cube_pad = F.pad(cube_pb, pad, mode="constant", value=0.0)
    if taper_map is not None:
        cube_pad = cube_pad * taper_map[None, :, :]
    fft_cube = torch.fft.fftshift(
        torch.fft.fft2(
            torch.fft.ifftshift(cube_pad, dim=(-2, -1)),
            norm="backward",
        ),
        dim=(-2, -1),
    )
    if half_plane:
        fft_cube = fft_cube[..., npix // 2:, :]
    return fft_cube * mask.float()


def _expected_mask_shape(n_chan, npix, half_plane):
    """Shape a uv mask must have: full plane or the Hermitian half-plane slab."""
    if half_plane:
        return (n_chan, (npix + 1) // 2, npix)
    return (n_chan, npix, npix)


class VisibilityCubePadded(Module):
    """
    Image-cube simulator -> model visibilities on the (padded) FFT grid.

    Conventions (2026-07 corrected pipeline):
      - ``freqs`` and ``line`` are in Hz.
      - ``image_taper_map`` (npix, npix) is MULTIPLIED into the padded image
        before the FFT. Pass the analytic taper of the gridding kernel
        (e.g. ``viscube.make_kb_taper_map``) to match kernel-weighted-mean
        gridded data. The legacy divide-by-apodization behavior is
        reproducible by passing ``viscube.stabilized_inverse_map(apo)``.
      - ``dish_diameter=None`` disables the primary beam (pb ≡ 1), which
        reproduces the legacy silently-flat-PB behavior explicitly.
      - ``output_half_plane=True`` returns only the non-redundant Hermitian
        half of the uv plane: rows [npix//2:] of the fftshifted grid, shape
        (N_chan, (npix+1)//2, npix). Requires odd npix and a mask of that
        slab shape (with the DC row's duplicate columns already zeroed, see
        ``viscube.half_plane_mask_fix``).

    caskade Params: ``flux`` — the total integrated line flux [Jy km/s] the
    cube is normalized to — plus everything owned by ``cube_simulator``.

    Parameters
    ----------
    cube_simulator : Module
        Source cube simulator; ``forward()`` must return
        ``(N_chan, S, S)`` with ``S = cube_simulator.N_pix`` (same parity as
        ``npix``, ``S <= npix``).
    mask : bool array
        uv cells to keep, shape ``(N_chan, npix, npix)`` — or the half-plane
        slab shape when ``output_half_plane=True``.
    freqs : sequence, shape (N_chan,)
        Channel frequencies [Hz], ascending.
    npix : int
        Side of the final (padded) uv grid.
    pixelscale : float
        Image pixel scale on the final grid [arcsec / pixel].
    dish_diameter : float or None, optional
        Antenna diameter [m] for the Gaussian primary beam; None disables
        the PB.
    line : float, optional
        Rest-frame line frequency [Hz] (default CO(2-1)).
    image_taper_map : array, optional
        ``(npix, npix)`` image-plane taper, multiplied before the FFT.
    fwhm_factor : float, optional
        PB FWHM = ``fwhm_factor * lambda / D`` (1.13 = ALMA effective).
    output_half_plane : bool, optional
        Return only the non-redundant Hermitian slab (requires odd
        ``npix``).
    """
    def __init__(
        self,
        cube_simulator,
        mask,
        freqs,
        npix,                 # final grid side
        pixelscale,           # ″ / pix on the final grid
        dish_diameter=12.0,   # meters; None -> no primary beam (pb ≡ 1)
        line=230.538e9,       # Hz
        image_taper_map=None, # (npix, npix), multiplied before the FFT
        fwhm_factor=1.13,     # PB FWHM = fwhm_factor * lambda / D
        output_half_plane=False,
    ):
        super().__init__()
        self.cube_simulator = cube_simulator
        self.freqs          = freqs
        self.npix           = int(npix)
        self.pixelscale     = pixelscale
        self.dish_diameter  = dish_diameter
        self.output_half_plane = bool(output_half_plane)

        self.device = cube_simulator.device
        self.dtype  = cube_simulator.dtype
        self.flux   = Param("flux", None)

        _check_hz(freqs[0], "freqs[0]")
        _check_hz(freqs[-1], "freqs[-1]")
        _check_hz(line, "line")

        if self.output_half_plane and self.npix % 2 != 1:
            raise ValueError(
                f"output_half_plane=True requires odd npix, got {self.npix}."
            )

        mask_t = torch.as_tensor(mask, device=self.device)
        exp_shape = _expected_mask_shape(len(freqs), self.npix, self.output_half_plane)
        if tuple(mask_t.shape) != exp_shape:
            raise ValueError(
                f"mask shape {tuple(mask_t.shape)} does not match expected "
                f"{exp_shape} (output_half_plane={self.output_half_plane})."
            )
        self.mask = mask_t

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
        if dish_diameter is None:
            self.pb_small = None
        else:
            self.pb_small = torch.stack(
                [
                    gaussian_pb(
                        diameter=dish_diameter,
                        freq_hz=f,
                        shape=(self.small_side, self.small_side),
                        deltal=self.pixelscale,
                        fwhm_factor=fwhm_factor,
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

        # ── image-plane taper (MULTIPLIED before the FFT) ────────────────
        self.image_taper_map = None
        if image_taper_map is not None:
            taper = torch.as_tensor(image_taper_map, device=self.device, dtype=self.dtype)
            if taper.shape != (self.npix, self.npix):
                raise ValueError(
                    "image_taper_map must have shape "
                    f"({self.npix}, {self.npix}), got {tuple(taper.shape)}."
                )
            self.image_taper_map = taper

    # ---------------------------------------------------------------------
    @forward
    def forward(self, flux=None):
        """Simulate the masked model visibility cube (caskade forward).

        Runs the source cube simulator, applies the primary beam, rescales
        so the integrated line flux equals ``flux`` [Jy km/s], then pads,
        tapers, FFTs and masks.

        Parameters
        ----------
        flux : optional
            The total-flux Param [Jy km/s]; defaults to 1.0 if the Param is
            unset.

        Returns
        -------
        Tensor (complex)
            Masked model visibilities: ``(N_chan, npix, npix)``, or
            ``(N_chan, (npix+1)//2, npix)`` when ``output_half_plane=True``.
        """
        cube_small = self.cube_simulator.forward()          # (N_chan, S, S)

        # 1. multiply by primary beam on the small grid
        if self.pb_small is not None:
            cube_pb = cube_small * self.pb_small            # (N_chan, S, S)
        else:
            cube_pb = cube_small
        del cube_small

        # 2. scale to requested total flux *before* padding
        if flux is None:
            flux = self.flux if self.flux is not None else 1.0
        cube_pb = cube_pb * flux / torch.abs(self.dv) / cube_pb.sum()  # Integrated flux in Jy km/s

        # 3. pad, taper, FFT, (half-plane), mask
        return _fft_visibilities(
            cube_pb, self.pad, self.image_taper_map, self.mask,
            self.output_half_plane, self.npix,
        )


class JointVisibilityCube(Module):
    """Joint, multi-dataset drop-in replacement for ``VisibilityCubePadded``.

    Runs **one** shared ``cube_simulator`` (built at the *finest* dataset's image grid), then for each
    dataset:
      1. down-samples the shared cube to that dataset's image grid (flux/area-conserving),
      2. multiplies by that dataset's primary beam,
      3. scales to that dataset's own ``flux`` (a separate Param per dataset),
      4. pads to that dataset's uv-grid, applies that dataset's image-plane taper, FFTs, and masks.

    ``forward()`` returns a list of per-dataset model visibility cubes (same shape/convention as
    ``VisibilityCubePadded.forward()`` would return for each dataset), so it plugs straight into a
    per-dataset Gaussian visibility likelihood that is summed across datasets.

    Parameters
    ----------
    cube_simulator : Module
        Shared source cube simulator, built at the finest dataset's image grid.
    datasets : list[dict]
        One dict per dataset, each with keys:
          ``name`` (str), ``mask`` (bool array; (N_chan, npix, npix), or the half-plane slab shape
          (N_chan, (npix+1)//2, npix) when ``output_half_plane`` is set), ``freqs`` (seq, len N_chan,
          in Hz), ``npix`` (int, final uv-grid side), ``pixelscale`` (float, ''/pix on the final
          grid), ``N_px`` (int, image-cube side for this dataset; must be <= cube_simulator.N_pix),
          optional ``taper_map`` ((npix, npix), MULTIPLIED before the FFT) and optional
          ``output_half_plane`` (bool, default False).
    dish_diameter, line, fwhm_factor, downsample_mode :
        Same conventions as ``VisibilityCubePadded`` (``line`` in Hz; ``dish_diameter=None``
        disables the PB; ``downsample_mode`` is the ``F.interpolate`` mode used to degrade the
        shared cube to a coarser grid; ``'area'`` is flux/area conserving).
    """

    def __init__(
        self,
        cube_simulator,
        datasets,
        *,
        dish_diameter=12.0,
        line: float = 230.538e9,
        fwhm_factor: float = 1.13,
        downsample_mode: str = "area",
        name: str = "joint_visibility",
    ):
        super().__init__(name)
        self.cube_simulator = cube_simulator
        self.device = cube_simulator.device
        self.dtype = cube_simulator.dtype
        self.fine_side = int(cube_simulator.N_pix)
        self.downsample_mode = downsample_mode

        _check_hz(line, "line")

        self._heads = []           # per-dataset precomputed (non-trainable) buffers
        self.fluxes = []           # per-dataset flux Params (registered as children)

        for i, ds in enumerate(datasets):
            nm = ds.get("name", f"d{i}")
            S = int(ds["N_px"])            # this dataset's image (small-cube) side
            npix = int(ds["npix"])         # this dataset's final uv-grid side
            pixelscale = float(ds["pixelscale"])
            freqs = ds["freqs"]
            half_plane = bool(ds.get("output_half_plane", False))

            _check_hz(freqs[0], f"dataset '{nm}': freqs[0]")

            if S > self.fine_side:
                raise ValueError(
                    f"dataset '{nm}': N_px ({S}) > cube_simulator.N_pix ({self.fine_side}); "
                    "the shared cube must be the FINEST grid."
                )
            if (S % 2) != (npix % 2):
                raise ValueError(f"dataset '{nm}': parity mismatch S={S} npix={npix}.")
            if half_plane and npix % 2 != 1:
                raise ValueError(f"dataset '{nm}': output_half_plane requires odd npix.")

            if dish_diameter is None:
                pb = None
            else:
                pb = torch.stack(
                    [gaussian_pb(diameter=dish_diameter, freq_hz=f, shape=(S, S),
                                 deltal=pixelscale, fwhm_factor=fwhm_factor,
                                 device=self.device, dtype=self.dtype)[0]
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
            exp_shape = _expected_mask_shape(len(freqs), npix, half_plane)
            if tuple(mask.shape) != exp_shape:
                raise ValueError(
                    f"dataset '{nm}': mask shape {tuple(mask.shape)} does not match "
                    f"expected {exp_shape} (output_half_plane={half_plane})."
                )

            taper = None
            taper_in = ds.get("taper_map", None)
            if taper_in is not None:
                taper = torch.as_tensor(taper_in, device=self.device, dtype=self.dtype)
                if taper.shape != (npix, npix):
                    raise ValueError(f"dataset '{nm}': taper_map must be ({npix},{npix}).")

            flux = Param(f"flux_{nm}", None)
            setattr(self, f"flux_{nm}", flux)                   # register as a child Param
            self.fluxes.append(flux)
            self._heads.append(dict(name=nm, S=S, npix=npix, pb=pb, dv=dv[0],
                                    pad=pad, mask=mask, taper=taper,
                                    half_plane=half_plane))

    # ------------------------------------------------------------------
    def _visibility_from_cube(self, cube, head, flux):
        """One dataset's cube -> masked UV visibilities (mirrors VisibilityCubePadded)."""
        if head["pb"] is not None:
            cube_pb = cube * head["pb"]
        else:
            cube_pb = cube
        cube_pb = cube_pb * flux / torch.abs(head["dv"]) / cube_pb.sum()   # Jy km/s total
        return _fft_visibilities(
            cube_pb, head["pad"], head["taper"], head["mask"],
            head["half_plane"], head["npix"],
        )

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
