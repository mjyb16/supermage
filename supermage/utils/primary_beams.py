"""Primary-beam models (antenna response across the field of view)."""
import numpy as np
import torch
from astropy import constants as const

def gaussian_pb(diameter=12, freq_hz=432058061289.4426, shape=(500, 500), deltal=0.004,
                fwhm_factor=1.13, device='cpu', dtype=torch.float64):
    """Gaussian primary beam on a square image grid, peak-normalized.

    ``freq_hz`` MUST be in Hz (the parameter was renamed from ``freq`` so
    stale callers passing GHz break loudly: a GHz value made the wavelength
    ~1e9x too large and the PB silently flat, pb = 1).

    Parameters
    ----------
    diameter : float, optional
        Antenna diameter [m].
    freq_hz : float, optional
        Observing frequency [Hz]; validated to be in ``(1e8, 1e13)``.
    shape : (int, int), optional
        Image grid shape (pixels).
    deltal : float, optional
        Pixel scale [arcsec / pixel].
    fwhm_factor : float, optional
        ``FWHM = fwhm_factor * lambda / D``.  1.13 is the ALMA Technical
        Handbook effective (tapered-illumination) 12-m beam; 1.02 is uniform
        illumination (the legacy value).
    device, dtype : optional
        Torch placement of the returned beam.

    Returns
    -------
    pb : Tensor, shape ``shape``
        The primary beam, normalized to a peak of 1.
    fwhm : float
        The beam FWHM [arcsec].

    Raises
    ------
    ValueError
        If ``freq_hz`` does not look like a Hz value.
    """
    f = float(freq_hz)
    if not (1e8 < f < 1e13):
        raise ValueError(
            f"gaussian_pb expects freq_hz in Hz (1e8 < f < 1e13); got {freq_hz!r}. "
            "A value of a few hundred suggests GHz — multiply by 1e9."
        )
    c = const.c.value # Speed of light in m/s
    wavelength = c / f
    fwhm = fwhm_factor * wavelength / diameter * (180 / torch.pi) * (3600)
    half_fov = deltal * shape[0] / 2

    # Grid for PB
    x = torch.linspace(-half_fov, half_fov, shape[0], device=device, dtype = dtype)
    y = torch.linspace(-half_fov, half_fov, shape[1], device=device, dtype = dtype)
    x, y = torch.meshgrid(x, y, indexing='xy')

    # just the exponent part 
    std = fwhm / (2 * torch.sqrt(2 * torch.log(torch.tensor(2.0, device=device))))
    r2 = x**2 + y**2
    pb  = torch.exp(-0.5 * r2 / std**2)

    return pb/pb.max(), fwhm

def casa_airy_beam(l,m,freq_chan,dish_diameter, blockage_diameter, ipower, max_rad_1GHz, n_sample=10000, device = "cpu"):
    """
    Airy disk function for the primary beam as implemented by CASA
    Credits: Noé Dia et al.

    .. warning::
        Legacy/unmaintained: this function currently references a Bessel
        function ``j1`` that is not imported (and ``casa_twiddle`` calls
        ``.value`` on a plain float), so it raises if called as-is. Kept for
        reference; use :func:`gaussian_pb` in the forward models.

    Parameters
    ----------
    l: float, radians
        Coordinate of a point on the image plane (the synthesis projected ascension and declination).
    m: float, radians
        Coordinate of a point on the image plane (the synthesis projected ascension and declination).
    freq_chan: float, Hz
        Frequency.
    dish_diameter: float, meters
        The diameter of the dish.
    blockage_diameter: float, meters
        The central blockage of the dish.
    ipower: int
        ipower = 1 single dish response.
        ipower = 2 baseline response for identical dishes.
    max_rad_1GHz: float, radians
        The max radius from which to sample scaled to 1 GHz.
        This value can be found in sirius_data.dish_models_1d.airy_disk.
        For example the Alma dish model (sirius_data.dish_models_1d.airy_disk import alma)
        is alma = {'func': 'airy', 'dish_diam': 10.7, 'blockage_diam': 0.75, 'max_rad_1GHz': 0.03113667385557884}.
    n_sample=10000
        The sampling used in CASA for PB math.
    Returns
    -------
    val : float
        The dish response.
    """
    c = const.c.value
    casa_twiddle = (180*7.016*c.value)/((np.pi**2)*(10**9)*1.566*24.5) # 0.9998277835716939

    r_max = max_rad_1GHz/(freq_chan/10**9)
    # print(r_max)
    k = (2*np.pi*freq_chan)/c
    aperture = dish_diameter/2

    if n_sample is not None:
        r = np.sqrt(l**2 + m**2)
        r_inc = ((r_max)/(n_sample-1))
        r = (int(r/r_inc)*r_inc)*aperture*k #Int rounding instead of r = (int(np.floor(r/r_inc + 0.5))*r_inc)*aperture*k
        r = r*casa_twiddle
    else:
        r = np.arcsin(np.sqrt(l**2 + m**2)*k*aperture)
        
    if (r != 0):
        if blockage_diameter==0.0:
            return torch.tensor((2.0*j1(r)/r)**ipower).to(device)
        else:
            area_ratio = (dish_diameter/blockage_diameter)**2
            length_ratio = (dish_diameter/blockage_diameter)
            return torch.tensor(((area_ratio * 2.0 * j1(r)/r   - 2.0 * j1(r * length_ratio)/(r * length_ratio) )/(area_ratio - 1.0))**ipower).to(device)
    else:
        return 1