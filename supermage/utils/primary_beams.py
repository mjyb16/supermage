import numpy as np
import torch
from astropy import constants as const

def gaussian_pb(diameter=12, freq=432058061289.4426, shape=(500, 500), deltal=0.004, device='cpu', dtype = torch.float64):
    c = const.c.value # Speed of light in m/s
    wavelength = c / freq
    fwhm = 1.02 * wavelength / diameter * (180 / torch.pi) * (3600)
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