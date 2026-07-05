"""Frequency <-> radio-convention velocity conversion helpers.

The conversion ``v = c * (f_rest - f) / f_rest`` is linear in ``f`` and
ratio-based, so it is agnostic to the frequency *unit* as long as the
frequency axis and the rest-frame line frequency use the same one (the
visibility simulators standardize on Hz).
"""
import torch
from astropy import constants as const

def create_velocity_grid_stable(
    f_start: float,
    f_end: float,
    num_points: int,
    target_dtype = torch.float32,
    device = "cpu",
    line = 230.538
):
    """Create a uniform radio-velocity grid from a uniform frequency axis.

    Exploits the linearity of the frequency-to-velocity conversion
    (``v = A*f + B``): the start velocity and the step are computed in
    float64, then the grid is constructed as ``v0 + i*dv`` directly in the
    target dtype — avoiding the catastrophic cancellation of converting
    each (float32) frequency separately.

    Parameters
    ----------
    f_start, f_end : float
        First and last channel frequency (same units as ``line``).
    num_points : int
        Number of channels.
    target_dtype : torch.dtype, optional
        Dtype of the returned grid (the internal math is float64).
    device : optional
        Device of the returned tensors.
    line : float, optional
        Rest-frame line frequency, same units as ``f_start``/``f_end``
        (default 230.538, CO(2-1) in GHz).

    Returns
    -------
    abs_velocities : Tensor, shape (num_points,)
        Absolute radio-convention velocities [km/s] (subtract the systemic
        velocity for rest-frame values).
    velocity_steps : Tensor, shape (num_points - 1,)
        Channel widths [km/s] (negative when frequency increases).
    """
    # --- Step 1: Define grid parameters in HIGH PRECISION (float64) ---
    f_start_64 = torch.tensor(f_start, dtype=torch.float64)
    f_end_64 = torch.tensor(f_end, dtype=torch.float64)
    df_64 = (f_end_64 - f_start_64) / (num_points - 1)

    # --- Step 2: Calculate v_start and delta_v in HIGH PRECISION ---
    # The transformation is v(f) = A*f + B, so a uniform freq grid (f_i = f_start + i*df)
    # becomes a uniform velocity grid (v_i = v_start + i*delta_v).
    
    # Calculate v_start = v(f_start_64)
    v_start_64 = freq_to_vel_absolute(f_start_64, rest_frame_freq = line)
    
    # Calculate delta_v = v(f_start_64 + df_64) - v(f_start_64)
    v_after_step_64 = freq_to_vel_absolute(f_start_64 + df_64, rest_frame_freq = line)
    delta_v_64 = v_after_step_64 - v_start_64
    
    # --- Step 3: Construct the final grid using the TARGET PRECISION (float32) ---
    # This operation is now numerically stable.
    v_start_final = v_start_64.to(dtype=target_dtype, device=device)
    delta_v_final = delta_v_64.to(dtype=target_dtype, device=device)
    indices = torch.arange(num_points, dtype=target_dtype, device=device)
    
    abs_velocities = v_start_final + indices * delta_v_final

    # --- Step 4: Step size calculation ---
    velocity_steps = abs_velocities[1:] - abs_velocities[:-1]

    return abs_velocities.to(device = device), velocity_steps.to(device = device)

def freq_to_vel_absolute(freq, rest_frame_freq, dtype = torch.float64):
    """Convert frequency to absolute velocity [km/s], radio convention.

    ``v = c * (f_rest - f) / f_rest``.  ``freq`` and ``rest_frame_freq``
    must share units (Hz with Hz, or GHz with GHz).

    Parameters
    ----------
    freq : Tensor
        Frequencies to convert.
    rest_frame_freq : float
        Rest-frame line frequency (same units as ``freq``).
    dtype : torch.dtype, optional
        Precision of the constants (keep float64 for stability).

    Returns
    -------
    Tensor
        Velocities [km/s], same shape as ``freq``.
    """
    # Use high precision for constants 
    c_kms = torch.tensor(const.c.value / 1e3, dtype=dtype, device=freq.device)
    rest_freq_ghz = torch.tensor(rest_frame_freq, dtype=dtype, device=freq.device)
    velocities = c_kms * (rest_freq_ghz - freq) / rest_freq_ghz
    return velocities