import torch
from astropy import constants as const

def create_velocity_grid_stable(
    f_start: float,
    f_end: float,
    num_points: int,
    target_dtype = torch.float32,
    device = "cpu",
    line = "co21"
):
    """
    Creates a velocity grid using a numerically stable approach.

    This method works by recognizing the frequency-to-velocity conversion is a
    linear transformation (v = A*f + B). It calculates the start velocity and
    the velocity step size using high-precision float64, then constructs the
    final grid using the target dtype (e.g., float32). This avoids all
    cumulative precision errors.

    Returns:
        A tuple containing (final velocity grid, velocity steps).
    """
    # --- Step 1: Define grid parameters in HIGH PRECISION (float64) ---
    f_start_64 = torch.tensor(f_start, dtype=torch.float64)
    f_end_64 = torch.tensor(f_end, dtype=torch.float64)
    df_64 = (f_end_64 - f_start_64) / (num_points - 1)

    # --- Step 2: Calculate v_start and delta_v in HIGH PRECISION ---
    # The transformation is v(f) = A*f + B, so a uniform freq grid (f_i = f_start + i*df)
    # becomes a uniform velocity grid (v_i = v_start + i*delta_v).
    
    # Calculate v_start = v(f_start_64)
    v_start_64 = freq_to_vel_absolute(f_start_64, line = line)
    
    # Calculate delta_v = v(f_start_64 + df_64) - v(f_start_64)
    v_after_step_64 = freq_to_vel_absolute(f_start_64 + df_64, line = line)
    delta_v_64 = v_after_step_64 - v_start_64
    
    # --- Step 3: Construct the final grid using the TARGET PRECISION (float32) ---
    # This operation is now numerically stable.
    v_start_final = v_start_64.to(target_dtype)
    delta_v_final = delta_v_64.to(target_dtype)
    indices = torch.arange(num_points, dtype=target_dtype)
    
    abs_velocities = v_start_final + indices * delta_v_final

    # --- Step 4: Step size calculation ---
    velocity_steps = abs_velocities[1:] - abs_velocities[:-1]

    return abs_velocities.to(device = device), velocity_steps.to(device = device)

def freq_to_vel_absolute(freq, line, dtype = torch.float64):
    """
    Converts frequency (GHz) to absolute velocity (km/s) using the radio convention.
    """
    # Use high precision for constants 
    c_kms = torch.tensor(const.c.value / 1e3, dtype=dtype, device=freq.device)
    if line == "co21":
        co21_rest_freq_ghz = torch.tensor(230.538, dtype=dtype, device=freq.device)
    
        velocities = c_kms * (co21_rest_freq_ghz - freq) / co21_rest_freq_ghz
    else:
        print("ERROR: Line not implemented")
    return velocities