import numpy as np
import torch
from scipy import signal
from scipy import ndimage
from scipy.ndimage import uniform_filter
from scipy.ndimage import rotate
from typing import Tuple, Union

def dirty_cube_tool(vis_bin_re_cube, vis_bin_imag_cube, roi_start, roi_end):
    # Define the region of interest for the cube (pixels 1000 to 1050)
    roi_start, roi_end = 225, 276
    num_frequencies = vis_bin_re_cube.shape[0]  # Total number of frequencies
    
    # Initialize an empty list to store the dirty images
    dirty_cube = []
    
    # Loop over each frequency slice to create the dirty image for each
    for i in range(num_frequencies):
        # Create the complex visibility data for the current frequency slice
        combined_vis = vis_bin_re_cube[i] + 1j * vis_bin_imag_cube[i]
        
        # Perform the inverse FFT to get the dirty image in the image plane
        dirty_image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(combined_vis), norm = "backward"))
        
        # Take the real part (intensity map) and restrict to the region of interest
        dirty_image_roi = np.abs(dirty_image)[roi_start:roi_end, roi_start:roi_end]
        
        # Append the region of interest for this frequency to the dirty cube
        dirty_cube.append(dirty_image_roi)
    
    # Stack all frequency slices to form a 3D array (dirty cube)
    dirty_cube = np.stack(dirty_cube, axis=-1)
    return dirty_cube

# Eric's mask making code
def smooth_mask(cube, sigma = 2, hann = 5, clip = 0.0002):  # updated by Eric
    """
    Apply a Gaussian blur, using sigma = 4 in the velocity direction (seems to work best), to the uncorrected cube.
    The mode 'nearest' seems to give the best results.
    :return: (ndarray) mask to apply to the un-clipped cube
    """
    smooth_cube = uniform_filter(cube, size=[sigma, sigma, 0], mode='constant')
    Hann_window=signal.windows.hann(hann)
    smooth_cube=signal.convolve(smooth_cube,Hann_window[np.newaxis,np.newaxis,:],mode="same")/np.sum(Hann_window)
    print("RMS of the smoothed cube in mJy/beam:",np.sqrt(np.nanmean(smooth_cube[0]**2))*1e3)
    mask=(smooth_cube > clip)
    mask_iter = mask.T # deliberately make them the same variable, convenient for updating

    print('final mask sum',np.sum(mask))
    return mask_iter.T

def rotate_spectral_cube_center_offset_arcsec(
    cube_in: np.ndarray,
    angle_deg: float,
    center_offset_arcsec: Tuple[float, float] = (0.0, 0.0),
    pixel_scale: float = 1.0,
    pad_mode: str = "constant",
    pad_cval: Union[int, float] = 0.0,
    interp_order: int = 3,
):
    """
    Rotate a spectral cube around a point specified as an *arcsecond offset*
    from the cube’s geometric centre.

    Parameters
    ----------
    cube : ndarray, shape (ny, nx, nchan)
        Spectral cube (channel-first).
    angle_deg : float
        Counter-clockwise rotation angle (degrees).
    center_offset_arcsec : (dx, dy)
        Offset from the cube centre in arcseconds:
            dx > 0 → right,  dy > 0 → up.
        Fractions allowed.
    pixel_scale : float
        Arcseconds per pixel (or any unit per pixel),
        used for the offset conversion *and* to report the new extent.
    pad_mode / pad_cval / interp_order
        Passed through to `np.y0.item()pad` and `scipy.ndimage.rotate`.

    Returns
    -------
    rotated_cube : ndarray
        Padded & rotated cube.
    extent : ((x_min, x_max), (y_min, y_max))
        Spatial extent in the same physical units as `pixel_scale`.
        The rotation point is at (0, 0).
    """
    ny, nx, n_chan = cube_in.shape

    # ------------------------------------------------------------------
    # 1. Convert arcsecond offset → pixel offset
    # ------------------------------------------------------------------
    dx_arcsec, dy_arcsec = center_offset_arcsec
    dx_pix = dx_arcsec / pixel_scale
    dy_pix = dy_arcsec / pixel_scale

    # geometric centre of the original image
    cx_orig = (nx - 1) / 2.0
    cy_orig = (ny - 1) / 2.0

    # absolute pixel coordinates of the rotation point
    x0 = cx_orig + dx_pix
    y0 = cy_orig + dy_pix

    # ------------------------------------------------------------------
    # 2. Pad so that (x0, y0) becomes the image centre
    # ------------------------------------------------------------------
    left   = x0
    right  = nx - 1 - x0
    top    = y0
    bottom = ny - 1 - y0

    half_width  = int(np.ceil(max(left, right )))
    half_height = int(np.ceil(max(top , bottom)))

    nx_pad = 2 * half_width  + 1
    ny_pad = 2 * half_height + 1

    pad_left   = half_width  - int(np.floor(left))
    pad_right  = half_width  - int(np.floor(right))
    pad_top    = half_height - int(np.floor(top))
    pad_bottom = half_height - int(np.floor(bottom))

    pad_width = (
        (0, 0),                      # spectral axis
        (pad_top, pad_bottom),       # y
        (pad_left, pad_right),       # x
    )

    cube = np.moveaxis(cube_in, -1, 0)

    cube_padded = np.pad(
        cube, pad_width, mode=pad_mode, constant_values=pad_cval
    )

    # ------------------------------------------------------------------
    # 3. Rotate every channel about the new centre
    # ------------------------------------------------------------------
    rotated_cube = np.empty_like(cube_padded)
    for k in range(n_chan):
        rotated_cube[k] = ndimage.rotate(
            cube_padded[k],
            angle_deg,
            reshape=False,
            order=interp_order,
            mode=pad_mode,
            cval=pad_cval,
        )

    # ------------------------------------------------------------------
    # 4. Compute physical extent
    # ------------------------------------------------------------------
    cx_new = (nx_pad - 1) / 2.0
    cy_new = (ny_pad - 1) / 2.0
    x_min = -(cx_new) * pixel_scale
    x_max = +(nx_pad - 1 - cx_new) * pixel_scale
    y_min = -(cy_new) * pixel_scale
    y_max = +(ny_pad - 1 - cy_new) * pixel_scale
    dx = (x_max - x_min)/rotated_cube.shape[2] 
    dy = (y_max - y_min)/rotated_cube.shape[1]
    extent = (x_min - dx/2, x_max+dx/2, y_min - dy/2, y_max + dy/2)
    #extent = (x_min, x_max, y_min, y_max)
    
    return np.moveaxis(rotated_cube, 0, -1), extent

def velocity_map(cube, velocities, backend = "numpy"):      
    # Calculate intensity-weighted average velocity
    if backend == "numpy":
        vel_map = np.sum(cube * velocities[None, None, :], axis=2) / np.sum(cube, axis=2)
        
    elif backend == "pytorch":
        vel_map = torch.sum(cube * velocities[None, None, :], dim=2) / torch.sum(cube, dim=2)

    else:
        print("ERROR: Not a valid backend")
        return

    return vel_map


def create_pvd(rotated_cube, slice_start, slice_end):
    """
    Rotated cube: shape (n_minor_axis, n_major_axis, n_freq)
    """
    return np.flip(np.rot90(rotated_cube[slice_start:slice_end, :, :].sum(axis = 0)))