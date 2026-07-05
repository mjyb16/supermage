"""Analytic surface-brightness models for the gas disk.

Intensity models expose ``brightness(R_map) -> Tensor`` where ``R_map``
contains intrinsic (deprojected, midplane) radii in parsec; the returned
map is in *relative* units — the cube simulators normalize total flux
separately (see the ``flux`` Param of the visibility simulators), so only
the profile *shape* matters here.
"""
import torch
from caskade import Module, forward, Param

class ExponentialDisk2D(Module):
    """Purely radial exponential surface-brightness profile of a thin disk.

    ``I(R) = exp(-R / scale)`` — the normalization is intentionally left to
    the downstream flux calibration.

    caskade Params: ``scale`` — the exponential scale length [pc].
    """
    def __init__(self):
        super().__init__()
        self.scale = Param("scale", None)              # pc

    @forward
    def brightness(self, R_map, scale=None):
        """Evaluate ``I(R) = exp(-R / scale)``.

        Parameters
        ----------
        R_map : Tensor
            Intrinsic radii [pc]; any shape (typically ``(H, W)``).
        scale : optional
            The scale-length Param [pc], filled by caskade.

        Returns
        -------
        Tensor
            Relative brightness, same shape as ``R_map``.
        """
        return torch.exp(-R_map / scale)               # (H,W)


def cutoff(r, start, end, device = "cuda"):
    """Boolean mask that zeroes a surface-brightness profile inside an annulus.

    PyTorch port of ``KinMS.sb_profs.cutoff`` (Davis et al. 2013): returns
    ``False`` (i.e. "cut") where ``start <= r < end`` and ``True`` elsewhere,
    so multiplying a brightness profile by the result removes the annulus.

    Parameters
    ----------
    r : Tensor or array-like
        Radii (any units, consistent with ``start``/``end``).
    start, end : float
        Inner and outer edge of the annulus to cut.
    device : str, optional
        Device used if ``r`` needs to be converted to a tensor.

    Returns
    -------
    Tensor of bool
        ``~((r >= start) & (r < end))``, same shape as ``r``.
    """

    # Convert all entries to PyTorch tensors
    if type(r) is not torch.tensor:
        r=torch.tensor(r, device = device)

    return ~((r>=start)&(r<end))
