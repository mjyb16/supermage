import torch
from caskade import Module, forward, Param

class ExponentialDisk2D(Module):
    """
    Purely radial exponential surface–brightness profile for a razor-thin disk.
    """
    def __init__(self):
        super().__init__()
        self.scale = Param("scale", None)              # pc

    @forward
    def brightness(self, R_map, scale=None):
        """
        I(R) ∝ exp(−R / scale)
        """
        return torch.exp(-R_map / scale)               # (H,W)


def cutoff(r, start, end, device = "cuda"):
    """
    Creates a cutoff in a surface brightness profile between two radii. 
    This is a PyTorch-compatible version of KinMS.sb_profs.cutoff of Davis et al. (2013)
    """
    
    # Convert all entries to PyTorch tensors
    if type(r) is not torch.tensor:
        r=torch.tensor(r, device = device)
    
    return ~((r>=start)&(r<end))