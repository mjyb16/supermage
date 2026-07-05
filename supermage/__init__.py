"""SuperMAGE: Superb masses from gas kinematics.

A differentiable, GPU-accelerated (PyTorch + caskade) forward-modeling
toolkit for galaxy gas kinematics — from analytic mass/intensity models to
spectral cubes, interferometric visibilities and (optionally lensed, via
caustics) observations — with gradient-based optimizers and samplers for
supermassive black-hole mass measurement.

Subpackages
-----------
``supermage.simulators``
    Intensity, velocity/mass and cube/visibility forward models.
``supermage.solvers``
    Levenberg-Marquardt optimizers, Sobol swarm strategies and MALA MCMC.
``supermage.utils``
    Unit conversions, primary beams, likelihood infrastructure and
    plotting/diagnostic tools.
"""
from . import __meta__

__version__ = __meta__.version
