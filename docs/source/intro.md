# SuperMAGE

**Super**b **ma**sses from **g**as kin**e**matics — a differentiable,
GPU-accelerated forward-modeling toolkit for galaxy gas kinematics, built on
PyTorch and [caskade](https://github.com/Ciela-Institute/caskade).

SuperMAGE's primary use case is measuring supermassive black-hole masses
from cold-gas rotation observed with interferometers such as ALMA, but every
piece is modular: if you work with spectral cubes and/or rotation curves in
galaxies, you can use any of its components independently. It also
interfaces with the gravitational-lensing package
[caustics](https://github.com/Ciela-Institute/caustics), so strongly lensed
gas disks can be modeled with the same machinery.

## What it does

A SuperMAGE model is a chain of differentiable modules:

1. **Mass model → rotation curve.** Analytic stellar profiles (Sérsic,
   Nuker, Core-Sérsic) are converted on the fly to Multi-Gaussian
   Expansions whose circular velocity — including a central black hole and
   optionally the gas disk's self-gravity — is computed with a fast, exact
   quadrature ({mod}`supermage.simulators.velocity_models`).
2. **Rotation curve + intensity profile → spectral cube.** Deterministic
   inverse-mapped rendering or a KinMS-style Monte-Carlo cloud catalogue
   ({mod}`supermage.simulators.analytic_cube`), optionally raytraced
   through a lens ({mod}`supermage.simulators.lensed_cube`).
3. **Cube → visibilities.** Primary beam, flux normalization, padding,
   gridding-kernel taper, FFT and uv masking
   ({mod}`supermage.simulators.visibility_cube`), matching data gridded
   with [viscube](https://github.com/mjyb16/viscube).
4. **Inference.** Because the whole chain is autograd-differentiable, you
   can fit it with gradient-based optimizers, sample it with
   gradient-based MCMC ({mod}`supermage.solvers`), or plug it into nested
   samplers via the vectorised multi-GPU likelihood helpers
   ({mod}`supermage.utils.likelihood`).

## Where to start

- {doc}`install`
- {doc}`notebooks/01_getting_started` — build a galaxy model and render
  your first cube.
- {doc}`notebooks/02_mass_models` — rotation curves, MGE machinery, and
  composite mass models.
- {doc}`notebooks/03_cloud_catalog` — the Monte-Carlo cloud renderer.
- {doc}`notebooks/04_visibilities` — from cubes to interferometric
  visibilities and dirty images.
- {doc}`notebooks/05_fitting_mock_data` — an end-to-end fit: global search,
  Levenberg–Marquardt polish, and MALA posterior sampling.
- {doc}`notebooks/06_lensed_cubes` — lensed kinematics with caustics.
- {doc}`api/index` — full API reference.

```{note}
SuperMAGE is under active development; interfaces may evolve between
releases. Bug reports and contributions are welcome on
[GitHub](https://github.com/mjyb16/supermage).
```
