# API reference

SuperMAGE is organized into three subpackages:

| Subpackage | Contents |
|---|---|
| [`supermage.simulators`](simulators.md) | Forward models: intensity profiles, velocity/mass models, spectral-cube renderers, visibility simulators, and lensed-cube simulators. |
| [`supermage.solvers`](solvers.md) | Inference: Levenberg–Marquardt optimizers, Sobol swarm global search, and MALA MCMC. |
| [`supermage.utils`](utils.md) | Support: unit conversions, velocity grids, primary beams, visibility likelihoods, and plotting/diagnostic helpers. |

## The caskade pattern

All forward models are [caskade](https://github.com/Ciela-Institute/caskade)
`Module`s. Two kinds of inputs are distinguished:

- **Static configuration** is passed to the constructor (grid sizes, device,
  dtype, frequency axes, ...). It never changes during a fit.
- **Simulation parameters** are `caskade.Param` attributes (inclination,
  black-hole mass, flux, ...). They can be supplied at call time as a flat
  tensor — `model.forward(theta)` — which is what makes every SuperMAGE
  model directly usable as a differentiable likelihood forward function.

Inspect any assembled model with `model.graphviz()` (parameter graph) and
`model.dynamic_params` (the flat parameter order expected by `forward`).
