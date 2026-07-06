# supermage — package notes for Claude

PyTorch/caskade forward modeling + inference for gas-kinematic SMBH
measurement. Editable install in the `latest_supermage` env; also deployed to
DRAC cluster virtualenvs (`$HOME/envs/supermage_2025`) which must be re-synced
after any commit here. Depends on `caustics` for lensing; **pykeops is NOT
required** (old notes claiming otherwise are stale).

## Layout
- `simulators/`: `velocity_models.py` (MGE / SersicMGE / NukerMGE /
  CoreSersicMGE + QuadratureVelocitySum), `analytic_cube.py` (CloudCatalog
  rasterizer), `lensed_cube.py` (CubeLens, AnalyticLens), `visibility_cube.py`
  (cube → UV FFT + taper), `intensity_models.py`, `velocity_scatter.py`
- `solvers/`: `optimizers.py` (LM), `swarm_optimization.py`, `MCMC.py` (MALA)
- `utils/`: `likelihood.py`, `primary_beams.py`, `doppler_velocities.py`,
  `plotting.py`, `angular_distance.py`

## Load-bearing code — do NOT "simplify" or revert
- `velocity_models.py::core_sersic_torch_1d` uses a logaddexp/log-space form.
  Reverting to direct `torch.pow` reintroduces an fp32 overflow that inflated
  profiles ×462 and created fake "extended R_e" posterior modes (fixed 5468ee6).
- `lensed_cube.py::AnalyticLens.forward` has a `_bad = ~isfinite(...)` guard
  zeroing I_map/v_los at the EPL center singularity. Without it ~1/8 of valid
  prior draws NaN-poison the whole cube via flux normalization and get
  rejected (fixed d1483b8). Pipeline drivers source-check this guard at
  preflight.
- `utils/likelihood.py`: chi²/logL reductions are cast to **float64** (fp32
  quantization → nested-sampling plateaus / "Distances are not positive").
  Half-plane npz support: n_chi = sum(mask) for half-plane files vs
  sum(mask)/2 for legacy full-plane (Hermitian double-count fix).
- `visibility_cube.py` **multiplies** `image_taper_map` into the padded image
  before the FFT (matches viscube m=1 KB gridding). The old
  divide-by-apodization convention was inverted — never restore it.

## Conventions
- caskade: models take a flat θ vector — verify `.dynamic_params` order before
  building priors/bounds; fix parameters with `.to_static(...)`.
- `m_bh` is log10(M☉); model radii are arcsec (→ pc via pc_per_arcsec =
  D_Mpc·π/0.648); parametric-MGE models fix M/L=1 (amplitude carries units).
- `create_velocity_grid_stable(GHz, ascending)`; `gaussian_pb(freq_hz, ...,
  fwhm_factor=1.13)` validates its input is Hz. `casa_airy_beam` is KNOWN
  BROKEN (documented in its docstring) — don't use it, don't "fix" it silently.
- `CloudCatalog` `fov_half_pc` has a deprojection gotcha — documented in docs
  tutorial notebook 03.

## Docs / tests
- Docs: jupyter-book (**must stay <2**, pinned) building `docs/source/`;
  tutorial notebooks 01–06 are executed; API pages are static (sphinx-apidoc
  was removed from the RTD build on purpose).
- **No test suite exists.** The de-facto regression checks are
  `~/Code/sknkwx_supermage/publication_code/ngc4697/publication_run_nautilus/scripts/selfcheck_pipeline_fixes.py`
  (25 checks) and the pipeline drivers' preflights. Run the selfcheck after
  touching `simulators/` or `utils/likelihood.py`.
