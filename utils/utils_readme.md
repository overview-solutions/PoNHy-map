# PoNHy Utils Reference (utils_readme)

This document explains the `utils/` modules used by `ponhy.py`: what each module does, the key functions, and the main parameters/outputs. It is intended as a practical reference for reading or extending the workflow.

---

## Main workflow: `ponhy.py`

`ponhy.py` is the main orchestrator: it loads the YAML configuration, validates required inputs, runs the inversion (if enabled), then runs H₂ quantification (if enabled). It also redirects output to a log file per run, so every run has a reproducible text report.

**Inputs (from YAML):**
- Paths: `BASE_DIR`, `TOPO_FILE`, `GRAV_FILE`, `MAG_FILE`, `INITIAL_MODEL`, `DB_DIR`, `TEMPERATURE_FILE`, `DENSITY_FILE`, `MAGSUS_FILE`.
- Inversion settings: mesh cell sizes, PGI regularization weights, misfit weighting, iteration limits.
- Quantification settings: rock/fluid parameters, Monte Carlo configs, flow‑target configs.

**Runtime outputs:**
- `Results_Inversion_YYYYMMDD_HHMMSS/` with CSVs + plots (gravity/magnetic fits, GMM plots, model exports).
- `Results_GenerationYYYYMMDD_HHMMSS/` with H₂ summary CSVs, sensitivity plots, and reports.

**Key stages inside `ponhy.py`:**
1. **Config loading**: `load_config(...)` parses YAML → a `Config` dataclass. This centralizes defaults and ensures required keys are present.
2. **Base/data resolution**: after loading the YAML, `ponhy.py` resolves `BASE_DIR` (auto or hardcoded), then finds `Data*` folders (case-insensitive) and prompts you to select one when multiple exist. Filenames in the YAML are resolved inside that folder; absolute paths and `BASE_DIR`-relative paths are also supported.
2. **Inversion** (if `RUN_INVERSION=true`):
  - Build mesh and load data from the paths in the config.
  - Run PGI inversion (SimPEG) to estimate density and magnetic susceptibility.
  - Export model grids and plots used later by the H₂ workflow.
3. **Quantification** (if `RUN_H2_QUANTIFICATION=true`):
  - Compute volumes, temperature‑range masks, and serpentinization degree.
  - Run **no‑saturation** MC first (upper‑bound H₂).
  - Run **saturation** MC next (solubility/flow‑limited H₂).
  - Optional convergence, univariate sweeps, and flow‑target diagnostics.

The important idea: `ponhy.py` does not implement low‑level math itself. It **delegates** all core calculations to `utils/` and only coordinates the workflow.

---

## How `ponhy.py` uses `utils/`

At a high level `ponhy.py`:

1. **Loads YAML config** via `utils.config.load_config` into a `Config` dataclass.
2. **Inversion routine** uses
   - `geometry.py` (mesh and interpolation),
   - `plotting.py` (misfit/field plots),
   - `reporting.py` (formatted tables),
   - `helpers.py` (sampling/progress/utility helpers).
3. **Quantification routine** uses
   - `general.py` (thermo lookups, serpentinization math, conversions),
   - `no_saturation.py` + `saturation.py` (Monte Carlo pipelines),
   - `plotting.py` + `reporting.py` (figures and tables),
   - `uncertainties.py` (MC sweeps and limiting‑factor analysis).

The `utils.config` module defines dataclasses and helper “builder” functions so that `ponhy.py` can pass *structured parameters* (instead of long argument lists) to each workflow function.

---

## Module by module

### `utils/config.py`
**Purpose:** YAML parsing + schema, and “builder” helpers that transform a runtime scope into structured dataclasses.

This module guarantees that all downstream computations receive **typed parameter bundles** instead of long positional argument lists. That makes it easier to validate inputs and keep functions stable as the workflow grows.

**Key dataclasses (inputs):**
- `Config`: full YAML configuration (paths, mesh, inversion, MC settings). This is the single source of truth for a run.
- `FlowTargetFractureConfig`: geometry + hydraulic ranges for the fracture Monte Carlo.
- `McNoSaturationConfig`: Monte Carlo controls for the no‑saturation path (sampling mode, iteration count, factor ranges).
- `McSaturationConfig`: Monte Carlo controls for the saturation path (sampling mode, dt, chunk size, factor ranges).
- `UnivariateAnalysisConfig`: one‑at‑a‑time sensitivity sweeps (points, reps, sampling).
- `McConvergenceSweepConfig`: convergence sweep settings (iter list, tolerance thresholds).
- `McFlowTargetConfig`: per‑flow‑target Monte Carlo settings.

**Workflow parameter dataclasses (used by other modules):**
- `InversionConfigSummary`, `InversionMeshConfig`, `InversionResultsSummary`.
- `SerpentinizationSummaryParams`, `MeanLithostaticPressureParams`.
- `ThermoLookupParams`, `ProductionRateReportParams`, `ProductionRateVolumetricParams`.
- `SerpentinizationFrontVelocityParams`, `WaterFlowSerpentinizationParams`.
- `NoSaturationWorkflowParams`, `SaturationWorkflowParams`.
- `NoSaturationSummaryReportParams`, `SaturationSummaryParams`.
- `FlowTargetLimitingFactorsParams`, `SaturationMonteCarloReportParams`.

These dataclasses bundle the *exact inputs* each workflow step needs, so functions stay stable and easy to test.

**Key functions:**
- `load_config(path: str) -> Config`
  - Reads YAML, validates keys, applies defaults for disabled routines, and returns a populated `Config`.
  - Enforces the global seed rule (when enabled) and fills optional sections if a routine is disabled.
- `build_*_params(scope: Dict[str, Any])`
  - Example: `build_no_saturation_workflow_params(cfg, locals())`.
  - These assemble dataclasses from a scope dictionary (`locals()`), using defaults and alias resolution so summaries don’t need repetitive alias variables.

**Important parameters:**
- `run_inversion`, `run_h2_quantification` to enable/disable whole sections.
- `mc_no_saturation_config`, `mc_saturation_config` for Monte Carlo ranges and iteration counts.
- `seed` + `use_global_seed` for reproducibility.

---

### `utils/helpers.py`
**Purpose:** generic utilities used across the workflow (sampling, progress bars, config parsing, normalization).

This module is the “toolbox.” Anything that is reused across MC sampling, reporting, or plotting ends up here so logic is not duplicated in multiple modules.

**Key functions:**
- `_resolve_base_dir_path(base_dir, path)`
  - Resolves inputs relative to `BASE_DIR` when needed (legacy helper used in some modules).
  - `ponhy.py` now also supports filename-only paths resolved inside the selected `Data*` folder.
- `_normalize_unit_parameters(labels, dens_adj, magsus, dens_disp, magsus_disp, weights)`
  - Validates that all unit lists have the same length.
  - Normalizes weights to sum to 1 and returns NumPy arrays for downstream math.
- `_clamp_worker_count(n)` / `_compute_mc_chunk_size(n_iter, range_count)`
  - Bounds parallel workers to available CPUs and picks a sensible MC chunk size.
- `_sample_unit_hypercube(n, d, mode, seed)` + `_scale_samples_to_ranges(unit, ranges)`
  - Generates MC samples in [0,1]^d and maps them into physical ranges.
- `_get_progress_bar(...)`
  - Ensures progress bars still display even when stdout is redirected to file.
- `_moving_average`, `_select_stable_window`, `_trim_trailing_dropoff`
  - Stabilize time‑series outputs (e.g., saturation convergence or flow‑target series).

---

### `utils/geometry.py`
**Purpose:** mesh construction, interpolation, and volume/mask calculations.

This module provides **spatial preprocessing**: it builds the mesh, interpolates data to mesh cells, and computes masks/volumes that later convert the inversion results into rock volumes for H₂ calculations.

**Key functions:**
- `build_mesh_from_topography(topo_xyz, dx, dy, dz, depth_core)`
  - Builds the 3D mesh used by SimPEG and returns `(mesh, nx, ny, nz)`.
- `interpolate_nearest_neighbor(xyz_src, values, xyz_target)`
  - Transfers properties (e.g., initial model densities) onto active mesh cells.
- `compute_temp_range_depths(temperature_mesh, xyz_mesh_temperature, temperature_ranges)`
  - Computes min/max depth per temperature bin for later reporting.
- `compute_rock_volumes_and_masks(temperature_mesh, dx, dy, dz, mask_not_nan)`
  - Computes serpentinite volumes per temperature range and masks used in the H₂ workflow.

---

### `utils/general.py`
**Purpose:** thermodynamic lookup utilities, serpentinization math, conversions, and data summaries.

Think of this module as the **geochemical and lookup layer**. It reads thermodynamic data, builds pressure/temperature lookups, and provides unit conversions used by the Monte Carlo routines.

**Key functions:**
- `load_h2_production_database(dsn_db, rock_codes=None, ftype="zarr")`
  - Loads Zarr databases into `xarray` objects used by the MC workflows.
- `build_thermo_lookup_by_range(params: ThermoLookupParams)`
  - Builds per‑temperature statistics (e.g., solubility, production rate).
- `build_production_rate_volumetric_dict(params: ProductionRateVolumetricParams)`
  - Creates a temperature‑range → production‑rate mapping grouped by W/R ratio.
- `weighted_average_rates(closest, waterrockratio, production_rate_volumetric, temp_bins)`
  - Interpolates between two W/R ratios when an exact match is absent.
- `compute_serpentinization_degree(...)`
  - Uses lab data surfaces to convert density/magsus into serpentinization %.
- `compute_serpentinization_front_velocities(params: SerpentinizationFrontVelocityParams)`
  - Scales a reference velocity across temperature bins using production rates.
- `compute_mean_lithostatic_pressure_by_range(params: MeanLithostaticPressureParams)`
  - Converts depth ranges into pressure estimates for each temperature bin.
- `convert_h2_mol_to_kg(mol_h2)` / `compute_h2_solubility_kk_pr(P_MPa, T_C)`
  - Unit conversions and solubility calculations used inside MC loops.
- `save_saturation_csv(stats_mc_saturation, mean_pressure_ranges, results_path)`
  - Writes the saturation summary table (means/stds per temperature bin).

---

### `utils/plotting.py`
**Purpose:** all plotting helpers (inversion plots, lab panels, MC summaries).

Plotting is centralized here so output styles and export logic are consistent. Most plots return a Matplotlib figure and then save PNG/SVG files into the results folder.

**Key functions:**
- `set_plot_save_svg(enabled: bool)`
  - Global switch for SVG output (in addition to PNG).
- `plot_misfit_evolution(save_values_directive, results_path)`
  - Saves gravity/magnetic misfit trends from the inversion iterations.
- `plot_learned_gmm(...)`
  - Plots the learned petrophysical clusters against the recovered model.
- `plot_gravity_observed_predicted_residual(...)` / `plot_magnetic_observed_predicted_residual(...)`
  - Compares observed data with model predictions and residuals.
- `plot_residual_histograms(residual_grav, residual_mag, results_path)`
  - Visual diagnostic of residual distribution and spread.
- `prepare_lab_plot_inputs(...)` + `plot_lab_data_panels(...)`
  - Generates the 8‑panel lab comparison figure (volumes, velocities, solubility, etc.).
- `plot_h2_production_summary(...)`
  - Summary H₂ bar plots by temperature range (with optional error bars).

---

### `utils/reporting.py`
**Purpose:** structured printouts and text‑table summaries.

This module turns computed arrays and DataFrames into **human‑readable tables** printed to the log files. It does not compute science outputs; it formats and explains them.

**Key functions:**
- `print_inversion_config_summary(summary: InversionConfigSummary)`
  - Full inversion configuration table (inputs + mesh + GMM parameters).
- `print_inversion_config(params: InversionMeshConfig)`
  - Compact print of mesh settings and fracture spacing.
- `print_inversion_results_summary(summary: InversionResultsSummary)`
  - Prints output file paths and runtime summary for inversion.
- `print_summary_serpentinization_section(params: SerpentinizationSummaryParams)`
  - Volume/serpentinization summary before the MC runs.
- `print_production_rate_volumetric_report(params: ProductionRateReportParams)`
  - Prints per‑temperature production rates from the thermo database.
- `print_parameters_water_flow_serpentinization(params: WaterFlowSerpentinizationParams)`
  - Prints hydraulic/flow parameters used in the saturation model.
- `print_no_saturation_report(df)` / `print_no_saturation_summary_report(...)`
  - Detailed and summary tables for the no‑saturation results.
- `print_saturation_monte_carlo_report(params: SaturationMonteCarloReportParams)`
  - Detailed per‑temperature saturation results from MC stats.
- `print_saturation_summary(params: SaturationSummaryParams)`
  - Final total H₂ statistics (total, std, efficiency).
- `save_mc_convergence_sweep_report(sweep_df, results_dir, save_plot=True)`
  - Saves CSV + convergence plot for MC iteration sweeps.

---

### `utils/no_saturation.py`
**Purpose:** Monte Carlo pipeline without saturation limits (theoretical H₂ production).

This module computes the **upper‑bound H₂ production** assuming solubility is unlimited. It is run first and provides a baseline to compare against saturation‑limited results.

**Key functions:**
- `run_no_saturation_workflow(params: NoSaturationWorkflowParams)`
  - End‑to‑end pipeline that returns MC stats, CSV paths, and summary totals.
- `compute_h2_production_no_saturation(...)`
  - Core MC sampler. Inputs control serpentinization velocity, production rates, and rock volumes.
  - Returns per‑temperature production results and (optionally) per‑iteration samples.

---

### `utils/saturation.py`
**Purpose:** Monte Carlo workflow with solubility/saturation limits.

This module applies solubility/flow constraints. It models daily water delivery and saturation, producing more realistic H₂ values and efficiency metrics.

**Key functions:**
- `run_saturation_workflow(params: SaturationWorkflowParams)`
  - Orchestrates the full saturation‑limited MC, summary stats, and plots.
- `run_saturation_monte_carlo(...)`
  - Core MC sampler for the saturation path. It handles chunking, parallel workers,
    and per‑temperature water/solubility calculations.

---

### `utils/uncertainties.py`
**Purpose:** Monte Carlo sweeps and sensitivity analysis.

This module answers the “how stable are the results?” question by running sweeps over `n_iter`, or by changing one parameter at a time to see how much each factor shifts the H₂ output.

**Key functions:**
- `run_fracture_monte_carlo_simulation(...)`
  - Uses random fracture geometries + Darcy flow to see which samples meet the flow target.
- `run_mc_convergence_sweep(config, base_df, mc_kwargs, results_dir=None)`
  - Re‑runs the MC at different `n_iter` values and checks mean/std stability.
- `run_univariate_saturation(...)` / `run_univariate_no_saturation(...)`
  - Sweeps one factor at a time to identify the most sensitive parameters.
- `analyze_limiting_factors_by_flow_target(params: FlowTargetLimitingFactorsParams)`
  - Runs a flow‑target sweep to label dominant limitations (e.g., solubility vs rate).

---

### `utils/logging.py`
**Purpose:** simple output redirection for logs.

`CustomPrint` duplicates stdout to a log file, so every run produces a reproducible text report alongside plots/CSVs.

**Key functions:**
- `CustomPrint(file_path)`
  - Redirects `stdout` to both terminal and a text log file for reproducible reports.
- `log_info(msg)`
  - Convenience wrapper for `print(..., flush=True)` used for lightweight logging.

---

## Practical usage tips

- **Prefer builder dataclasses** from `utils.config` instead of passing long argument lists.
- **Use `set_plot_save_svg`** to control SVG output globally.
- **Large MC runs**: control runtime with `mc_*_config.n_iter` and `N_CORES` in YAML.
- **Missing parameters**: `ponhy.py` validates required inputs; disabled routines can omit their sections.

---

## Where to look next

- `ponhy.py` — orchestrates everything and shows exact data flow.
- `utils/config.py` — the best place to understand required parameters.
- `utils/no_saturation.py` / `utils/saturation.py` — core MC computations.
- `utils/uncertainties.py` — convergence sweeps and flow‑target analysis.

```
