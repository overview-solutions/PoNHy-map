# PoNHy - Potential for Natural Hydrogen

PoNHy is a research workflow that runs two main routines:

1. **Geophysical inversion** (gravity + magnetic) to build density/susceptibility models (You must check for the convergnece of the inversion!!!).
2. **H₂ quantification** using thermodynamic databases and Monte Carlo sampling.

Everything is driven from `ponhy.py` and an editable configuration file.

---

## What this code does (high level)

- Loads topography, gravity, magnetic data, and an optional initial model.
- Builds a 3D mesh and runs a **PGI inversion** (SimPEG) to estimate density and susceptibility.
- Saves outputs (CSV + plots) into a timestamped results folder.
- Uses the inversion results to compute **serpentinization**, **H₂ production**, and **flow/pressure** analyses.
- Runs Monte Carlo simulations (with/without saturation), convergence sweeps, and sensitivity analyses.

---

## Key files

- `ponhy.py` - **main entry point** (runs the full workflow).
- `ponhy_config_*.yaml` - **configuration files** (the runner will prompt you to choose).
- `utils/` - helper modules (plotting, physics, Monte Carlo, reporting, etc.).
- `Data/` - input datasets (topography, gravity, magnetic, temperature, etc.).

---

## Configuration (`ponhy_config_*.yaml`)

The config file is YAML and is required. It defines the two main routines and all parameters.

### Selecting a YAML at runtime

When you run `ponhy.py`, it scans the **current working directory** for `*.yaml` / `*.yml` files.
If multiple YAMLs are found, it lists them and prompts you to choose one. In non-interactive runs,
it defaults to the first YAML found. If no YAMLs are found, it falls back to
`ponhy_config_pyrenees.yaml`.

Example prompt (selecting Pyrenees):

```
[PoNHy] Config YAMLs found:
	1) ponhy_config_california.yaml
	2) ponhy_config_pyrenees.yaml
Select YAML to use (number): 2
```

### Rules

- Booleans: `true/false`.
- Strings: use quotes for paths, e.g. `"/Data/Ext_Topo.txt"`.
- Arrays: use YAML lists: `[1, 2, 3]` or multi-line lists.
- Nested blocks: e.g. `MC_NO_SATURATION_CONFIG:` define grouped settings.

### Optional sections via run flags

- If `RUN_INVERSION: false`, **all inversion parameters may be omitted**.
- If `RUN_H2_QUANTIFICATION: false`, **all quantification parameters may be omitted**.
- If a routine is enabled (`true`), its parameters are required.

---

## Running the code

## Environment setup

PoNHy ships with a pinned Conda environment definition (`environment.yml`).

### Conda (recommended)

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate hydrogen
```

From the `PoNHy` folder, run:

```bash
python ponhy.py
```

Outputs are saved in a timestamped folder inside `BASE_DIR`, for example:

```
Results_Inversion_YYYYMMDD_HHMMSS/
Results_GenerationYYYYMMDD_HHMMSS/
```

---

## What the `utils/` modules do

PoNHy is modular; key helpers live in `utils/`:

- `geometry.py` - mesh construction, interpolation, volume/mask calculations.
- `general.py` - thermodynamic lookup, serpentinization math, conversion utilities.
- `plotting.py` - plotting utilities and result visualizations.
- `reporting.py` - formatted summaries and reports for each step.
- `helpers.py` - common helpers (sampling, scaling, progress bars, etc.).
- `saturation.py` - saturated H₂ Monte Carlo workflow.
- `no_saturation.py` - no‑saturation H₂ Monte Carlo workflow.
- `uncertainties.py` - convergence sweeps and limiting‑factor analysis.
- `logging.py` - custom print logging to file.

---

## Typical workflow

1. Edit `ponhy_config.yaml` to set paths and parameters.
2. Run `ponhy.py`.
3. Review plots and CSV outputs in the timestamped results folder.

---

## Notes

- The script is designed for scientific workflows and can be compute‑intensive.
- If you run both routines, the inversion outputs are reused in the H₂ stage.
- If you set `RUN_INVERSION: false`, only the H₂ quantification runs (you still need the serpentinite inputs).
- If you set `RUN_H2_QUANTIFICATION: false`, only the inversion runs.

### Reproducibility

- `SEED` + `USE_GLOBAL_SEED: true` enforce a **single global seed**.
- Individual sampling seeds are not used when global seed is enabled.

---

## Troubleshooting

- **Missing file**: check paths in `ponhy_config.yaml`.
- **Config error**: PoNHy prints the invalid key or missing section and exits.
- **Slow runs**: reduce Monte Carlo iterations or set `N_CORES` to a smaller value.

---

## Contact

Developed by **Rodolfo Christiansen (LIAG)**  
Contact: rodo_christiansen@hotmail.com - Rodolfo.Christiansen@liag-institut.de
