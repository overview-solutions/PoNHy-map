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
- `ponhy_config.txt` - **configuration file** loaded automatically if present.
- `utils/` - helper modules (plotting, physics, Monte Carlo, reporting, etc.).
- `Data/` - input datasets (topography, gravity, magnetic, temperature, etc.).

---

## Configuration (`ponhy_config.txt`)

The config file is a plain text file with `KEY = value` pairs and comments. If it exists, it **overrides** the defaults in `ponhy.py`. If it’s missing, PoNHy uses the internal defaults.

### Rules

- Booleans: `true/false` (case‑insensitive).
- Strings: use quotes for paths, e.g. `"/Data/Ext_Topo.txt"`.
- Arrays: use Python‑style lists: `[1, 2, 3]`.
- Sections: lines like `[MC_NO_SATURATION_CONFIG]` start a nested dictionary.

If the file has errors or missing keys, PoNHy prints the issue and **exits**.

---

## Running the code

## Environment setup

PoNHy ships with a pinned Conda environment definition (`environment.yml`).

### Option A - Conda (recommended)

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate ponhy
```

### Option B - Pip

If you prefer pip in an existing environment:

```bash
pip install -r requirements.txt
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

1. Edit `ponhy_config.txt` to set paths and parameters.
2. Run `ponhy.py`.
3. Review plots and CSV outputs in the timestamped results folder.

---

## Notes

- The script is designed for scientific workflows and can be compute‑intensive.
- If you run both routines, the inversion outputs are reused in the H₂ stage.
- If you set `RUN_INVERSION = false`, only the H₂ quantification runs (but you need to provide files for the serpentinites).

---

## Troubleshooting

- **Missing file**: check paths in `ponhy_config.txt`.
- **Config error**: PoNHy prints the invalid key or missing section and exits.
- **Slow runs**: reduce Monte Carlo iterations or set `N_CORES` to a smaller value.

---

## Contact

Developed by **Rodolfo Christiansen (LIAG)**  
Contact: rodo_christiansen@hotmail.com - Rodolfo.Christiansen@liag-institut.de
