# PoNHy - Potential for Natural Hydrogen

PoNHy is a research workflow that runs two main routines:

---

## Quick start

1. **Environment** (choose one):
   - **Conda:** `conda env create -f environment.yml` then `conda activate hydrogen`
   - **Pip:** `python3 -m venv .venv`, `source .venv/bin/activate`, `pip install -r requirements.txt`
2. **Run** (use the venv so dependencies like `packaging` are available):
   - **Easiest:** `./run_ponhy.sh` (activates `.venv` and runs `ponhy.py`)
   - **Or:** `source .venv/bin/activate` then `python ponhy.py` from the `PoNHy` directory
3. If multiple config or Data folders exist, the script will prompt you to choose (or in non-interactive mode it uses the first of each). **Use the Data folder that matches your config** (e.g. Pyrenees config → Data_Pyrenees, California config → Data_California).

Outputs go into timestamped folders under the same directory (e.g. `Results_Inversion_YYYYMMDD_HHMMSS/`).

---

1. **Geophysical inversion** (gravity + magnetic) to build density/susceptibility models (You must check for the convergnece of the inversion!!!).
2. **H₂ quantification** using thermodynamic databases and Monte Carlo sampling.

Everything is driven from `ponhy.py` and an editable configuration file.

---

## Runtime note (important)

This workflow is **computationally expensive**. A full run (inversion + H₂ quantification + Monte Carlo) can take **hours** and can **overload or crash** a laptop (high CPU/memory).

**To avoid overloading your machine:**

- **Use `./run_ponhy.sh`** – It limits parallelism (e.g. 2 cores by default). To allow more: `PONHY_MAX_CORES=4 ./run_ponhy.sh`. To remove limits: `PONHY_UNLIMITED=1 ./run_ponhy.sh`.
- **Use the light config** – When prompted for a YAML, choose **`ponhy_config_light.yaml`** (same data as California but fewer iterations, 1 core, and optional analyses disabled). Much lighter on CPU and memory.
- **Disable heavy parts** – In any YAML you can set `RUN_INVERSION: false` or `RUN_H2_QUANTIFICATION: false`, or turn off `RUN_MONTECARLO_FAULT`, `RUN_UNIVARIATE_ANALYSIS_*`, `RUN_MC_CONVERGENCE_SWEEP`, and `RUN_ANALYZE_LIMITING_FACTORS` to reduce load.

To speed things up without crashing, you can **disable components** in the YAML (e.g. set `RUN_INVERSION` or `RUN_H2_QUANTIFICATION` to `false`, or skip the univariate/MC sweeps).

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
- `Auxiliary_functions/` - auxiliary utilities (e.g., temperature distribution calculations).
- `Data/` - input datasets (topography, gravity, magnetic, temperature, etc.).
- `export_to_geojson.py` - **optional**: convert result CSVs (or input topo/gravity files) to GeoJSON (WGS84) for use on web maps. Requires `pyproj`. Example (run commands one at a time):

  ```bash
  pip install pyproj
  python export_to_geojson.py -o map.geojson
  python export_to_geojson.py Data_Pyrenees/Ext_Topo.txt -o topo.geojson --format topo
  ```
  Use **`--polygons`** to export grid cells as polygons (color by value in Kepler.gl or QGIS).  
  Use **`--value-ranges`** to merge cells into one polygon per value band (contour-style; needs `shapely`):

  ```bash
  python export_to_geojson.py Results_Inversion_*/Density_complete_model.csv -o density_polygons.geojson --polygons
  python export_to_geojson.py Results_Inversion_*/Density_complete_model.csv -o density_value_ranges.geojson --value-ranges --bins 8
  ```
  Then open the `.geojson` in geojson.io, Kepler.gl, or any GIS. Use `--utm 11` for California data.

  **Vanilla Mapbox map:** `mapbox_map.html` in this repo loads the three polygon GeoJSONs (density, gravity, topography) with the Mapbox API. **Set your Mapbox token** in the HTML (replace `YOUR_MAPBOX_ACCESS_TOKEN` with a token from [Mapbox Access Tokens](https://account.mapbox.com/access-tokens/)); do not commit real tokens. Open the map over a **local server** (e.g. `python -m http.server 8080` in the PoNHy folder, then visit `http://localhost:8080/mapbox_map.html`) so the GeoJSON files can be fetched. To change the initial view (e.g. Southern Africa), edit the `MAP_CONFIG` object in the HTML: `center: [longitude, latitude]` (e.g. `[26, -26]` for Southern Africa), `zoom`, and `style` (e.g. `mapbox://styles/mapbox/satellite-v9`).

  **Sharing a Kepler.gl map (Foursquare/Studio):** Saving to Foursquare often stores only the map config, not the datasets, so the published map can open with no data or a default basemap. To keep data and basemap when sharing:
  - **Export as HTML** (if available in your Kepler.gl): produces a single file with map + data embedded; you can open it locally or host it (e.g. GitHub Pages).
  - **Export map as JSON** (with "include data" if there is an option): keep that file as a backup and re-import it where needed.
  - **Host GeoJSON on a public URL** (e.g. GitHub repo "raw" link) and add those URLs when opening the shared map so the same layers and view are reproducible.

---

## Environment setup

PoNHy ships with a pinned Conda environment definition (`environment.yml`).

### Conda (recommended)

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate hydrogen
```

### Pip / venv (alternative)

If you don't use Conda, create a virtual environment and install from `requirements.txt`:

```bash
cd PoNHy
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then run from the same directory: `python ponhy.py`.

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

## Configuration (`ponhy_config_*.yaml`)

The config file is YAML and is required. It defines the two main routines and all parameters.

### Selecting a YAML at runtime

When you run `ponhy.py`, it scans the **current working directory** for `*.yaml` / `*.yml` files.
If multiple YAMLs are found, it lists them and prompts you to choose one. In non-interactive runs,
it defaults to the first YAML found. 

Example prompt (selecting Pyrenees):

```
[PoNHy] Config YAMLs found:
	1) ponhy_config_california.yaml
	2) ponhy_config_pyrenees.yaml
Select YAML to use (number): 2
```

### Rules

- Booleans: `true/false`.
- Strings: use quotes for paths and names.
- Arrays: use YAML lists: `[1, 2, 3]` or multi-line lists.
- Nested blocks: e.g. `MC_NO_SATURATION_CONFIG:` define grouped settings.

### Base directory and data folders

- `BASE_DIR: "auto"` uses the folder where you run `ponhy.py`.
- You can also set `BASE_DIR` to a fixed path if you prefer hardcoding it.
- After loading the YAML, PoNHy searches inside `BASE_DIR` for folders that start with `Data` (case-insensitive, e.g. `Data_Pyrenees`, `data_pyrenees`) and prompts you to pick one if there are multiple.

### Data file paths

- In the YAML, you can provide **only the filename** (recommended), e.g. `Ext_Topo.txt`.
- Those filenames are resolved inside the selected `Data*` folder.
- If you want to hardcode, you can also provide an absolute path or a path relative to `BASE_DIR`.

### Optional sections via run flags

- If `RUN_INVERSION: false`, **all inversion parameters may be omitted**.
- If `RUN_H2_QUANTIFICATION: false`, **all quantification parameters may be omitted**.
- If a routine is enabled (`true`), its parameters are required.

---

## Running the code

From the `PoNHy` folder, run:

```bash
python ponhy.py
```

Outputs are saved in a timestamped folder inside the resolved `BASE_DIR`, for example:

```
Results_Inversion_YYYYMMDD_HHMMSS/
Results_GenerationYYYYMMDD_HHMMSS/
```

---

## Adding a new region (different input data)

To run PoNHy on **another area** (e.g. Southern Africa, Australia) instead of Pyrenees or California, you change the **input data** and a **config file**. The code in [RodolfoChristiansen/PoNHy](https://github.com/RodolfoChristiansen/PoNHy) does not need to be edited; only data files and YAML config.

### Reference: Pyrenees raw data to replicate

**Everything in `Data_Pyrenees/` is input data.** PoNHy never writes into any `Data_*` folder; all outputs go to timestamped folders (`Results_Inversion_*`, `Results_Generation*`). The files below are the reference formats to replicate for other areas. All paths are under the repo root.

| Purpose | Pyrenees file | Format |
|--------|----------------|--------|
| Topography | `Data_Pyrenees/Ext_Topo.txt` | Tab-separated, no header: X, Y, Z |
| Gravity | `Data_Pyrenees/Ext_Grav.txt` | Tab-separated, no header: X, Y, Z, gravity |
| Magnetic | `Data_Pyrenees/Ext_Mag_down.txt` | Tab-separated, no header: X, Y, Z, magnetic |
| Initial geology model | `Data_Pyrenees/Updated_Geology_Pyrenees.csv` | CSV header: X,Y,Z,Density,Susceptibility |
| Temperature (H₂) | `Data_Pyrenees/temperature_70.csv` | CSV header: X,Y,Z,Temperature |
| Serpentinite density | `Data_Pyrenees/Density_Serpentinite.csv` | CSV (from inversion or prior) |
| Serpentinite susceptibility | `Data_Pyrenees/Magsus_Serpentinite.csv` | CSV (from inversion or prior) |
| H₂ thermodynamic DB | `Data_Pyrenees/DB0/` | Zarr datasets (HZ1, LH1); can copy from repo |

**California** uses the same layout with different filenames: `Data_California/Topo_data.txt`, `Grav_data.txt`, `Mag_data_down.txt`, `Initial_geology_small_20.csv`, `Temperature_20.csv`, plus `Density_Serpentinite.csv`, `Magsus_Serpentinite.csv`, and `DB0/`. Match column order and units to the Pyrenees files when building data for a new region.

### 1. Get or prepare data for your region

You need the following in the **same grid** (same X,Y points for every file; no gaps). Coordinates in **meters (UTM)** for your region's UTM zone.

| File type | Format | Columns (order matters) |
|-----------|--------|-------------------------|
| **Topography** | Tab-separated, **no header** | `X` `Y` `Z` (elevation) |
| **Gravity** | Tab-separated, no header | `X` `Y` `Z` (or topo) `gravity_value` (last column is used) |
| **Magnetic** | Tab-separated, no header | `X` `Y` `Z` (or topo) `magnetic_value` (last column is used) |
| **Initial model** (optional) | CSV, **with header** | `X,Y,Z,Density,Susceptibility` (one row per cell; density/susc can be contrasts) |
| **Temperature** (for H₂) | CSV, with header | `X,Y,Z,Temperature` |
| **H₂ DB** (for H₂) | Same as in `Data_Pyrenees/DB0` | Copy from repo or provide equivalent |

**Where to source input data (public/free):**

- **Gravity:** [ICGEM](https://icgem.gfz-potsdam.de/) (GFZ Potsdam) — global gravity field models (e.g. EIGEN-6C4); calculation service for grids. [BGI](https://bgi.obs-mip.fr/) and national survey archives often provide regional gravity.
- **Magnetic:** [EMAG2 / EMAG2v3](https://ngdc.noaa.gov/geomag/emag2.html) (NOAA NCEI) — global 2-arc-minute magnetic anomaly grid; [downloads](https://ncei.noaa.gov/products/earth-magnetic-model-anomaly-grid-2/download-data) (GeoTIFF, CSV, etc.). Resample to your grid and convert to the same units as the Pyrenees magnetic file.
- **Topography / DEM:** [SRTM](https://www.earthdata.nasa.gov/sensors/srtm) (NASA Earthdata, 1″ or 3″ global), [USGS EarthExplorer](https://earthexplorer.usgs.gov/), [Copernicus DEM](https://spacedata.copernicus.eu/). Export elevation on a regular X,Y grid in UTM.
- **Temperature at depth:** Often from geothermal or lithospheric models (region-specific); can use simple conductive gradients or public geothermal datasets where available.
- **Initial geology model:** From geological maps, prior inversions, or literature; build a CSV with X,Y,Z and density/susceptibility per unit.

Resample all datasets to one common X,Y grid (and Z if needed) in UTM, then export in the column order above.

**Public data with similar (output-style) results:**

- [USGS Geologic Hydrogen Prospectivity](https://www.usgs.gov/data/data-release-prospectivity-mapping-geologic-hydrogen) — raster and tables for hydrogen accumulation potential in the conterminous US (CC0); not PoNHy output but comparable "H₂ potential" mapping.
- [EGI Natural Hydrogen Knowledge Platform](https://egi.utah.edu/natural-hydrogen/) — GIS of natural H₂ sites and geological context.
- PoNHy itself does not publish a central repository of result datasets; runs are local. Sharing outputs would mean publishing your own result CSVs/GeoJSON (e.g. via Zenodo, your repo, or a data portal).

### 2. Create a new data folder

Inside the PoNHy repo (same level as `Data_Pyrenees`), create e.g. `Data_MyRegion`. Put your files there. You can use any filenames (e.g. `Topo_SA.txt`, `Grav_SA.txt`).

### 3. Create a new config YAML

Copy `ponhy_config_pyrenees.yaml` (or `ponhy_config_california.yaml`) to e.g. `ponhy_config_myregion.yaml`. Edit at least:

- **Paths:** `TOPO_FILE`, `GRAV_FILE`, `MAG_FILE`, `INITIAL_MODEL` (if used), `TEMPERATURE_FILE`, `DENSITY_FILE`, `MAGSUS_FILE` — set to your filenames inside `Data_MyRegion`.
- **Mesh:** `DX`, `DY`, `DZ`, `DEPTH_CORE`, `EXPANSION_TYPE`, `ORIGINAL_X`, `ORIGINAL_Y`, `DOMINANT_SIDE` to match your grid extent and expansion.
- **Geophysics:** `INCLINATION`, `DECLINATION`, `STRENGTH` — use the Earth's field for your region (e.g. [IGRF](https://www.ngdc.noaa.gov/IAGA/vmod/igrf.html) or a calculator for your lat/lon).
- **Petrophysics / units:** Adjust `UNIT_LABELS`, `UNIT_DENS_*`, `UNIT_MAGSUS_*`, `SERPENTINITE_LABEL`, etc. if your geology differs.

### 4. Run PoNHy

From the PoNHy folder run `./run_ponhy.sh` (or `python ponhy.py`). When prompted, select your new config (e.g. `ponhy_config_myregion.yaml`) and your new data folder (e.g. `Data_MyRegion`). The workflow then uses your input data and writes results to a new timestamped folder.

### 5. Export to GeoJSON and map

Use `export_to_geojson.py` with the **UTM zone** for your region (e.g. Southern Africa often zone 34–36; use `--utm 35` for zone 35N; southern hemisphere uses different EPSG codes — if you need that, the export script may need a small change to use the southern-hemisphere UTM EPSG). Then point `mapbox_map.html` at the new GeoJSON files or replace the existing ones.

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
- With the current configuration, gravity, magnetic, and topography points must align on the same grid with no gaps. You can generate a common point set and sample each dataset onto it to ensure coverage.

### Reproducibility

- `SEED` + `USE_GLOBAL_SEED: true` enforce a **single global seed**.
- Individual sampling seeds are not used when global seed is enabled.

---

## Troubleshooting

- **Missing file**: check paths in `ponhy_config.yaml`. If the initial model is missing, check that the chosen Data folder matches your config (e.g. Pyrenees config → Data_Pyrenees, California config → Data_California).
- **Config error**: PoNHy prints the invalid key or missing section and exits.
- **Slow runs**: reduce Monte Carlo iterations or set `N_CORES` to a smaller value.

---

## Contact

Developed by **Rodolfo Christiansen (LIAG)**  
Contact: rodo_christiansen@hotmail.com - Rodolfo.Christiansen@liag-institut.de
