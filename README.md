# PoNHy-map

This repo is a fork of [RodolfoChristiansen/PoNHy](https://github.com/RodolfoChristiansen/PoNHy) (Potential for Natural Hydrogen) with changes focused on **running PoNHy reliably** and **mapping results**.

**General changes in this fork:**

- **Run script & deps:** `run_ponhy.sh` (venv + optional CPU limits) and `requirements.txt` for pip users; `packaging` and other deps pinned so the workflow runs without conda.
- **SimPEG v0.24+:** Inversion uses `active_cells` in gravity/magnetic simulations; plotting still uses `ind_active` where required.
- **Stability:** `ponhy_config_light.yaml` for lighter runs (fewer iterations, 1 core, optional analyses off); README notes on avoiding overload.
- **Paths & config:** Absolute paths in config are left unchanged; clearer errors when the chosen Data folder doesn’t match the config (e.g. Pyrenees config → Data_Pyrenees).
- **Non-interactive:** When multiple Data folders exist and stdin isn’t a TTY, the first Data folder is used automatically.
- **GeoJSON & map:** `export_to_geojson.py` converts result CSVs (or topo/gravity inputs) to WGS84 GeoJSON (points, `--polygons`, `--value-ranges`); `index.html` loads density/gravity/topography polygon layers (e.g. on GitHub Pages from the main branch).

For full documentation (quick start, environment, configuration, troubleshooting, contact), see **[README_original.md](README_original.md)**.

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
