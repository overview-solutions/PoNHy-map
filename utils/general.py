from __future__ import annotations

import csv
import math
import os
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde


def load_h2_production_database(dsn_db, rock_codes=None, ftype="zarr"):
    """Load hydrogen production databases from Zarr into xarray objects."""
    if rock_codes is None:
        rock_codes = ["HZ1", "LH1"]
    output = {}

    for code in rock_codes:
        folder = os.path.join(dsn_db, f"{code}_all")
        path = os.path.join(folder, f"{code}_xarray_glpro_all.{ftype}")
        ds = xr.open_zarr(path)

        if isinstance(ds, xr.Dataset) and len(ds.data_vars) > 1:
            da = ds.to_array(dim="fields")
        else:
            varnames = list(ds.data_vars) if isinstance(ds, xr.Dataset) else []
            if varnames:
                da = ds[varnames[0]]
            else:
                da = ds
            if "fields" not in da.dims:
                da = da.expand_dims(fields=["__xarray_dataarray_variable__"])

        output[code] = da
    return output


def parse_temperature_range(temp_range: str) -> Tuple[float, float]:
    """Split a "lo_hi" temperature range string into floats."""
    lo, hi = temp_range.split("_")
    return float(lo), float(hi)


def sort_temperature_ranges(ranges: Sequence[str]) -> List[str]:
    """Return ranges sorted by lower bound."""
    return sorted(ranges, key=lambda tr: parse_temperature_range(tr)[0])


def convert_h2_mol_to_kg(mol_h2):
    """Convert hydrogen from moles to kilograms."""
    kg_h2 = mol_h2 * 0.002016
    return kg_h2


def extract_column(col_name, stat="mean", *, stats_mc_saturation=None, temp_ranges_plot=None):
    """Return per-temperature values for a given column/stat from stats_mc_saturation."""
    if stats_mc_saturation is None or not temp_ranges_plot:
        return []

    col_multi = (col_name, stat)
    values = []
    if col_multi in stats_mc_saturation.columns:
        for tr in temp_ranges_plot:
            values.append(stats_mc_saturation.at[tr, col_multi])
        return values
    if col_name in stats_mc_saturation.columns:
        for tr in temp_ranges_plot:
            values.append(stats_mc_saturation.at[tr, col_name])
        return values
    for tr in temp_ranges_plot:
        try:
            val = stats_mc_saturation.loc[tr][col_name]
            if isinstance(val, dict) and stat in val:
                values.append(val[stat])
            else:
                values.append(val)
        except Exception:
            values.append(0.0)
    return values


def save_saturation_csv(stats_mc_saturation, mean_pressure_ranges, results_path):
    """Write the Monte Carlo saturation summary to a CSV file."""
    sat_csv_path = os.path.join(results_path, "hydrogen_generation_summary_with_saturation.csv")
    if stats_mc_saturation.empty:
        print("Hydrogen generation summary with saturation was not generated (empty results).")
        return
    sat_headers = [
        "Temp",
        "Flow Max",
        "Diff H2O",
        "Frac H2O",
        "H2O abs",
        "Solub.",
        "Saturation",
        "Eff",
        "H₂ Diss",
        "H₂ Gas",
        "H₂ Total",
        "Std H₂",
        "Rocks",
        "W/R",
    ]

    sat_units = [
        "[°C]",
        "[kg/day]",
        "[kg/day]",
        "[kg/day]",
        "[kg/day]",
        "[mol/kg]",
        "[mol/kg]",
        "[%]",
        "[tons/yr]",
        "[tons/yr]",
        "[tons/yr]",
        "[tons/yr]",
        "[kg/day]",
        "[-]",
    ]
    zero_priority_cols = {
        "Flow target [kg/day]",
        "daily_diffused_H2O [kg/day]",
        "daily_fractured_H2O [kg/day]",
    }

    def get_stat(row, col, stat="mean"):
        try:
            data = row[col]
            val = data[stat] if isinstance(data, pd.Series) else data
        except Exception:
            return 0.0 if col in zero_priority_cols and stat == "mean" else "-"
        if isinstance(val, (int, float)) and np.isfinite(val):
            return val
        return 0.0 if col in zero_priority_cols and stat == "mean" else "-"

    sat_rows = []

    all_ranges = (
        sorted(mean_pressure_ranges.keys(), key=lambda x: float(x.split("_")[0]))
        if mean_pressure_ranges
        else list(stats_mc_saturation.index)
    )

    for temp_range in all_ranges:
        if temp_range in stats_mc_saturation.index:
            r = stats_mc_saturation.loc[temp_range]
        else:
            if isinstance(stats_mc_saturation.columns, pd.MultiIndex):
                filler = {}
                for col_name in stats_mc_saturation.columns.get_level_values(0).unique():
                    filler[(col_name, "mean")] = 0.0 if col_name in zero_priority_cols else None
                    filler[(col_name, "std")] = None
                r = pd.Series(filler)
            else:
                filler = {
                    col_name: (0.0 if col_name in zero_priority_cols else None)
                    for col_name in stats_mc_saturation.columns
                }
                r = pd.Series(filler)

        sat_rows.append(
            [
                temp_range,
                get_stat(r, "Flow target [kg/day]"),
                get_stat(r, "daily_diffused_H2O [kg/day]"),
                get_stat(r, "daily_fractured_H2O [kg/day]"),
                get_stat(r, "H2O absorbed [kg/day]"),
                get_stat(r, "Solubility [mol/kg]"),
                get_stat(r, "Saturation [mol/kg]"),
                get_stat(r, "Efficiency [%]"),
                get_stat(r, "H2 dissolved [tons]"),
                get_stat(r, "H2 gaseous [tons]"),
                get_stat(r, "H2 total [tons]"),
                get_stat(r, "H2 total [tons]", stat="std"),
                get_stat(r, "Rocks [kg/day]"),
                get_stat(r, "W/R ratio [-]"),
            ]
        )

    with open(sat_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(sat_headers)
        writer.writerow(sat_units)
        writer.writerows(sat_rows)

        sat_desc = stats_mc_saturation.attrs.get("column_descriptions", [])
        if sat_desc:
            writer.writerow([])
            writer.writerow(["# Column meanings:"])
            for desc in sat_desc:
                writer.writerow([f"# {desc}"])
    print(f"Hydrogen generation summary with saturation saved to: {sat_csv_path}")
    return sat_csv_path


def build_production_rate_volumetric_dict(
    thermo_data,
    lithology_code,
    waterrockratio,
    temperature_ranges,
    mean_pressure_ranges,
    thermo_lookup=None,
):
    """Build average H₂ production per temperature range for a lithology and W/R ratio."""
    if thermo_lookup is None:
        thermo_lookup = build_thermo_lookup_by_range(
            thermo_data,
            lithology_code,
            waterrockratio,
            temperature_ranges,
            mean_pressure_ranges,
        )
    rates = []
    for tr in temperature_ranges:
        stats = thermo_lookup.get(tr) if thermo_lookup else None
        if stats is None:
            mol_kg_h2 = 0.0
        else:
            mol_kg_h2 = stats["mol_kg_h2"]
        rates.append((tr, mol_kg_h2))
    return {f"w/r:{waterrockratio}": rates}


def weighted_average_rates(closest, waterrockratio, production_rate_volumetric, temp_bins):
    """Interpolate volumetric production rates between nearby water/rock ratios."""
    averaged_rates = {temp_range: 0.0 for temp_range in temp_bins}
    rates_closest_dicts = []
    for w_r in closest:
        key = f"w/r:{w_r}"
        if key in production_rate_volumetric:
            rates_list = production_rate_volumetric[key]
            rates_dict = {temp_range: rate for temp_range, rate in rates_list}
            rates_closest_dicts.append(rates_dict)

    exact_match_key = f"w/r:{waterrockratio}"

    if exact_match_key in production_rate_volumetric:
        matched_rates_list = production_rate_volumetric[exact_match_key]
        matched_rates_dict = {temp_range: rate for temp_range, rate in matched_rates_list}
        return matched_rates_dict
    total_diff = abs(closest[0] - closest[1])

    if total_diff == 0:
        weights = [0.5, 0.5]
    else:
        weights = [abs(closest[1] - waterrockratio) / total_diff, abs(closest[0] - waterrockratio) / total_diff]

    for temp_range in averaged_rates:
        temp_rates = [rates_dict.get(temp_range, 0) for rates_dict in rates_closest_dicts]
        if sum(weights) > 0:
            averaged_rates[temp_range] = sum(weight * rate for weight, rate in zip(weights, temp_rates)) / sum(weights)

    return averaged_rates


def compute_mean_lithostatic_pressure_by_range(
    temperature_mesh,
    xyz_mesh_temperature,
    temperature_ranges,
    *,
    density_litho,
    gravity,
):
    """Calculate mean lithostatic pressure for each temperature range."""
    surface_level = np.max(xyz_mesh_temperature[:, 2])
    relative_depths = surface_level - xyz_mesh_temperature[:, 2]

    mean_pressure_ranges = {}
    mean_depths = {}
    for temp_range in temperature_ranges.keys():
        lo, hi = map(float, temp_range.split("_"))
        idx = np.where((temperature_mesh >= lo) & (temperature_mesh < hi))[0]

        if idx.size > 0:
            mean_depth = np.mean(relative_depths[idx])
            pressure_litho = density_litho * gravity * mean_depth
            pressure_hydro = 1000 * gravity * mean_depth

            if mean_depth <= 3000:
                mean_pressure_pa = pressure_hydro
            elif mean_depth <= 7500:
                f = (mean_depth - 3000) / (7500 - 3000)
                start = pressure_hydro
                end = 1.05 * pressure_hydro
                mean_pressure_pa = (1 - f) * start + f * end
            elif mean_depth <= 15000:
                f = (mean_depth - 7500) / (15000 - 7500)
                start = 1.05 * pressure_hydro
                end = 1.50 * pressure_hydro
                mean_pressure_pa = (1 - f) * start + f * end
            else:
                f = (mean_depth - 15000) / (20000 - 15000)
                start = 1.50 * pressure_hydro
                end = 0.75 * pressure_litho
                mean_pressure_pa = (1 - f) * start + f * end

            mean_pressure_bar = mean_pressure_pa / 1e5
            mean_pressure_ranges[temp_range] = float(mean_pressure_bar)
            mean_depths[temp_range] = mean_depth
        else:
            mean_pressure_ranges[temp_range] = np.nan
            mean_depths[temp_range] = np.nan

    sorted_keys = sort_temperature_ranges(list(mean_pressure_ranges.keys()))
    pressures = [mean_pressure_ranges[k] for k in sorted_keys]
    depths = [mean_depths[k] for k in sorted_keys]
    depths_arr = np.array(depths)
    litho = (density_litho * gravity * depths_arr) / 1e5
    hydro = (1000 * gravity * depths_arr) / 1e5

    plt.figure(figsize=(6, 5))
    plt.plot(pressures, depths, marker="o", markersize=4, label="Estimated Pressure", color="blue")
    plt.plot(litho, depths, linestyle="--", label="Lithostatic Pressure", color="red")
    plt.plot(hydro, depths, linestyle="--", label="Hydrostatic Pressure", color="green")
    plt.xlabel("Pressure [bar]")
    plt.ylabel("Depth [m]")
    plt.title("Pressure vs Depth per Temperature Range")
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return mean_pressure_ranges, mean_depths


def build_thermo_lookup_by_range(thermo_data, lithology_code, waterrockratio, temperature_ranges, mean_pressure_ranges):
    """Precompute thermo-data averages per temperature range to avoid repeated xarray selections."""
    da = thermo_data[lithology_code]
    temps = da.coords["temperature"].values
    slice_wr = da.sel(w2r=waterrockratio, method="nearest")
    rho_wr = thermo_data[lithology_code].sel(fields="rho_H2O").sel(w2r=waterrockratio, method="nearest")
    lookup = {}
    field_map = {
        "mol_kg_H2": "mol_kg_h2",
        "n_H2": "n_h2",
        "V_H2": "v_h2",
        "n_H2O_solids": "n_h2o_solids",
    }
    fields_to_average = tuple(field_map.keys())

    for t_range in temperature_ranges:
        lo, hi = map(float, t_range.split("_"))
        temp_mask = (temps >= lo) & (temps < hi)
        p_bar = mean_pressure_ranges.get(t_range, float("nan"))
        if not temp_mask.any() or np.isnan(p_bar):
            lookup[t_range] = None
            continue

        pressure_slice = slice_wr.sel(pressure=p_bar, method="nearest")
        temp_slice = pressure_slice.sel(temperature=temps[temp_mask])

        stats = {"pressure_bar": p_bar, "temperature_mid": (lo + hi) / 2}
        for field in fields_to_average:
            out_key = field_map[field]
            stats[out_key] = float(temp_slice.sel(fields=field).mean(dim="temperature").values)

        rho_sel = rho_wr.sel(pressure=p_bar, method="nearest").sel(
            temperature=stats["temperature_mid"], method="nearest"
        )
        stats["rho_h2o"] = float(rho_sel.values) * 1000.0
        lookup[t_range] = stats

    return lookup


def compute_h2_solubility_kk_pr(p_mpa, t_c):
    """Compute H₂ solubility in pure water using KK + Peng-Robinson fugacity."""
    p_mpa_arr = np.asarray(p_mpa, dtype=float)
    t_c_arr = np.asarray(t_c, dtype=float)
    p_pa, t_k = np.broadcast_arrays(p_mpa_arr * 1e6, t_c_arr + 273.15)

    p0 = 1.0e5
    r_gas = 8.314462618
    v_inf = 1.7e-5
    m0 = 8.0e-4

    t_crit = 33.19
    p_crit = 1.293e6
    omega = -0.216

    kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
    t_r = t_k / t_crit
    alpha = (1 + kappa * (1 - np.sqrt(t_r))) ** 2
    a_param = 0.45724 * r_gas**2 * t_crit**2 / p_crit * alpha
    b_param = 0.07780 * r_gas * t_crit / p_crit

    a_term = a_param * p_pa / (r_gas**2 * t_k**2)
    b_term = b_param * p_pa / (r_gas * t_k)

    def _pr_z(a_scalar, b_scalar):
        coeffs = [
            1.0,
            -(1.0 - b_scalar),
            a_scalar - 3.0 * b_scalar**2 - 2.0 * b_scalar,
            -(a_scalar * b_scalar - b_scalar**2 - b_scalar**3),
        ]
        roots = np.roots(coeffs)
        real_roots = roots[np.isreal(roots)].real
        return np.max(real_roots) if real_roots.size else np.nan

    z = np.vectorize(_pr_z, otypes=[float])(a_term, b_term)

    sqrt_2 = np.sqrt(2.0)
    ln_phi = np.zeros_like(z)
    mask = (p_pa > 0) & (b_term > 0)
    if np.any(mask):
        z_mask = z[mask]
        b_mask = b_term[mask]
        a_mask = a_term[mask]
        z_minus_b = np.maximum(z_mask - b_mask, 1e-16)
        log_term = np.log(
            (z_mask + (1 + sqrt_2) * b_mask) / (z_mask + (1 - sqrt_2) * b_mask)
        )
        ln_phi[mask] = z_mask - 1 - np.log(z_minus_b) - (a_mask / (2 * sqrt_2 * b_mask)) * log_term

    phi = np.exp(ln_phi)
    solubility = m0 * (phi * p_pa / p0) * np.exp(-v_inf * (p_pa - p0) / (r_gas * t_k))
    solubility = np.where(p_pa > 0, solubility, 0.0)

    return float(solubility) if np.ndim(solubility) == 0 else solubility


def compute_serpentinization_degree(density_export_cells, magsus_export_cells, mask_not_nan, x_points, y_points, values):
    """Estimate serpentinization degree from density and magnetic susceptibility."""
    filtered_density = density_export_cells[mask_not_nan]
    filtered_density_adjusted = 2.67 - filtered_density
    filtered_magsus = magsus_export_cells[mask_not_nan]
    finite_mask = np.isfinite(filtered_density_adjusted) & np.isfinite(filtered_magsus)
    filtered_density_adjusted = filtered_density_adjusted[finite_mask]
    filtered_magsus = filtered_magsus[finite_mask]

    positive_filtered_magsus = filtered_magsus[filtered_magsus > 0]

    density_kde = gaussian_kde(filtered_density_adjusted)
    x_dens = np.linspace(min(filtered_density_adjusted), max(filtered_density_adjusted), 1000)
    pdf_dens = density_kde(x_dens)
    mode_density = x_dens[np.argmax(pdf_dens)]

    magsus_kde = gaussian_kde(positive_filtered_magsus)
    x_magsus = np.linspace(min(positive_filtered_magsus), max(positive_filtered_magsus), 1000)
    pdf_magsus = magsus_kde(x_magsus)
    mode_magsus = x_magsus[np.argmax(pdf_magsus)]

    min_density, max_density = x_points.min(), x_points.max()
    min_magsus, max_magsus = y_points.min(), y_points.max()

    norm_densities = (x_points - min_density) / (max_density - min_density)
    norm_magsus_data = (y_points - min_magsus) / (max_magsus - min_magsus)
    norm_input_density = (mode_density - min_density) / (max_density - min_density)
    norm_input_magsus = (mode_magsus - min_magsus) / (max_magsus - min_magsus)

    distances = np.sqrt(
        (norm_densities - norm_input_density) ** 2 + (norm_magsus_data - norm_input_magsus) ** 2
    )
    closest_index = np.argmin(distances)
    serpentinization_degree = values[closest_index]

    return mode_density, mode_magsus, serpentinization_degree


def plot_serpentinization_heatmap(
    density_export_cells,
    magsus_export_cells,
    mask_not_nan,
    mode_density,
    mode_magsus,
    x_points,
    y_points,
    values,
    *,
    results_path,
    save_svg: bool = True,
):
    """Create a density-magnetic susceptibility heatmap annotated with serpentinization degree."""
    filtered_density = density_export_cells[mask_not_nan]
    filtered_density_adjusted = 2.67 - filtered_density
    filtered_magsus = magsus_export_cells[mask_not_nan]
    plt.rcParams.update({"font.size": 12})

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor("none")

    hb = ax.hexbin(
        filtered_density_adjusted,
        filtered_magsus,
        gridsize=60,
        cmap="viridis",
        bins="log",
        mincnt=1,
    )
    cbar = plt.colorbar(hb, ax=ax)
    cbar.set_label("Log Frequency")

    ax.set_xlim(2.4, 3.3)
    ax.set_ylim(-0.01, 0.1)
    ax.set_xlabel("Density [g/cc]")
    ax.set_ylabel("Magnetic Susceptibility [SI]")
    ax.set_title("Overall Serpentinization Degree based on geophysical inversion")
    ax.scatter([mode_density], [mode_magsus], color="red", zorder=2, label="Mode Density & Magsus")

    ax.plot(x_points, y_points, "-o", color="black", markersize=2)
    for x_val, y_val, value in zip(x_points, y_points, values):
        if value % 10 == 0:
            ax.text(x_val + 0.01, y_val + 0.00005, f"{value}", color="black", ha="center", va="bottom")

    file_name_png = os.path.join(results_path, "serpentinization_heat_map.png")
    file_name_svg = os.path.join(results_path, "serpentinization_heat_map.svg")
    fig.savefig(file_name_png)
    if save_svg:
        fig.savefig(file_name_svg)
    plt.show()
    plt.close(fig)
    return file_name_png, file_name_svg


def compute_serpentinization_front_velocities(production_rate_volumetric, waterrockratio, t_ref_range, v_ref_synthetic):
    """Estimate serpentinization front velocities from volumetric production rates."""
    print("\n" + " Estimated front velocities ".center(150, "-") + "\n")
    wr_key = f"w/r:{waterrockratio}"
    rates_list = production_rate_volumetric[wr_key]
    rates_dict = {tr: rate for tr, rate in rates_list}

    t_ref_mid = sum(map(int, t_ref_range.split("_"))) / 2

    valid_bins = [
        (tr, sum(map(int, tr.split("_"))) / 2, rate) for tr, rate in rates_dict.items() if rate > 0
    ]

    if not valid_bins:
        raise ValueError(" No non-zero production rates found.")

    tr_ref, t_mid_ref, r_ref_model = min(valid_bins, key=lambda x: abs(x[1] - t_ref_mid))

    if tr_ref != t_ref_range:
        print(f"Reference bin '{t_ref_range}' has zero rate. Using closest valid: '{tr_ref}' ({t_mid_ref:.1f} °C)")
    else:
        print(f"Using reference bin: '{tr_ref}' ({t_mid_ref:.1f} °C)")

    def compute_velocity(r_new, r_ref, v_ref):
        return 0.0 if r_ref == 0 else v_ref * (r_new / r_ref)

    serpentinization_front_velocities = {}
    print("\n{:^20} {:^20}".format("Temp Range [°C]", "Serp Vel [cm/day]"))
    print("{:^50}".format("-" * 50))
    for tr, rate in sorted(rates_dict.items()):
        v_avg = compute_velocity(rate, r_ref_model, v_ref_synthetic)
        serpentinization_front_velocities[tr] = [("avg", v_avg)]
        print(f"{tr:^20} {v_avg:>20.2e}")
    return serpentinization_front_velocities
