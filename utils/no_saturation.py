from __future__ import annotations

import csv
import os
import sys
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.general import (
    build_thermo_lookup_by_range,
    compute_h2_solubility_kk_pr,
    convert_h2_mol_to_kg,
    weighted_average_rates,
)
from utils.helpers import (
    _compute_mc_chunk_size,
    _get_progress_bar,
    _sample_unit_hypercube,
    _scale_samples_to_ranges,
)
from utils.reporting import print_no_saturation_report, print_no_saturation_summary_report


_NO_SAT_WORKER_CONTEXT: Optional[Dict[str, Any]] = None


def run_no_saturation_workflow(
    temperature_ranges,
    serpentinization_corrections,
    serpentinization_front_velocities,
    density_serpentinite,
    production_rate_volumetric,
    waterrockratio,
    volume_at_temperature,
    mean_pressure_ranges,
    thermo_data,
    lithology_code,
    porosity_front,
    results_path,
    years,
    dist_x,
    dist_y,
    dist_z,
    *,
    mc_config,
    temp_bins,
    molar_mass_h2,
    molar_mass_h2o,
    thermo_lookup=None,
    surface_area_per_km3=None,
    run_prints=True,
):
    """End-to-end helper for the no-saturation pathway."""

    if surface_area_per_km3 is None:
        original_volume_m3 = 1000**3
        voxel_volume_m3 = dist_x * dist_y * dist_z
        total_voxels = original_volume_m3 / voxel_volume_m3
        voxel_surface_area = 2 * (dist_x * dist_y + dist_x * dist_z + dist_y * dist_z)
        surface_area_per_km3 = voxel_surface_area * total_voxels

    results_temp_volumetric, kg_rocks_dict, std_results_temp_volumetric, iter_total_samples = (
        compute_h2_production_no_saturation(
            temperature_ranges=temperature_ranges,
            serpentinization_corrections=serpentinization_corrections,
            serpentinization_front_velocities=serpentinization_front_velocities,
            density_serpentinite=density_serpentinite,
            production_rate_volumetric=production_rate_volumetric,
            waterrockratio=waterrockratio,
            volume_at_temperature=volume_at_temperature,
            surface_area_per_km3=surface_area_per_km3,
            mc_config=mc_config,
            return_samples=True,
            show_progress=bool(mc_config.get("show_progress", True)),
        )
    )

    if run_prints:
        summarize_no_saturation_production(results_temp_volumetric)

    total_kg_rocks = sum(kg_rocks_dict.values())

    iter_totals_tons = None
    if isinstance(iter_total_samples, np.ndarray) and iter_total_samples.size > 0:
        try:
            iter_totals_tons = ((np.asarray(iter_total_samples, dtype=float) * 365.0) / 1000.0).tolist()
        except Exception:
            iter_totals_tons = None

    df_no_saturation = compute_saturation_from_volumetric(
        results_temp_volumetric,
        kg_rocks_dict,
        mean_pressure_ranges,
        waterrockratio,
        thermo_data,
        lithology_code,
        porosity_front,
        density_serpentinite,
        std_results_temp_volumetric,
        temp_bins=temp_bins,
        molar_mass_h2=molar_mass_h2,
        molar_mass_h2o=molar_mass_h2o,
        thermo_lookup=thermo_lookup,
    )

    df_no_saturation, total_tons_no_sat, no_sat_csv_path = build_no_saturation_summary(
        df_no_saturation=df_no_saturation,
        std_results_temp_volumetric=std_results_temp_volumetric,
        results_path=results_path,
        total_kg_rocks=total_kg_rocks,
        years=years,
        mc_config=mc_config,
        iter_totals=iter_totals_tons,
    )

    if run_prints:
        print_no_saturation_report(df_no_saturation)
        print_no_saturation_summary_report(
            df_no_saturation=df_no_saturation,
            total_tons_no_sat=total_tons_no_sat,
            total_kg_rocks=total_kg_rocks,
            years=years,
            no_sat_csv_path=no_sat_csv_path,
        )

    return (
        results_temp_volumetric,
        kg_rocks_dict,
        std_results_temp_volumetric,
        df_no_saturation,
        total_tons_no_sat,
        no_sat_csv_path,
        total_kg_rocks,
    )


def compute_h2_production_no_saturation(
    temperature_ranges,
    serpentinization_corrections,
    serpentinization_front_velocities,
    density_serpentinite,
    production_rate_volumetric,
    waterrockratio,
    *,
    volume_at_temperature=None,
    surface_area_per_km3=None,
    deterministic_factors: Optional[Dict[str, float]] = None,
    return_samples: bool = False,
    factor_ranges_override: Optional[Dict[str, Tuple[float, float]]] = None,
    sampling_seed_override: Optional[int] = None,
    show_progress: bool = False,
    progress_desc: Optional[str] = None,
    progress_position: int = 0,
    progress_leave: bool = True,
    mc_config=None,
):
    """Compute theoretical maximum H₂ production using volumetric rates (no saturation)."""

    mc_cfg = mc_config or {}
    w_r_values = [float(wr.split(":")[1]) for wr in production_rate_volumetric]
    w_r_values.sort()
    closest_wr = sorted(w_r_values, key=lambda x: abs(x - waterrockratio))[:2]
    averaged_rates = weighted_average_rates(closest_wr, waterrockratio, production_rate_volumetric, temp_bins=temperature_ranges)

    results_temp_volumetric = {}
    kg_rocks_dict = {}
    std_results_temp_volumetric = {}
    iter_total_samples: Optional[np.ndarray] = None

    sampling_mode = mc_cfg.get("sampling", "uniform").lower()
    sampling_seed = sampling_seed_override if sampling_seed_override is not None else mc_cfg.get("sampling_seed", None)

    def _as_range_tuple(value: Optional[Tuple[float, float]]) -> Tuple[float, float]:
        if value is None:
            return (1.0, 1.0)
        lo, hi = value
        return (float(lo), float(hi))

    range_specs: List[Tuple[str, Tuple[float, float]]] = [
        ("v_ref_range", _as_range_tuple(mc_cfg.get("v_ref_range", (1.0, 1.0)))),
        ("prod_rate_range", _as_range_tuple(mc_cfg.get("prod_rate_range", (1.0, 1.0)))),
        ("volume_range", _as_range_tuple(mc_cfg.get("volume_range", (1.0, 1.0)))),
        ("serp_correction_range", _as_range_tuple(mc_cfg.get("serp_correction_range", (1.0, 1.0)))),
        ("surface_area_range", _as_range_tuple(mc_cfg.get("surface_area_range", (1.0, 1.0)))),
    ]

    if factor_ranges_override:
        range_specs = [
            (
                name,
                _as_range_tuple(factor_ranges_override.get(name, default)),
            )
            for name, default in range_specs
        ]
    n_iter = mc_cfg.get("n_iter", 0)
    if n_iter <= 0:
        return results_temp_volumetric, kg_rocks_dict, std_results_temp_volumetric, iter_total_samples

    range_count = max(1, len(temperature_ranges))
    chunk_size = _compute_mc_chunk_size(n_iter, range_count)
    progress_total = float(n_iter)
    progress_label = progress_desc or "Monte Carlo (no saturation)"
    enable_progress = bool(show_progress and progress_total > 0)
    progress_ctx = (
        _get_progress_bar(
            total=progress_total,
            desc=progress_label,
            file=sys.stdout,
            position=progress_position,
            leave=progress_leave,
        )
        if enable_progress
        else nullcontext()
    )

    if return_samples:
        iter_total_samples = np.zeros(n_iter, dtype=float)

    def _sample_factors(n: int) -> np.ndarray:
        dim = len(range_specs)
        unit = _sample_unit_hypercube(n, dim, sampling_mode, sampling_seed)
        return _scale_samples_to_ranges(unit, range_specs)

    d_factors = deterministic_factors or {}
    f_v = d_factors.get("v_ref", 1.0)
    f_prod = d_factors.get("prod_rate", 1.0)
    f_vol = d_factors.get("volume", 1.0)
    f_serp = d_factors.get("serp_correction", 1.0)
    f_sa = d_factors.get("surface_area", 1.0)

    def _base_inputs(temp_range: str):
        serp_vel_entries = serpentinization_front_velocities.get(temp_range, [])
        serp_vel_raw = next((val for label, val in serp_vel_entries if label == "avg" and val is not None), 0.0)
        serp_vel = serp_vel_raw * f_v

        serp_correction_raw = serpentinization_corrections.get(temp_range, {}).get("avg", 0.0) or 0.0
        serp_correction = serp_correction_raw * f_serp
        production_rate = (averaged_rates.get(temp_range, 0) or 0) * f_prod
        volume = ((volume_at_temperature.get(temp_range, 0) if volume_at_temperature else 0) or 0) * f_vol
        sa_val = (surface_area_per_km3 or 0) * f_sa
        return serp_vel, serp_correction, production_rate, volume, sa_val

    zero_causes = []

    with progress_ctx as pbar:
        for temp_range in temperature_ranges:
            serp_vel, serp_correction, production_rate, volume, sa_val = _base_inputs(temp_range)

            vol_cm3 = sa_val * 1e4 * (serp_vel or 0) * volume
            kg_rocks = vol_cm3 / 1e6 * density_serpentinite * (serp_correction or 0)
            kg_rocks_dict[temp_range] = kg_rocks

            avg_production_rate_vol = kg_rocks * production_rate
            results_temp_volumetric[temp_range] = convert_h2_mol_to_kg(avg_production_rate_vol)

            if results_temp_volumetric[temp_range] <= 0:
                zero_causes.append(
                    {
                        "temp_range": temp_range,
                        "serp_vel": serp_vel,
                        "serp_correction": serp_correction,
                        "production_rate": production_rate,
                        "volume": volume,
                        "surface_area_per_km3": sa_val,
                    }
                )

            factors_full = _sample_factors(n_iter)
            kg_samples = np.empty(n_iter, dtype=float)

            for start in range(0, n_iter, chunk_size):
                end = min(start + chunk_size, n_iter)
                chunk = factors_full[start:end]
                serp_vel_s = serp_vel * chunk[:, 0]
                production_rate_s = production_rate * chunk[:, 1]
                volume_s = volume * chunk[:, 2]
                serp_corr_s = np.clip(serp_correction * chunk[:, 3], 0.0, 1.0)
                surface_area_s = sa_val * chunk[:, 4]

                vol_cm3_s = surface_area_s * 1e4 * serp_vel_s * volume_s
                kg_rocks_s = vol_cm3_s / 1e6 * density_serpentinite * serp_corr_s
                mol_s = kg_rocks_s * production_rate_s
                kg_chunk = convert_h2_mol_to_kg(mol_s)
                kg_samples[start:end] = kg_chunk

                if iter_total_samples is not None:
                    iter_total_samples[start:end] += kg_chunk

                if pbar is not None:
                    increment = (end - start) / range_count
                    pbar.update(increment)

            std_results_temp_volumetric[temp_range] = float(np.std(kg_samples))

        if pbar is not None and pbar.n < progress_total:
            pbar.update(progress_total - pbar.n)

    if len(zero_causes) == len(temperature_ranges):
        print("[Univariate][NoSat][WARN] Base production = 0 in every range. Check!!!")
        for entry in zero_causes[:5]:
            print(
                f"  Range={entry['temp_range']}: serp_vel={entry['serp_vel']}, "
                f"prod_rate={entry['production_rate']}, volume={entry['volume']}, "
                f"serp_corr={entry['serp_correction']}, surface_area_per_km3={entry['surface_area_per_km3']}"
            )

    return results_temp_volumetric, kg_rocks_dict, std_results_temp_volumetric, iter_total_samples


def _init_no_saturation_worker_context(context: Optional[Dict[str, Any]] = None):
    global _NO_SAT_WORKER_CONTEXT
    _NO_SAT_WORKER_CONTEXT = context


def _clear_no_saturation_worker_context():
    _init_no_saturation_worker_context(None)


def _get_no_saturation_worker_context() -> Dict[str, Any]:
    if _NO_SAT_WORKER_CONTEXT is None:
        raise RuntimeError("No-saturation worker context not initialized.")
    return _NO_SAT_WORKER_CONTEXT


def _run_no_saturation_sample(task):
    """Run one no-saturation MC sample and return the annual total."""
    role, param_name, sample_value, deterministic_factors, factor_overrides = task
    context = _get_no_saturation_worker_context()
    mc_kwargs = context["mc_kwargs"]
    sampling_seed = context["sampling_seed"]
    deterministic = deterministic_factors if deterministic_factors else None
    factor_override = factor_overrides if factor_overrides else None
    results_temp_volumetric, _, _, _ = compute_h2_production_no_saturation(
        **mc_kwargs,
        deterministic_factors=deterministic,
        factor_ranges_override=factor_override,
        sampling_seed_override=sampling_seed,
        show_progress=False,
    )
    total_kg_day = sum(v for v in results_temp_volumetric.values() if np.isfinite(v))
    total_tons_year = (total_kg_day * 365.0) / 1000.0
    return role, param_name, sample_value, total_tons_year


def summarize_no_saturation_production(results_temp_volumetric):
    """Summarize hydrogen production rates from the volumetric estimator across temperature ranges."""
    data = []
    for tr, rate in results_temp_volumetric.items():
        if rate and not (isinstance(rate, float) and np.isnan(rate)):
            data.append({"Temperature Range": tr, "Production Rate": rate})

    df = pd.DataFrame(data)
    if df.empty:
        raise ValueError("No production data available.")

    df["LogRate"] = np.log10(df["Production Rate"])
    summary_stats = df.groupby("Temperature Range")["LogRate"].agg(mean="mean", std=lambda x: x.std(ddof=0)).reset_index()
    summary_stats["std"] = summary_stats["std"].fillna(0.0)

    mean_log = summary_stats["mean"]
    std_log = summary_stats["std"]
    lower_log = mean_log - std_log
    upper_log = mean_log + std_log

    avg_kg = convert_h2_mol_to_kg(10 ** mean_log)
    lower_kg = convert_h2_mol_to_kg(10 ** lower_log)
    upper_kg = convert_h2_mol_to_kg(10 ** upper_log)

    summary_df = pd.DataFrame(
        {
            "Temperature Range [°C]": summary_stats["Temperature Range"],
            "Source": "Volumetric",
            "Avg [kg/day]": avg_kg,
            "Lower [kg/day]": lower_kg,
            "Upper [kg/day]": upper_kg,
        }
    )

    summary_df["Avg [kg/day]"] = summary_df["Avg [kg/day]"].apply(
        lambda x: f"{int(round(x))}" if pd.notnull(x) else "NaN"
    )
    summary_df["Lower [kg/day]"] = summary_df["Lower [kg/day]"].apply(
        lambda x: f"{int(round(x))}" if pd.notnull(x) else "NaN"
    )
    summary_df["Upper [kg/day]"] = summary_df["Upper [kg/day]"].apply(
        lambda x: f"{int(round(x))}" if pd.notnull(x) else "NaN"
    )

    return summary_df


def compute_saturation_from_volumetric(
    results_temp_volumetric,
    kg_rocks_dict,
    mean_pressure_ranges,
    waterrockratio,
    thermo_data,
    lithology_code,
    porosity_front,
    density_serpentinite,
    std_results_temp_volumetric=None,
    *,
    temp_bins,
    molar_mass_h2,
    molar_mass_h2o,
    thermo_lookup=None,
):
    """Compute saturation-limited H₂ production from volumetric (no-sat) outputs."""

    temperature_ranges = list(temp_bins)

    if thermo_lookup is None:
        thermo_lookup = build_thermo_lookup_by_range(
            thermo_data,
            lithology_code,
            waterrockratio,
            temperature_ranges,
            mean_pressure_ranges,
        )

    results = []

    for t_range in temperature_ranges:
        if t_range not in kg_rocks_dict or t_range not in results_temp_volumetric or t_range not in mean_pressure_ranges:
            continue

        stats = thermo_lookup.get(t_range)
        if stats is None:
            continue
        rock_mass_kg_day = kg_rocks_dict[t_range]
        lo, hi = map(float, t_range.split("_"))
        p_bar = mean_pressure_ranges[t_range]
        p_mpa = p_bar / 10
        t_c = (lo + hi) / 2
        solubility = compute_h2_solubility_kk_pr(p_mpa, t_c)
        rho_h2o = stats["rho_h2o"]

        volume_rock_m3 = rock_mass_kg_day / density_serpentinite
        volume_water_m3 = volume_rock_m3 * porosity_front
        water_mass_kg_day = volume_water_m3 * rho_h2o

        actual_kg_day = results_temp_volumetric[t_range]
        actual_mol_day = actual_kg_day / molar_mass_h2 if actual_kg_day else 0
        actual_tons_year = actual_kg_day * 365 / 1000

        std_actual_mol_day = 0.0
        if std_results_temp_volumetric is not None and t_range in std_results_temp_volumetric:
            std_kg = std_results_temp_volumetric[t_range]
            std_actual_mol_day = std_kg / molar_mass_h2

        n_h2o_sol_per_kg = stats["n_h2o_solids"]
        mol_absorbed_total = n_h2o_sol_per_kg * rock_mass_kg_day
        h2o_absorbed_kg_day = mol_absorbed_total * molar_mass_h2o

        effective_water_kg = max(0.0, water_mass_kg_day - h2o_absorbed_kg_day)
        dissolvable_mol_day = solubility * effective_water_kg

        saturation_fraction = actual_mol_day / dissolvable_mol_day if dissolvable_mol_day > 0 else np.inf
        h2_gaseous_mol_day = max(0.0, actual_mol_day - dissolvable_mol_day)

        daily_diffused = stats.get("daily_diffused_H2O [kg/day]", 0.0)
        daily_fractured = stats.get("daily_fractured_H2O [kg/day]", 0.0)

        results.append(
            {
                "Temperature Range": t_range,
                "Temperature [°C]": t_c,
                "Pressure [MPa]": p_mpa,
                "Solubility [mol/kg]": solubility,
                "Water [kg/day]": water_mass_kg_day,
                "Effective Water [kg/day]": effective_water_kg,
                "Dissolvable [mol/day]": dissolvable_mol_day,
                "Actual [kg/day]": actual_kg_day,
                "Actual [mol/day]": actual_mol_day,
                "Actual [mol/day] std": std_actual_mol_day,
                "H2 [tons/yr]": actual_tons_year,
                "Absorbed H₂O [mol/day]": mol_absorbed_total,
                "Absorbed H₂O [kg/day]": h2o_absorbed_kg_day,
                "Sat ratio": saturation_fraction,
                "H2 gaseous [mol/day]": h2_gaseous_mol_day,
                "daily_diffused_H2O [kg/day]": daily_diffused,
                "daily_fractured_H2O [kg/day]": daily_fractured,
            }
        )

    df = pd.DataFrame(results).drop_duplicates(subset=["Temperature Range"], keep="first").reset_index(drop=True)

    df.attrs["column_descriptions"] = [
        " Temp         : Temperature interval [°C]",
        " Pressure     : Average pressure [MPa]",
        " Solubility   : Estimated H₂ solubility [mol/kg]",
        " H2O front    : Water at the reaction front [kg/day]",
        " H2O abs      : Water absorbed into the solid phase [kg/day]",
        " H₂ Diss      : Max H₂ that water can dissolve [mol/day]",
        " H₂ Gas       : Gaseous H₂ exceeding saturation limit [mol/day]",
        " Daily H₂     : Daily H₂ production [mol/day] and [kg/day]",
        " Annual H₂    : Annual H₂ production [tons/yr]",
        " Std clip     : Winsorized standard deviation of Annual H₂ [tons/yr] (clipping multiplicative uncertainty to the 1st-99th percentiles)",
        " Sat ratio    : H₂ production / H₂ dissolution capacity [-]",
    ]
    return df


def build_no_saturation_summary(
    df_no_saturation,
    std_results_temp_volumetric,
    results_path,
    total_kg_rocks,
    years,
    *,
    mc_config,
    iter_totals: Optional[Sequence[float]] = None,
):
    """Compute winsorized uncertainty, persist CSV, and return totals for the no-saturation workflow."""
    std_annual_tons_p01_p99_col = []

    for _, row in df_no_saturation.iterrows():
        try:
            base_kg_per_day = float(row.get("Actual [kg/day]", 0.0) or 0.0)
            if base_kg_per_day <= 0.0:
                std_clip_t = 0.0
            else:
                n_clip = 20000
                mode = mc_config.get("sampling", "uniform").lower()
                seed = mc_config.get("sampling_seed", None)
                ranges = [
                    mc_config.get("v_ref_range", (1.0, 1.0)),
                    mc_config.get("prod_rate_range", (1.0, 1.0)),
                    mc_config.get("volume_range", (1.0, 1.0)),
                    mc_config.get("serp_correction_range", (1.0, 1.0)),
                    mc_config.get("surface_area_range", (1.0, 1.0)),
                ]

                def _sample_units(mode_local: str) -> np.ndarray:
                    dim = len(ranges)
                    if mode_local == "sobol":
                        try:
                            from scipy.stats import qmc as _qmc

                            engine = _qmc.Sobol(d=dim, scramble=True, seed=seed)
                            m = int(np.ceil(np.log2(max(1, n_clip))))
                            return engine.random_base2(m=m)[:n_clip, :]
                        except Exception:
                            rng = np.random.default_rng(seed)
                            return rng.random((n_clip, dim))
                    if mode_local == "lhs":
                        try:
                            from scipy.stats import qmc as _qmc

                            engine = _qmc.LatinHypercube(d=dim, seed=seed)
                            return engine.random(n_clip)
                        except Exception:
                            unit = np.zeros((n_clip, dim))
                            rng = np.random.default_rng(seed)
                            for j in range(dim):
                                perm = rng.permutation(n_clip)
                                unit[:, j] = (perm + rng.random(n_clip)) / n_clip
                            return unit
                    rng = np.random.default_rng(seed)
                    return rng.random((n_clip, dim))

                unit = _sample_units(mode)
                scaled = np.empty_like(unit)
                for i, (lo, hi) in enumerate(ranges):
                    scaled[:, i] = lo + (hi - lo) * unit[:, i]
                factors = np.prod(scaled, axis=1)
                lo = np.quantile(factors, 0.01)
                hi = np.quantile(factors, 0.99)
                factors = np.clip(factors, lo, hi)
                std_clip_kg = float(np.std(base_kg_per_day * factors))
                std_clip_t = std_clip_kg * 365.0 / 1000.0
        except Exception:
            std_clip_t = 0.0
        std_annual_tons_p01_p99_col.append(std_clip_t)

    df_no_saturation["Std clip [tons/yr]"] = std_annual_tons_p01_p99_col
    clip_vals = np.asarray(std_annual_tons_p01_p99_col, dtype=float)
    total_std_clip = float(np.sqrt(np.sum(np.square(clip_vals)))) if clip_vals.size else 0.0

    df_no_saturation.attrs["std_total_clip"] = total_std_clip
    df_no_saturation.attrs["std_results_temp_volumetric"] = std_results_temp_volumetric
    if iter_totals is not None:
        try:
            df_no_saturation.attrs["iter_totals"] = list(np.asarray(iter_totals, dtype=float))
        except Exception:
            pass
    no_sat_headers = [
        "Temp",
        "Pressure",
        "Solubility",
        "H2O front",
        "H2O abs",
        "H₂ Diss",
        "H₂ Gas",
        "Daily H₂ (mol)",
        "Daily H₂ (kg)",
        "Annual H₂",
        "Std clip",
        "Sat ratio",
    ]
    no_sat_units = [
        "[°C]",
        "[MPa]",
        "[mol/kg]",
        "[kg/day]",
        "[kg/day]",
        "[mol/day]",
        "[mol/day]",
        "[mol/day]",
        "[kg/day]",
        "[tons/yr]",
        "[tons/yr]",
        "[-]",
    ]
    no_sat_csv_path = os.path.join(results_path, "hydrogen_generation_summary_no_saturation.csv")

    no_sat_rows = []
    for _, row in df_no_saturation.iterrows():
        no_sat_rows.append(
            [
                row["Temperature Range"],
                row["Pressure [MPa]"],
                row["Solubility [mol/kg]"],
                row["Water [kg/day]"],
                row["Absorbed H₂O [kg/day]"],
                row["Dissolvable [mol/day]"],
                row["H2 gaseous [mol/day]"],
                row["Actual [mol/day]"],
                row["Actual [kg/day]"],
                row["H2 [tons/yr]"],
                row.get("Std clip [tons/yr]", 0.0),
                row["Sat ratio"],
            ]
        )

    with open(no_sat_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(no_sat_headers)
        writer.writerow(no_sat_units)
        writer.writerows(no_sat_rows)

        no_sat_desc = df_no_saturation.attrs.get("column_descriptions", [])
        if no_sat_desc:
            writer.writerow([])
            writer.writerow(["# Column meanings:"])
            for desc in no_sat_desc:
                writer.writerow([f"# {desc}"])
    total_tons_no_sat = df_no_saturation["H2 [tons/yr]"].sum()

    return df_no_saturation, total_tons_no_sat, no_sat_csv_path
