from __future__ import annotations

import math
import random
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.general import compute_h2_solubility_kk_pr
from utils.helpers import (
    _as_scalar_value,
    _clamp_worker_count,
    _get_progress_bar,
    _moving_average,
    _sample_unit_hypercube,
    _select_stable_window,
    _trim_trailing_dropoff,
)
from utils.plotting import plot_h2_production_summary
from utils.reporting import print_saturation_monte_carlo_report, print_saturation_summary

_MC_WORKER_CONTEXT: Optional[Dict[str, Any]] = None


def run_saturation_workflow(
    volume_at_temperature,
    df_saturation_table,
    mean_pressure_ranges,
    serpentinization_degree,
    int_fracture_spacing,
    permeability_fractures,
    flow_target,
    production_rate_volumetric,
    years,
    dist_x,
    dist_y,
    dist_z,
    kg_rocks_dict,
    total_kg_rocks,
    total_tons_no_sat,
    results_path,
    *,
    mc_config,
    n_cores,
    seed,
    porosity_front,
    density_serpentinite,
    run_prints=True,
):
    """Wrapper for the full with-saturation workflow (MC -> plots -> reports)."""

    cfg = mc_config or {}
    stats_mc_saturation = run_saturation_monte_carlo(
        n_iter=cfg.get("n_iter", 0),
        volume_at_temperature=volume_at_temperature,
        df_saturation_table=df_saturation_table,
        mean_pressure_ranges=mean_pressure_ranges,
        serpentinization_degree=serpentinization_degree,
        int_fracture_spacing=int_fracture_spacing,
        permeability_fractures=permeability_fractures,
        flow_target=flow_target,
        production_rate_volumetric=production_rate_volumetric,
        years=years,
        dist_x=dist_x,
        dist_y=dist_y,
        dist_z=dist_z,
        kg_rocks_dict=kg_rocks_dict,
        show_progress=True,
        mc_config=cfg,
        n_cores=n_cores,
        seed=seed,
        porosity_front=porosity_front,
        density_serpentinite=density_serpentinite,
    )

    if run_prints:
        print_saturation_monte_carlo_report(stats_mc_saturation, years, flow_target)

    total_tons_sat = 0.0
    std_total_mc = 0.0
    if stats_mc_saturation is not None and not stats_mc_saturation.empty:
        mean_col = ("H2 total [tons]", "mean")
        std_col = ("H2 total [tons]", "std")
        if mean_col in stats_mc_saturation.columns:
            total_tons_sat = float(stats_mc_saturation[mean_col].sum())
        try:
            iter_totals = stats_mc_saturation.attrs.get("iter_totals", None)
            if iter_totals is not None and len(iter_totals) > 1:
                std_total_mc = float(np.std(np.asarray(iter_totals, dtype=float), ddof=0))
        except Exception:
            std_total_mc = 0.0
    mean_efficiency = (total_tons_sat / total_tons_no_sat) * 100 if total_tons_no_sat > 0 else 0.0

    df_plot = pd.DataFrame(
        {
            "Temperature Range": stats_mc_saturation.index if stats_mc_saturation is not None else [],
            "H2 total [tons]": stats_mc_saturation[("H2 total [tons]", "mean")].values
            if stats_mc_saturation is not None
            else [],
            "H2 std [tons]": stats_mc_saturation[("H2 total [tons]", "std")].values
            if stats_mc_saturation is not None and ("H2 total [tons]", "std") in stats_mc_saturation.columns
            else [],
            "Label": [""] * (len(stats_mc_saturation) if stats_mc_saturation is not None else 0),
        }
    )

    plot_h2_production_summary(
        df_plot,
        volume_at_temperature,
        years=years,
        results_path=results_path,
        stats=stats_mc_saturation,
        production_rate_volumetric=production_rate_volumetric,
    )

    print_saturation_summary(
        total_kg_rocks=total_kg_rocks,
        years=years,
        total_tons_sat=total_tons_sat,
        std_total_mc=std_total_mc,
        mean_efficiency=mean_efficiency,
    )

    return stats_mc_saturation, total_tons_sat, std_total_mc, mean_efficiency


def run_saturation_monte_carlo(
    n_iter,
    volume_at_temperature,
    df_saturation_table,
    mean_pressure_ranges,
    serpentinization_degree,
    int_fracture_spacing,
    permeability_fractures,
    flow_target,
    production_rate_volumetric,
    years=1,
    dist_x=1.0,
    dist_y=1.0,
    dist_z=1.0,
    kg_rocks_dict=None,
    verbose=True,
    show_progress=True,
    worker_count: Optional[int] = None,
    progress_desc=None,
    progress_position=0,
    progress_leave=True,
    progress_callback=None,
    custom_factors_list: Optional[List[Optional[Dict[str, float]]]] = None,
    iter_total_metric: str = "final",
    capacity_col_name: str = "H2 capacity limit [tons]",
    seed_base: Optional[int] = None,
    disable_scale_factor_ws: bool = False,
    dt_day: Optional[float] = None,
    use_equilibrium: bool = True,
    track_timeseries: bool = False,
    *,
    mc_config,
    n_cores,
    seed,
    porosity_front,
    density_serpentinite,
):
    """Run the saturation-aware H₂ estimator multiple times and aggregate statistics."""
    dfs = []
    config_mc = mc_config or {}
    if dt_day is None:
        dt_day = config_mc.get("dt_day", 1 / 2500)
    sampling = config_mc.get("sampling", "random").lower()
    sampling_seed_value = config_mc.get("sampling_seed", seed)
    sampling_seed = None if sampling_seed_value is None else int(sampling_seed_value)

    prod_vol = production_rate_volumetric
    if isinstance(prod_vol, list):
        prod_vol = dict(prod_vol)
    if isinstance(prod_vol, dict) and len(prod_vol) == 1:
        only_key, only_val = next(iter(prod_vol.items()))
        if isinstance(only_val, list) and all(isinstance(t, tuple) and len(t) == 2 for t in only_val):
            prod_vol = dict(only_val)
    prod_vol_clean = {k: _as_scalar_value(v) for k, v in prod_vol.items()}

    effective_n_iter = n_iter if custom_factors_list is None else len(custom_factors_list)
    if effective_n_iter <= 0:
        return pd.DataFrame()

    requested_workers = worker_count if worker_count is not None else n_cores
    worker_count = _clamp_worker_count(requested_workers)
    worker_count = min(worker_count, max(1, effective_n_iter))

    if effective_n_iter <= worker_count:
        chunk_size_value = effective_n_iter
    else:
        batch_estimate = math.ceil(effective_n_iter / 2500)
        target_batches = min(8, max(1, batch_estimate))
        chunk_size_value = math.ceil(effective_n_iter / target_batches)
        chunk_size_value = max(worker_count, chunk_size_value)
    chunk_size_value = min(chunk_size_value, effective_n_iter)

    max_chunk_cfg = config_mc.get("max_chunk_size")
    if isinstance(max_chunk_cfg, (int, float)) and max_chunk_cfg > 0:
        chunk_size_value = max(1, min(chunk_size_value, int(max_chunk_cfg)))

    if seed_base is not None:
        _seed_base = int(seed_base)
    elif sampling_seed is not None:
        _seed_base = int(sampling_seed)
    else:
        _seed_base = seed
    seeds = [(_seed_base + i * 1_000_003) % 1_000_000_000 for i in range(effective_n_iter)]

    range_specs: List[Tuple[str, Tuple[float, float]]] = [
        ("vol_range", config_mc["vol_range"]),
        ("mean_press_range", config_mc["mean_press_range"]),
        ("serp_deg_range", config_mc["serp_deg_range"]),
        ("spacing_range", config_mc["spacing_range"]),
        ("perm_range", config_mc["perm_range"]),
        ("prod_rate_range", config_mc["prod_rate_range"]),
        ("d_range", config_mc["d_range"]),
        ("kg_rocks_range", config_mc["kg_rocks_range"]),
        ("solubility_scaling_range", config_mc["solubility_scaling_range"]),
    ]

    factors_list: List[Optional[Dict[str, float]]] = [None] * effective_n_iter
    if custom_factors_list is not None:
        typed_custom: List[Optional[Dict[str, float]]] = []
        for entry in custom_factors_list:
            if entry is None:
                typed_custom.append(None)
            else:
                typed_custom.append({k: float(v) for k, v in dict(entry).items()})
        factors_list = typed_custom
    else:
        if sampling in {"lhs", "sobol"}:
            d = len(range_specs)
            U = _sample_unit_hypercube(effective_n_iter, d, sampling, sampling_seed)
            factors_list = [
                {name: float(a + U[i, j] * (b - a)) for j, (name, (a, b)) in enumerate(range_specs)}
                for i in range(effective_n_iter)
            ]
        else:
            factors_list = [None] * effective_n_iter

    worker_context = {
        "volume_at_temperature": volume_at_temperature,
        "df_saturation_table": df_saturation_table,
        "mean_pressure_ranges": mean_pressure_ranges,
        "serpentinization_degree": serpentinization_degree,
        "int_fracture_spacing": int_fracture_spacing,
        "permeability_fractures": permeability_fractures,
        "flow_target": flow_target,
        "prod_vol_clean": prod_vol_clean,
        "years": years,
        "dist_x": dist_x,
        "dist_y": dist_y,
        "dist_z": dist_z,
        "kg_rocks_dict": kg_rocks_dict,
        "disable_scale_factor_ws": disable_scale_factor_ws,
        "dt_day": dt_day,
        "use_equilibrium": use_equilibrium,
        "track_timeseries": track_timeseries,
        "config": config_mc,
        "porosity_front": porosity_front,
        "density_serpentinite": density_serpentinite,
    }

    payloads = [(i, seeds[i], factors_list[i]) for i in range(effective_n_iter)]

    def handle_result(iter_idx, df):
        if df is not None and not df.empty:
            df["iter"] = iter_idx
            dfs.append(df)

    desc_label = progress_desc or "Monte Carlo with saturation"
    progress_ctx = (
        _get_progress_bar(
            total=effective_n_iter,
            desc=desc_label,
            file=sys.stdout,
            position=progress_position,
            leave=progress_leave,
        )
        if show_progress
        else nullcontext()
    )

    with progress_ctx as pbar:
        completed_iters = 0

        def mark_progress():
            nonlocal completed_iters
            completed_iters += 1
            if pbar is not None:
                pbar.update(1)
            if progress_callback is not None:
                try:
                    progress_callback(completed_iters, effective_n_iter)
                except Exception:
                    pass

        if worker_count == 1:
            _init_saturation_mc_worker_context(worker_context)
            try:
                for payload in payloads:
                    iter_idx, df = _run_saturation_mc_iteration(payload)
                    handle_result(iter_idx, df)
                    mark_progress()
            finally:
                _clear_saturation_mc_worker_context()
        else:
            inflight_cap = max(1, min(chunk_size_value, worker_count * 2, effective_n_iter))
            with ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=_init_saturation_mc_worker_context,
                initargs=(worker_context,),
            ) as executor:
                payload_iter = iter(payloads)
                inflight = set()

                try:
                    for _ in range(inflight_cap):
                        payload = next(payload_iter)
                        inflight.add(executor.submit(_run_saturation_mc_iteration, payload))
                except StopIteration:
                    pass

                while inflight:
                    done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                    for future in done:
                        iter_idx, df = future.result()
                        handle_result(iter_idx, df)
                        mark_progress()

                    try:
                        for _ in range(inflight_cap - len(inflight)):
                            payload = next(payload_iter)
                            inflight.add(executor.submit(_run_saturation_mc_iteration, payload))
                    except StopIteration:
                        pass

    if not dfs:
        return pd.DataFrame()

    df_all = pd.concat(dfs, ignore_index=True)
    for df_part in dfs:
        ts = getattr(df_part, "attrs", {}).get("timeseries_sample") if hasattr(df_part, "attrs") else None
        if ts:
            df_all.attrs["timeseries_sample"] = ts
            break

    if iter_total_metric.lower() in {"capacity", "cap", "solubility"}:
        if capacity_col_name not in df_all.columns:
            if "H2_total_capacity_limit_mol" in df_all.columns:
                df_all[capacity_col_name] = df_all["H2_total_capacity_limit_mol"].astype(float) * 2.016 / 1e6
            else:
                df_all[capacity_col_name] = 0.0

    metric_key = iter_total_metric.lower()
    if metric_key in {"final", "capped", "observed"}:
        iter_col = "H2 total [tons]"
    elif metric_key in {"capacity", "cap", "solubility"}:
        iter_col = capacity_col_name
    else:
        iter_col = "H2 total [tons]"

    if iter_col in df_all.columns:
        if metric_key in {"final", "capped", "observed"}:
            iter_totals = df_all.groupby("iter")[iter_col].sum()
        else:
            iter_totals = df_all.groupby("iter")[iter_col].max()
    else:
        iter_totals = pd.Series(dtype=float)
    limiter_series = df_all["Limiting factor"]
    limiter_series_clean = limiter_series[limiter_series != "-"]
    dominant_limiter = limiter_series_clean.mode().iloc[0] if not limiter_series_clean.mode().empty else "-"

    numeric_cols = df_all.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        stats = pd.DataFrame(index=df_all["Temperature Range"].unique())
    else:
        stats = df_all.groupby("Temperature Range")[numeric_cols].agg(["mean", "std"])
    limiting_factors = df_all.groupby("Temperature Range")["Limiting factor"].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else "-"
    )

    if mean_pressure_ranges:
        temp_order = sorted(mean_pressure_ranges.keys(), key=lambda x: float(x.split("_")[0]))
        stats = stats.reindex(temp_order)

    for col in [
        "Flow target [kg/day]",
        "daily_diffused_H2O [kg/day]",
        "daily_fractured_H2O [kg/day]",
    ]:
        key = (col, "mean")
        if key in stats.columns:
            stats[key] = stats[key].fillna(0.0)

    stats.attrs["iter_totals"] = iter_totals
    stats.attrs["iter_total_metric"] = iter_total_metric
    stats.attrs["iter_total_column"] = iter_col
    stats.attrs["dominant_limiter"] = dominant_limiter
    stats.attrs["limiting_factors_per_range"] = {tr: limiting_factors.get(tr, "-") for tr in stats.index}
    stats.attrs["df_all"] = df_all

    return stats


def _init_saturation_mc_worker_context(context: Optional[Dict[str, Any]] = None):
    global _MC_WORKER_CONTEXT
    _MC_WORKER_CONTEXT = context


def _clear_saturation_mc_worker_context():
    _init_saturation_mc_worker_context(None)


def _get_saturation_mc_worker_context() -> Dict[str, Any]:
    if _MC_WORKER_CONTEXT is None:
        raise RuntimeError("Monte Carlo worker context not initialized.")
    return _MC_WORKER_CONTEXT


def _run_saturation_mc_iteration(args):
    iteration_idx, seed, factors = args
    context = _get_saturation_mc_worker_context()
    volume_at_temperature = context["volume_at_temperature"]
    df_saturation_table = context["df_saturation_table"]
    mean_pressure_ranges = context["mean_pressure_ranges"]
    serpentinization_degree = context["serpentinization_degree"]
    int_fracture_spacing = context["int_fracture_spacing"]
    permeability_fractures = context["permeability_fractures"]
    flow_target = context["flow_target"]
    prod_vol_clean = context["prod_vol_clean"]
    years = context["years"]
    dist_x = context["dist_x"]
    dist_y = context["dist_y"]
    dist_z = context["dist_z"]
    kg_rocks_dict = context["kg_rocks_dict"]
    disable_scale_factor_ws = context["disable_scale_factor_ws"]
    dt_day = context["dt_day"]
    use_equilibrium = context["use_equilibrium"]
    track_timeseries = context["track_timeseries"]
    config = context["config"]
    porosity_front = context["porosity_front"]
    density_serpentinite = context["density_serpentinite"]
    rng_stdlib = random.Random(seed)
    rng_np = np.random.default_rng(seed) if hasattr(np.random, "default_rng") else None
    track_this_iter = bool(track_timeseries and iteration_idx == 0)

    range_specs: List[Tuple[str, Tuple[float, float]]] = [
        ("vol_range", config["vol_range"]),
        ("mean_press_range", config["mean_press_range"]),
        ("serp_deg_range", config["serp_deg_range"]),
        ("spacing_range", config["spacing_range"]),
        ("perm_range", config["perm_range"]),
        ("prod_rate_range", config["prod_rate_range"]),
        ("d_range", config["d_range"]),
        ("kg_rocks_range", config["kg_rocks_range"]),
        ("solubility_scaling_range", config["solubility_scaling_range"]),
    ]

    if factors is None:
        vol = {k: v * rng_stdlib.uniform(*config["vol_range"]) for k, v in volume_at_temperature.items()}
        mean_press = {k: v * rng_stdlib.uniform(*config["mean_press_range"]) for k, v in mean_pressure_ranges.items()}
        serp_deg = serpentinization_degree * rng_stdlib.uniform(*config["serp_deg_range"])
        spacing = int_fracture_spacing * rng_stdlib.uniform(*config["spacing_range"])
        perm = permeability_fractures * rng_stdlib.uniform(*config["perm_range"])
        prod_rate = {k: v * rng_stdlib.uniform(*config["prod_rate_range"]) for k, v in prod_vol_clean.items()}
        d_factor = rng_stdlib.uniform(*config["d_range"])
        dx = dist_x * d_factor
        dy = dist_y * d_factor
        dz = dist_z * d_factor
        if kg_rocks_dict is None:
            kg_rocks = None
        else:
            kg_rocks = {k: v * rng_stdlib.uniform(*config["kg_rocks_range"]) for k, v in kg_rocks_dict.items()}
        solubility_scaling = {k: rng_stdlib.uniform(*config["solubility_scaling_range"]) for k in volume_at_temperature.keys()}
        applied_factors = {name.replace("_range", ""): np.nan for name, _ in range_specs}
    else:
        f = lambda name: float(factors.get(name, 1.0))
        vol = {k: v * f("vol_range") for k, v in volume_at_temperature.items()}
        mean_press = {k: v * f("mean_press_range") for k, v in mean_pressure_ranges.items()}
        serp_deg = serpentinization_degree * f("serp_deg_range")
        spacing = int_fracture_spacing * f("spacing_range")
        perm = permeability_fractures * f("perm_range")
        prod_rate = {k: v * f("prod_rate_range") for k, v in prod_vol_clean.items()}
        d_factor = f("d_range")
        dx = dist_x * d_factor
        dy = dist_y * d_factor
        dz = dist_z * d_factor
        if kg_rocks_dict is None:
            kg_rocks = None
        else:
            kg_rocks = {k: v * f("kg_rocks_range") for k, v in kg_rocks_dict.items()}
        solubility_scaling = {k: f("solubility_scaling_range") for k in volume_at_temperature.keys()}
        applied_factors = {name.replace("_range", ""): f(name) for name, _ in range_specs}

    days = years * 365
    results = []

    flow_info = _prepare_flow_target_scaling(
        volume_at_temperature=vol,
        df_saturation_table=df_saturation_table,
        production_rate_volumetric=prod_rate,
        flow_target=flow_target,
        kg_rocks_dict=kg_rocks,
    )
    valid_ranges = flow_info["valid_ranges"]
    flow_targets_scaled = flow_info["flow_targets_scaled"]

    if disable_scale_factor_ws:
        scale_factor_ws = 1.0
    else:
        actual_vals = df_saturation_table["Actual [mol/day]"].dropna().values
        std_vals = df_saturation_table["Actual [mol/day] std"].dropna().values
        if len(actual_vals) > 0 and len(std_vals) > 0 and np.mean(actual_vals) > 0:
            scale = np.mean(std_vals) / np.mean(actual_vals)
            normal = rng_np.normal if (rng_np is not None and hasattr(rng_np, "normal")) else np.random.normal
            scale_factor_ws = normal(loc=1.0, scale=scale)
        else:
            scale_factor_ws = 1.0

    for _, row in df_saturation_table.iterrows():
        t_range = row.get("Temperature Range", None)
        result_row = _simulate_temperature_range(
            row=row,
            t_range=t_range,
            valid_ranges=valid_ranges,
            flow_targets_scaled=flow_targets_scaled,
            volume_at_temperature=vol,
            mean_pressure_ranges=mean_press,
            serpentinization_degree=serp_deg,
            int_fracture_spacing=spacing,
            permeability_fractures=perm,
            flow_target=flow_target,
            solubility_scaling=solubility_scaling,
            days=days,
            dist_x=dx,
            dist_y=dy,
            dist_z=dz,
            kg_rocks_dict=kg_rocks,
            scale_factor_ws=scale_factor_ws,
            dt_day=dt_day,
            verbose=False,
            track_timeseries=track_this_iter,
            porosity_front=porosity_front,
            density_serpentinite=density_serpentinite,
            use_equilibrium=use_equilibrium,
        )

        if result_row is not None:
            results.append(result_row)

    df = pd.DataFrame(results)
    pore_sat_series = df.get("Pore saturation time [days]")
    df = df.fillna(0.0)
    if pore_sat_series is not None:
        df["Pore saturation time [days]"] = pore_sat_series
    if track_this_iter:
        ts = {
            r.get("Temperature Range", f"{r.get('Temperature [°C]', 'NA')}"): r.get("timeseries")
            for r in results
            if isinstance(r, dict) and "timeseries" in r
        }
        if ts:
            df.attrs["timeseries_sample"] = ts
    if "H2 total [tons] std" in df.columns:
        _std_total = float(np.sqrt(np.sum(np.square(df["H2 total [tons] std"].fillna(0.0)))))
    else:
        _std_total = None

    for key, value in applied_factors.items():
        df[key] = value
    return iteration_idx, df


def _prepare_flow_target_scaling(volume_at_temperature, df_saturation_table, production_rate_volumetric, flow_target, kg_rocks_dict=None):
    """Allocate the global clean-water target across temperature ranges."""
    valid_ranges = {k: v for k, v in volume_at_temperature.items() if v > 0}
    total_vol = sum(valid_ranges.values())
    fraction_for_nonproductive = 0.0
    t_min = df_saturation_table["Temperature [°C]"].min()
    t_max = df_saturation_table["Temperature [°C]"].max()

    if isinstance(production_rate_volumetric, list):
        production_rate_volumetric = dict(production_rate_volumetric)

    if isinstance(production_rate_volumetric, dict) and len(production_rate_volumetric) == 1:
        only_key, only_val = next(iter(production_rate_volumetric.items()))
        if isinstance(only_val, list) and all(isinstance(t, tuple) and len(t) == 2 for t in only_val):
            production_rate_volumetric = dict(only_val)

    production_rate_volumetric_clean = {k: _as_scalar_value(v) for k, v in production_rate_volumetric.items()}

    max_prod_global = max(production_rate_volumetric_clean.values()) if production_rate_volumetric_clean else 0.0

    weights = {}
    for _, row in df_saturation_table.iterrows():
        required_keys = ["Water [kg/day]", "Actual [mol/day]", "Absorbed H₂O [kg/day]"]
        if any(pd.isna(row.get(k)) or row.get(k) == "-" for k in required_keys):
            continue

        t_range = row.get("Temperature Range", None)
        if t_range not in valid_ranges:
            continue

        if kg_rocks_dict is not None:
            rocks_val = _as_scalar_value(kg_rocks_dict.get(t_range, 0.0))
            if rocks_val <= 0:
                continue

        t_c = row["Temperature [°C]"]

        temp_penalty = (t_max - t_c) / (t_max - t_min)
        range_volume_km3 = valid_ranges.get(t_range, 0.0)
        base_weight = (range_volume_km3 / total_vol) * temp_penalty if total_vol > 0 else 0.0

        prod_value = production_rate_volumetric_clean.get(t_range, 0.0)
        actual_mol_day = _as_scalar_value(row.get("Actual [mol/day]", 0.0))
        if prod_value <= 0 or actual_mol_day <= 0:
            continue

        if max_prod_global > 0 and prod_value > 0:
            prod_factor = prod_value / max_prod_global
        else:
            prod_factor = fraction_for_nonproductive

        weights[t_range] = base_weight * prod_factor

    sum_weights = sum(weights.values())
    if sum_weights > 0:
        flow_targets_scaled = {t_range: flow_target * (w / sum_weights) for t_range, w in weights.items()}
    else:
        flow_targets_scaled = {k: 0.0 for k in valid_ranges}
    return {
        "valid_ranges": valid_ranges,
        "production_rate_volumetric_clean": production_rate_volumetric_clean,
        "flow_targets_scaled": flow_targets_scaled,
        "t_min": t_min,
        "t_max": t_max,
    }


def _run_range_time_integration(setup, days, dt_day=1 / 2500, use_equilibrium=True, track_timeseries=False):
    """Integrate the daily multi-substep loop for one temperature range."""
    max_prod_mol_day = setup["Max_prod_mol_day"]
    max_absorbed_kg_day = setup["Max_absorbed_kg_day"]
    max_front_mass_kg_day = setup["Max_front_mass_kg_day"]
    diff_h2o_total = setup["Diff_H2O_total"]
    frac_h2o_total = setup["Frac_H2O_total"]
    solubility = setup["solubility"]
    flow_target_scaled = setup["flow_target_scaled"]
    rho_water = setup.get("rho_water", 1000.0)
    pore_volume_m3 = setup.get("pore_volume_m3", 0.0)
    pore_volume_turnover_days = setup.get("pore_volume_turnover_days", float("nan"))
    min_equil_days = setup.get("min_equil_days", days)

    h2_saturation_fraction = 0.0
    h2_total_prod_mol = 0.0
    h2_total_dissolved_mol = 0.0
    h2o_total_absorbed_clean_kg = 0.0
    h2o_total_diffused_kg = 0.0
    h2_total_raw_possible_mol = max_prod_mol_day * days
    h2_total_capacity_limit_mol = 0.0

    dt_day = float(dt_day)
    if dt_day <= 0:
        raise ValueError(f"dt_day must be > 0, got {dt_day}")
    substeps = max(1, int(math.ceil(1.0 / dt_day)))
    dt_sub = 1.0 / substeps
    h2o_parcel_water_mass_kg = 0.0
    h2_saturation_fraction = 0
    prev_day_saturation = 0.0
    daily_diffused_kg_total = 0.0
    daily_fractured_kg_total = 0.0

    warmup_days = 25
    cooldown_days = 25
    main_days = max(days, min_equil_days)
    eq_days_needed = main_days
    eq_days_count = 0
    prev_h2_prod = prev_abs_water = None
    warmup_cap_days = warmup_days
    total_sim_days = warmup_days + main_days + cooldown_days
    max_sim_days = total_sim_days
    final_avg_daily_saturation = 0.0
    sum_daily_saturation_eq = 0.0
    eq_days_count_for_sat = 0
    day = 0

    ts_prod = []
    ts_abs = []
    ts_sat = []
    eq_prod_days = []
    stable_meta = {}
    saturation_threshold = 0.99
    pore_saturation_time_days = None

    if not use_equilibrium:
        main_days = days
        eq_days_needed = main_days
        total_sim_days = warmup_days + main_days + cooldown_days
        max_sim_days = total_sim_days

    while day < max_sim_days:
        h2_prod_mol_day = 0.0
        h2_dissolved_mol_day = 0.0
        h2o_absorbed_clean_kg_day = 0.0
        h2o_pore_mass_kg_day = 0.0
        h2_unsat_fraction_integrated = 0.0

        h2_saturation_fraction = prev_day_saturation

        max_clean_water_kg = min(diff_h2o_total + frac_h2o_total, flow_target_scaled)

        daily_new_clean_water_kg = 0.0
        daily_saturation_integrated = 0.0

        for step in range(substeps):
            clean_water_added_kg = 0.0
            max_capacity_kg = max_front_mass_kg_day * dt_sub

            new_clean_water_kg = max_clean_water_kg * dt_sub
            daily_new_clean_water_kg += new_clean_water_kg

            old_water_kg = h2o_parcel_water_mass_kg
            total_attempted_water = old_water_kg + new_clean_water_kg

            if total_attempted_water > max_capacity_kg:
                flushed_kg = total_attempted_water - max_capacity_kg
                actual_new_water_kg = new_clean_water_kg - flushed_kg
            else:
                flushed_kg = 0.0
                actual_new_water_kg = new_clean_water_kg

            h2o_parcel_water_mass_kg = old_water_kg + actual_new_water_kg

            if h2o_parcel_water_mass_kg > 0.0:
                h2_saturation_fraction = (h2_saturation_fraction * old_water_kg) / h2o_parcel_water_mass_kg
            else:
                h2_saturation_fraction = 0.0

            unsat_fraction = max(0.0, 1.0 - h2_saturation_fraction)
            can_produce_h2 = unsat_fraction > 1e-12
            pore_water_mass_kg = unsat_fraction * max_capacity_kg if can_produce_h2 else 0.0
            h2o_pore_mass_kg_day += pore_water_mass_kg

            chem_unsat = 1.0 - h2_saturation_fraction

            if can_produce_h2 and pore_water_mass_kg > 0.0 and max_capacity_kg > 0.0:
                potential_absorption = min(
                    max_absorbed_kg_day * chem_unsat * dt_sub,
                    h2o_parcel_water_mass_kg,
                )

                frac_of_ref_water = min(h2o_parcel_water_mass_kg / max_capacity_kg, 1.0)
                potential_h2_mol = max_prod_mol_day * dt_sub * frac_of_ref_water

                capacity_h2_mol = chem_unsat * solubility * h2o_parcel_water_mass_kg

                if potential_h2_mol > 0:
                    scale_factor = min(1.0, capacity_h2_mol / potential_h2_mol)
                else:
                    scale_factor = 0.0

                absorbed_clean_kg = potential_absorption * scale_factor
            else:
                absorbed_clean_kg = 0.0

            if new_clean_water_kg > 0:
                total_inflow = diff_h2o_total + frac_h2o_total
                diff_frac_ratio = diff_h2o_total / total_inflow if total_inflow > 0 else 0.0
                daily_diffused_kg = new_clean_water_kg * diff_frac_ratio
                daily_fractured_kg = new_clean_water_kg * (1.0 - diff_frac_ratio)
            else:
                daily_diffused_kg = 0.0
                daily_fractured_kg = 0.0

            daily_diffused_kg_total += daily_diffused_kg
            daily_fractured_kg_total += daily_fractured_kg

            h2o_parcel_water_mass_kg -= absorbed_clean_kg
            clean_water_added_kg = absorbed_clean_kg
            h2o_parcel_water_mass_kg = min(h2o_parcel_water_mass_kg, h2o_pore_mass_kg_day)

            if max_capacity_kg > 0:
                h2_saturation_fraction = min(1.0, h2o_parcel_water_mass_kg / max_capacity_kg)
            else:
                h2_saturation_fraction = 0.0

            if pore_saturation_time_days is None and h2_saturation_fraction >= saturation_threshold:
                pore_saturation_time_days = day + (step + 1) * dt_sub

            if can_produce_h2 and h2o_parcel_water_mass_kg > 0.0 and max_capacity_kg > 0.0:
                frac_of_ref_water = h2o_parcel_water_mass_kg / max_capacity_kg
                frac_of_ref_water = min(frac_of_ref_water, 1.0)

                potential_h2_mol = max_prod_mol_day * dt_sub * frac_of_ref_water
                capacity_h2_mol = chem_unsat * solubility * h2o_parcel_water_mass_kg
                h2_generated_mol = min(potential_h2_mol, capacity_h2_mol)

                if solubility * h2o_parcel_water_mass_kg > 0:
                    added_sat = h2_generated_mol / (solubility * h2o_parcel_water_mass_kg)
                else:
                    added_sat = 0.0
                h2_saturation_fraction = min(1.0, h2_saturation_fraction + added_sat)

                h2_total_capacity_limit_mol += capacity_h2_mol
            else:
                h2_generated_mol = 0.0

            if (not use_equilibrium) or (day > 0):
                h2_prod_mol_day += h2_generated_mol
                h2_dissolved_mol_day += h2_generated_mol
                h2o_absorbed_clean_kg_day += clean_water_added_kg
                h2_unsat_fraction_integrated += unsat_fraction * dt_sub
                daily_saturation_integrated += h2_saturation_fraction * dt_sub

        prev_h2_prod = h2_prod_mol_day
        prev_abs_water = h2o_absorbed_clean_kg_day

        if warmup_days <= day < (warmup_days + main_days):
            eq_days_count += 1
            eq_days_count_for_sat += 1
            h2_total_prod_mol += h2_prod_mol_day
            h2_total_dissolved_mol += h2_dissolved_mol_day
            h2o_total_absorbed_clean_kg += h2o_absorbed_clean_kg_day
            h2o_total_diffused_kg += daily_new_clean_water_kg
            sum_daily_saturation_eq += daily_saturation_integrated
            eq_prod_days.append(h2_prod_mol_day)

        if track_timeseries:
            ts_prod.append(h2_prod_mol_day)
            ts_abs.append(h2o_absorbed_clean_kg_day)
            ts_sat.append(h2_saturation_fraction)

        day += 1
        prev_day_saturation = h2_saturation_fraction

    target_len_days = max(1, int(math.ceil(days)))
    cooldown_days = 25
    series_for_window = eq_prod_days if eq_prod_days else ts_prod
    if cooldown_days > 0 and len(series_for_window) > cooldown_days:
        series_for_window = series_for_window[:-cooldown_days]
    stable_meta = _select_stable_window(series_for_window, target_len=target_len_days, warmup_days=0)
    mean_stable = stable_meta.get("mean", 0.0)
    used_len = stable_meta.get("len_used", 0)
    extend_needed = stable_meta.get("extend_needed", False)
    h2_total_prod_mol = mean_stable * target_len_days if used_len > 0 else h2_total_prod_mol
    h2_total_dissolved_mol = h2_total_prod_mol
    eq_days_count = target_len_days

    if eq_days_count_for_sat > 0:
        final_avg_daily_saturation = sum_daily_saturation_eq / eq_days_count_for_sat
    else:
        final_avg_daily_saturation = 0.0

    return dict(
        H2_total_prod_mol=h2_total_prod_mol,
        H2_total_dissolved_mol=h2_total_dissolved_mol,
        H2O_total_absorbed_clean_kg=h2o_total_absorbed_clean_kg,
        H2O_total_diffused_kg=h2o_total_diffused_kg,
        H2_total_raw_possible_mol=max_prod_mol_day * eq_days_count,
        H2_total_capacity_limit_mol=h2_total_capacity_limit_mol,
        H2_efficiency_percent=(
            h2_total_prod_mol / (max_prod_mol_day * eq_days_count) * 100.0
            if max_prod_mol_day * eq_days_count > 0
            else 0.0
        ),
        daily_diffused_kg_total=daily_diffused_kg_total / max(eq_days_count, 1),
        daily_fractured_kg_total=daily_fractured_kg_total / max(eq_days_count, 1),
        eq_days_count=eq_days_count,
        final_avg_daily_saturation=final_avg_daily_saturation,
        dt_day=dt_day,
        pore_volume_m3=pore_volume_m3,
        pore_volume_turnover_days=pore_volume_turnover_days,
        pore_saturation_time_days=float(pore_saturation_time_days) if pore_saturation_time_days is not None else float("nan"),
        timeseries={
            "H2_prod_mol_day": stable_meta.get("smoothed", _moving_average(_trim_trailing_dropoff(ts_prod))),
            "H2_prod_mol_day_full": ts_prod,
            "H2O_absorbed_kg_day": ts_abs,
            "sat_fraction": ts_sat,
            "__meta": {
                "warmup_days": warmup_days,
                "cooldown_days": cooldown_days,
                "stable_window": stable_meta,
            },
        }
        if track_timeseries
        else None,
        stable_window=stable_meta,
    )


def _build_saturation_range_result_row(setup, sim, years, kg_rocks_dict):
    """Build the per-temperature-range summary row after time integration."""

    t_range = setup["t_range"]
    t_c = setup["T_C"]
    d_eff = setup["D_eff"]
    solubility = setup["solubility"]
    flow_target_scaled = setup["flow_target_scaled"]
    rho_water = setup.get("rho_water", 1000.0)

    h2_total_prod_mol = sim["H2_total_prod_mol"]
    h2_total_dissolved_mol = sim["H2_total_dissolved_mol"]
    h2o_total_absorbed_clean_kg = sim["H2O_total_absorbed_clean_kg"]
    h2o_total_diffused_kg = sim["H2O_total_diffused_kg"]
    h2_total_raw_possible_mol = sim["H2_total_raw_possible_mol"]

    total_water_delivered_kg = h2o_total_diffused_kg
    solubility_cap_mol = solubility * max(total_water_delivered_kg, 0.0)
    h2_total_prod_mol = min(h2_total_prod_mol, solubility_cap_mol)
    h2_total_dissolved_mol = min(h2_total_prod_mol, solubility_cap_mol)
    h2_total_capacity_limit_mol = min(sim["H2_total_capacity_limit_mol"], solubility_cap_mol)
    h2_efficiency_percent = sim["H2_efficiency_percent"]
    daily_diffused_kg_total = sim["daily_diffused_kg_total"]
    daily_fractured_kg_total = sim["daily_fractured_kg_total"]
    eq_days_count = sim["eq_days_count"]
    final_avg_daily_saturation = sim["final_avg_daily_saturation"]
    pore_volume_m3 = sim.get("pore_volume_m3", float("nan"))
    range_pore_volume_m3 = setup.get("range_pore_volume_m3", pore_volume_m3)
    pore_volume_from_mass_m3 = setup.get("pore_volume_for_sat_m3", pore_volume_m3)
    pore_volume_turnover_days = sim.get("pore_volume_turnover_days", float("nan"))
    pore_saturation_time_sim = sim.get("pore_saturation_time_days", float("nan"))
    dt_day_used = sim.get("dt_day", float("nan"))

    avg_absorbed_clean_kg_per_day = h2o_total_absorbed_clean_kg / max(eq_days_count, 1)
    h2_total_tons = h2_total_prod_mol * 2.016 / 1e6
    h2_dissolved_tons = h2_total_dissolved_mol * 2.016 / 1e6
    h2_gaseous_tons = h2_total_tons - h2_dissolved_tons
    sat_mol_per_kg = (
        h2_total_dissolved_mol / max(total_water_delivered_kg, 1e-12) if total_water_delivered_kg > 0 else 0.0
    )

    pore_volume_for_sat = (
        pore_volume_from_mass_m3
        if np.isfinite(pore_volume_from_mass_m3) and pore_volume_from_mass_m3 > 0
        else range_pore_volume_m3
    )
    if not (np.isfinite(pore_volume_for_sat) and pore_volume_for_sat > 0):
        pore_volume_for_sat = pore_volume_m3

    pore_water_mass_kg = pore_volume_for_sat * rho_water if np.isfinite(pore_volume_for_sat) else float("nan")
    daily_dissolved_mol = h2_total_dissolved_mol / max(eq_days_count, 1) if eq_days_count > 0 else float("nan")

    pore_saturation_time_days = float("nan")
    if np.isfinite(pore_saturation_time_sim) and pore_saturation_time_sim > 0:
        pore_saturation_time_days = pore_saturation_time_sim
    elif (
        np.isfinite(solubility)
        and np.isfinite(pore_water_mass_kg)
        and pore_water_mass_kg > 0
        and np.isfinite(daily_dissolved_mol)
        and daily_dissolved_mol > 1e-9
    ):
        pore_saturation_time_days = (solubility * pore_water_mass_kg) / daily_dissolved_mol

    water_limited_threshold_tons = 10.0

    if h2_total_tons == 0.0:
        limiter = "-"
    elif h2_total_tons < water_limited_threshold_tons:
        limiter = "water"
    else:
        if np.isfinite(pore_saturation_time_days) and np.isfinite(pore_volume_turnover_days):
            limiter = "sol" if pore_saturation_time_days < pore_volume_turnover_days else "rate"
        else:
            limiter = "-"

    rocks_kg_day = 0.0
    if kg_rocks_dict is not None and h2_efficiency_percent is not None:
        efficiency_value = float(h2_efficiency_percent) if not pd.isna(h2_efficiency_percent) else 0.0
        rocks_kg_day = kg_rocks_dict.get(t_range, 0.0) * (efficiency_value / 100.0)

    return {
        "Temperature Range": t_range or f"{t_c:.0f}°C",
        "Temperature [°C]": t_c,
        "D_eff [m²/day]": d_eff,
        "Flow target [kg/day]": flow_target_scaled,
        "Diff H2O [kg/day]": setup["Diff_H2O_total"],
        "Frac H2O [kg/day]": setup["Frac_H2O_total"],
        "Solubility [mol/kg]": solubility,
        "Saturation [mol/kg]": sat_mol_per_kg,
        "Efficiency [%]": h2_efficiency_percent,
        "H2 total [tons]": h2_total_tons,
        "H2 dissolved [tons]": h2_dissolved_tons,
        "H2 gaseous [tons]": h2_gaseous_tons,
        "H2O absorbed [kg/day]": avg_absorbed_clean_kg_per_day,
        "H2O diffused [kg]": h2o_total_diffused_kg,
        "Limiting factor": limiter,
        "daily_diffused_H2O [kg/day]": daily_diffused_kg_total,
        "daily_fractured_H2O [kg/day]": daily_fractured_kg_total,
        "Rocks [kg/day]": rocks_kg_day,
        "W/R ratio [-]": flow_target_scaled / rocks_kg_day if rocks_kg_day else 0.0,
        "Pore volume [m³]": pore_volume_m3,
        "Pore volume turnover [days]": pore_volume_turnover_days,
        "Pore saturation time [days]": pore_saturation_time_days,
        "dt_day_used [day]": dt_day_used,
        "Equilibrium days simulated": eq_days_count,
        "H2_total_raw_possible_mol": float(h2_total_raw_possible_mol),
        "H2_total_capacity_limit_mol": float(h2_total_capacity_limit_mol),
    }


def _simulate_temperature_range(
    row,
    t_range,
    valid_ranges,
    flow_targets_scaled,
    volume_at_temperature,
    mean_pressure_ranges,
    serpentinization_degree,
    int_fracture_spacing,
    permeability_fractures,
    flow_target,
    solubility_scaling,
    days,
    dist_x,
    dist_y,
    dist_z,
    kg_rocks_dict,
    scale_factor_ws,
    dt_day=1 / 2500,
    use_equilibrium=True,
    verbose=True,
    track_timeseries=False,
    *,
    porosity_front,
    density_serpentinite,
):
    """Simulate one temperature bin for the saturation Monte Carlo."""

    required_keys = ["Water [kg/day]", "Actual [mol/day]", "Absorbed H₂O [kg/day]"]
    if any(pd.isna(row.get(k)) or row.get(k) == "-" for k in required_keys):
        if verbose:
            print(f"[Warning] Skipping row due to missing data at Temp={row.get('Temperature Range')}")
        return None

    flow_target_scaled = flow_targets_scaled.get(t_range, 0.0)
    t_c = row["Temperature [°C]"]

    if t_range not in valid_ranges:
        return None

    range_volume_km3 = valid_ranges.get(t_range, 0.0)
    if range_volume_km3 == 0.0:
        return None

    max_prod_mol_day = row["Actual [mol/day]"] * scale_factor_ws
    max_absorbed_kg_day = row.get("Absorbed H₂O [kg/day]", 0.0) * scale_factor_ws
    rho_water = row.get("Water Density [kg/m³]", 1000.0)
    max_front_mass_kg_day = row["Water [kg/day]"] * scale_factor_ws

    p_bar = mean_pressure_ranges.get(t_range, 1000)
    p_mpa = p_bar / 10.0

    area_block = dist_x * dist_y

    serp_deg_used = min(serpentinization_degree, 98.0)

    volume_total = dist_x * dist_y * dist_z
    volume_core = volume_total * (1 - serp_deg_used / 100)
    side_core = volume_core ** (1 / 3) if volume_core > 0.0 else 0.0
    shell_thickness = (dist_z - side_core) / 2 if dist_z > side_core else dist_z / 2

    d_eff_t = 6.1e-10 * np.exp(-67000 / (8.314 * (t_c + 273.15)))
    d_eff = d_eff_t * np.exp(-0.005 * (p_mpa - 200)) * 86400

    diff_h2o_cell = d_eff * area_block / shell_thickness * rho_water if shell_thickness > 0 else 0.0

    n_fractures = max(1, int(shell_thickness / int_fracture_spacing)) if int_fracture_spacing > 0 else 1
    area_fractures = n_fractures * area_block * 6

    k_f = permeability_fractures
    delta_p = p_mpa * 1e6

    t_k = t_c + 273.15
    mu = 2.414e-5 * 10 ** (247.8 / (t_k - 140))

    qv_fract_s = (k_f * area_fractures / (mu * shell_thickness)) * delta_p if shell_thickness > 0 else 0.0
    qv_fract_d = qv_fract_s * 86400
    frac_h2o_cell = qv_fract_d * rho_water

    cell_volume_m3 = dist_x * dist_y * dist_z
    total_volume_km3 = volume_at_temperature.get(t_range, 0.0)
    total_volume_m3 = total_volume_km3 * 1e9
    num_cells = total_volume_m3 / cell_volume_m3 if cell_volume_m3 > 0 and total_volume_m3 > 0 else 0.0

    diff_h2o_total = diff_h2o_cell * num_cells
    frac_h2o_total = frac_h2o_cell * num_cells

    total_inflow_possible = diff_h2o_total + frac_h2o_total
    if total_inflow_possible > flow_target_scaled and total_inflow_possible > 0:
        scale = flow_target_scaled / total_inflow_possible
        diff_h2o_total *= scale
        frac_h2o_total *= scale

    solubility_factor = 1.0
    if solubility_scaling:
        solubility_factor = solubility_scaling.get(t_range, 1.0)
    solubility = compute_h2_solubility_kk_pr(p_mpa, t_c) * solubility_factor

    porosity_used = porosity_front if porosity_front <= 1 else porosity_front / 100.0
    range_pore_volume_m3 = range_volume_km3 * 1e9 * porosity_used

    pore_volume_from_mass_m3 = 0.0
    if kg_rocks_dict is not None and density_serpentinite > 0:
        rock_mass_kg = kg_rocks_dict.get(t_range, 0.0)
        pore_volume_from_mass_m3 = (rock_mass_kg / density_serpentinite) * porosity_used

    pore_volume_for_sat_m3 = pore_volume_from_mass_m3 if pore_volume_from_mass_m3 > 0 else range_pore_volume_m3
    pore_volume_m3 = pore_volume_for_sat_m3 if pore_volume_for_sat_m3 > 0 else max(range_pore_volume_m3, 0.0)

    pore_volume_turnover_days = float("nan")
    total_flow_m3_day = flow_target_scaled / rho_water if rho_water > 0 else 0.0
    if total_flow_m3_day > 0 and pore_volume_m3 > 0:
        pore_volume_turnover_days = pore_volume_m3 / total_flow_m3_day

    min_dt_cap = 1.0 / 10000.0
    baseline_dt = float(dt_day)

    desired_dt = (
        pore_volume_turnover_days / 2.0
        if math.isfinite(pore_volume_turnover_days) and pore_volume_turnover_days > 0
        else float("nan")
    )
    if not math.isfinite(desired_dt) or desired_dt <= 0:
        desired_dt = baseline_dt

    dt_effective = max(min_dt_cap, min(desired_dt, baseline_dt))

    turnover_clamp_factor = 1.0
    turnover_days_effective = pore_volume_turnover_days
    turnover_clamped = False

    min_equil_days = days
    warmup_cap_days = pore_volume_turnover_days if math.isfinite(pore_volume_turnover_days) else 0.0

    setup = {
        "t_range": t_range,
        "T_C": t_c,
        "P_MPa": p_mpa,
        "D_eff": d_eff,
        "Diff_H2O_total": diff_h2o_total,
        "Frac_H2O_total": frac_h2o_total,
        "Max_prod_mol_day": max_prod_mol_day,
        "Max_absorbed_kg_day": max_absorbed_kg_day,
        "Max_front_mass_kg_day": max_front_mass_kg_day,
        "rho_water": rho_water,
        "solubility": solubility,
        "flow_target_scaled": flow_target_scaled,
        "kg_rocks_dict": kg_rocks_dict,
        "range_volume_km3": range_volume_km3,
        "pore_volume_m3": pore_volume_m3,
        "range_pore_volume_m3": range_pore_volume_m3,
        "pore_volume_for_sat_m3": pore_volume_for_sat_m3,
        "pore_volume_turnover_days": pore_volume_turnover_days,
        "turnover_days_effective": turnover_days_effective,
        "turnover_clamped": turnover_clamped,
        "turnover_clamp_factor": turnover_clamp_factor,
        "min_equil_days": min_equil_days,
        "warmup_cap_days": warmup_cap_days,
    }

    sim = _run_range_time_integration(
        setup=setup,
        days=max(days, min_equil_days),
        dt_day=dt_effective,
        use_equilibrium=use_equilibrium,
        track_timeseries=track_timeseries,
    )

    years_from_days = days / 365.0
    result_row = _build_saturation_range_result_row(
        setup=setup,
        sim=sim,
        years=years_from_days,
        kg_rocks_dict=kg_rocks_dict,
    )

    if track_timeseries and "timeseries" in sim:
        result_row["timeseries"] = sim["timeseries"]

    return result_row
