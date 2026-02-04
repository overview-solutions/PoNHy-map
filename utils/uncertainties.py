from __future__ import annotations

import math
import os
import sys
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils.helpers import _clamp_worker_count, _get_progress_bar, _sample_unit_hypercube
from utils.no_saturation import compute_h2_production_no_saturation
from utils.plotting import _save_plot_pair
from utils.reporting import save_mc_convergence_sweep_report
from utils.saturation import run_saturation_monte_carlo


# ======================================================== Fracture Monte Carlo ========================================================

def run_fracture_monte_carlo_simulation(
    flow_target_fracture_config: Dict[str, float],
    *,
    flow_target: float,
    n_samples: int = 100000,
    verbose: bool = False,
    make_plot: bool = False,
    width_size: int = 100000,
    fractured_length_size: int = 100000,
    depth_grid_num: int = 100,
    depth_samples_size: int = 100000,
    fracture_density_size: int = 100000,
    delta_p_num: int = 100,
    mu_num: int = 100,
    connection_fraction_size: int = 100000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Monte Carlo sampling of fracture flow using Darcy's law."""

    l_fault = float(flow_target_fracture_config["L_fault"])
    width_min = float(flow_target_fracture_config["width_min"])
    width_max = float(flow_target_fracture_config["width_max"])
    frac_l_min = float(flow_target_fracture_config["fractured_length_min"])
    frac_l_max = float(flow_target_fracture_config["fractured_length_max"])
    d_min = float(flow_target_fracture_config["D_min"])
    d_max = float(flow_target_fracture_config["D_max"])
    fd_min = float(flow_target_fracture_config["fracture_density_min"])
    fd_max = float(flow_target_fracture_config["fracture_density_max"])
    dp_min = float(flow_target_fracture_config["deltaP_min"])
    dp_max = float(flow_target_fracture_config["deltaP_max"])
    mu_min = float(flow_target_fracture_config["mu_min"])
    mu_max = float(flow_target_fracture_config["mu_max"])
    conn_min = float(flow_target_fracture_config["connection_fraction_min"])
    conn_max = float(flow_target_fracture_config["connection_fraction_max"])

    width_vals = np.random.uniform(width_min, width_max, width_size)
    fractured_length_fraction = np.random.uniform(frac_l_min, frac_l_max, fractured_length_size)
    d_vals = np.linspace(d_min, d_max, depth_grid_num)
    depth_samples = np.random.choice(d_vals, size=depth_samples_size)

    fracture_density = (np.random.uniform(fd_min, fd_max, fracture_density_size) * 1000 / depth_samples)
    n_fractures = fracture_density * width_vals

    a_vals = n_fractures * (l_fault * fractured_length_fraction) * depth_samples
    delta_p_vals = np.linspace(dp_min, dp_max, delta_p_num)
    mu_vals = np.linspace(mu_min, mu_max, mu_num)
    conn_vals = np.random.uniform(conn_min, conn_max, connection_fraction_size)

    k_vals = 10 ** (
        -3.2 * np.log10(depth_samples / 1000) - 14 + np.random.uniform(-0.3, 0.3, depth_samples.size)
    )

    a_raw = np.random.choice(a_vals, size=n_samples)
    conn = np.random.choice(conn_vals, size=n_samples)
    a_eff = a_raw * conn
    d_len = np.random.choice(depth_samples, size=n_samples)
    dp = np.random.choice(delta_p_vals, size=n_samples) * 1e6
    mu = np.random.choice(mu_vals, size=n_samples)
    k = np.random.choice(k_vals, size=n_samples)

    flow_target_m3s = float(flow_target) / 1000.0 / 86400.0
    q_m3s = (k * a_eff * dp) / (mu * d_len)
    success = q_m3s >= flow_target_m3s
    success_rate = float(np.mean(success) * 100.0) if success.size else 0.0

    if verbose:
        print(
            f"[Fracture MC] Success rate: {success_rate:.1f}% | "
            f"target={flow_target:.2e} L/day | samples={n_samples}"
        )

    _ = make_plot
    return k, success


# ======================================================== Convergence sweeps ==========================================================

def _iter_totals_from_df(df: Optional[pd.DataFrame], result_col: str) -> pd.Series:
    if df is None or df.empty or "iter" not in df.columns:
        return pd.Series(dtype=float)
    if result_col not in df.columns:
        return pd.Series(dtype=float)
    return df.groupby("iter")[result_col].sum()


def run_mc_convergence_sweep(
    config: Dict[str, Any],
    base_df: Optional[pd.DataFrame],
    mc_kwargs: Dict[str, Any],
    results_dir: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Sweep Monte Carlo iterations to check mean/std stability of total H₂."""

    iter_values = sorted(config.get("iter_values", []))
    if not iter_values:
        print("[MC Convergence] No iter_values configured.")
        return None

    result_col = config.get("result_column", "H2 total [tons]")
    reuse_from_single_run = bool(config.get("reuse_from_single_run", False))
    tolerance_mean = float(config.get("tolerance_mean_pct", 10.0))
    tolerance_std = float(config.get("tolerance_std_pct", 5.0))
    silence_runs = bool(config.get("silence_runs", True))

    results: List[Dict[str, Any]] = []
    prev_mean = None
    prev_std = None

    def run_mc_for_iterations(n_value: int) -> Optional[pd.DataFrame]:
        if reuse_from_single_run and base_df is not None:
            return base_df
        local_kwargs = dict(mc_kwargs)
        local_kwargs.setdefault("verbose", False)
        local_kwargs.setdefault("show_progress", False)
        stats = run_saturation_monte_carlo(n_iter=int(n_value), **local_kwargs)
        if stats is None:
            return None
        return stats.attrs.get("df_all")

    pbar = _get_progress_bar(
        total=int(len(iter_values)),
        desc="MC Convergence Sweep",
        file=sys.stdout,
        leave=True,
        position=0,
    )
    pbar.refresh()

    for idx, n in enumerate(iter_values, start=1):
        pbar.set_description(f"MC Convergence Sweep ({idx}/{len(iter_values)}) | n_iter={n}")
        df_subset = run_mc_for_iterations(int(n))
        iter_totals = _iter_totals_from_df(df_subset, result_col)

        if iter_totals.empty:
            if not silence_runs:
                tqdm.write(f"[MC Convergence] n_iter={n}: unable to retrieve MC results; skipping.")
            pbar.update(1)
            pbar.refresh()
            continue

        if reuse_from_single_run and len(iter_totals) >= n:
            iter_totals = iter_totals.iloc[: int(n)]

        sample_count = len(iter_totals)
        if sample_count == 0:
            if not silence_runs:
                tqdm.write(f"[MC Convergence] n_iter={n}: no totals found; skipping.")
            pbar.update(1)
            pbar.refresh()
            continue

        mean_val = float(iter_totals.mean())
        std_val = float(iter_totals.std(ddof=0) if sample_count > 1 else 0.0)
        prev_mean_val = 0.0 if prev_mean is None else float(prev_mean)
        prev_std_val = 0.0 if prev_std is None else float(prev_std)
        mean_delta = 0.0 if prev_mean_val == 0.0 else abs(mean_val - prev_mean_val) / abs(prev_mean_val) * 100
        std_delta = 0.0 if prev_std_val == 0.0 else abs(std_val - prev_std_val) / abs(prev_std_val) * 100
        meets_tol = (prev_mean is not None) and (mean_delta <= tolerance_mean) and (std_delta <= tolerance_std)

        results.append(
            {
                "n_iter": n,
                "mean_total_tons": mean_val,
                "std_total_tons": std_val,
                "mean_delta_pct": mean_delta,
                "std_delta_pct": std_delta,
                "meets_tolerance": meets_tol,
            }
        )

        prev_mean = mean_val
        prev_std = std_val
        pbar.update(1)
        pbar.refresh()

    pbar.close()
    sweep_df = pd.DataFrame(results)
    if results_dir and sweep_df is not None:
        save_mc_convergence_sweep_report(sweep_df, results_dir, save_plot=bool(config.get("save_plot", True)))

    return sweep_df


def run_dt_convergence_sweep(
    config: Dict[str, Any],
    mc_kwargs: Dict[str, Any],
    results_dir: Optional[str] = None,
    *,
    mc_saturation_config: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """Sweep over dt_day values and track convergence of Monte Carlo totals."""

    dt_values = config.get("dt_values") or config.get("range") or []
    dt_values = sorted([float(v) for v in dt_values if float(v) > 0])
    if not dt_values:
        print("[DT Convergence] No dt_values configured.")
        return None

    config_mc = mc_saturation_config or {}
    result_col = config.get("result_column", "H2 total [tons]")
    n_iter = int(config.get("n_iter", config_mc.get("n_iter", 100)))
    silence_runs = bool(config.get("silence_runs", True))

    def _generate_fixed_custom_factors_list(n: int) -> List[Optional[Dict[str, float]]]:
        sampling = str(config_mc.get("sampling", "random")).lower()
        sampling_seed_value = config_mc.get("sampling_seed", seed)
        sampling_seed = None if sampling_seed_value is None else int(sampling_seed_value)

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

        if sampling == "lhs":
            d = len(range_specs)
            U = np.asarray(_sample_unit_hypercube(n=n, d=d, mode="lhs", seed=sampling_seed), dtype=float)
            return [
                {name: float(a + U[i, j] * (b - a)) for j, (name, (a, b)) in enumerate(range_specs)}
                for i in range(n)
            ]
        if sampling == "sobol":
            d = len(range_specs)
            U = np.asarray(_sample_unit_hypercube(n=n, d=d, mode="sobol", seed=sampling_seed), dtype=float)
            return [
                {name: float(a + U[i, j] * (b - a)) for j, (name, (a, b)) in enumerate(range_specs)}
                for i in range(n)
            ]

        return [{name: 1.0 for name, _ in range_specs} for _ in range(n)]

    fixed_custom_factors_list = _generate_fixed_custom_factors_list(n_iter)

    def run_mc_for_dt(dt_value: float) -> Optional[pd.DataFrame]:
        local_kwargs = dict(mc_kwargs)
        local_kwargs.setdefault("verbose", False)
        local_kwargs.setdefault("show_progress", False)
        stats_tmp = run_saturation_monte_carlo(
            n_iter=n_iter,
            dt_day=dt_value,
            custom_factors_list=fixed_custom_factors_list,
            **local_kwargs,
        )
        if stats_tmp is None:
            return None
        df_tmp = stats_tmp.attrs.get("df_all")
        if df_tmp is None or df_tmp.empty:
            return None
        return df_tmp

    results: List[Dict[str, Any]] = []
    prev_mean = None
    prev_std = None

    pbar = _get_progress_bar(
        total=int(len(dt_values)),
        desc="DT Convergence Sweep",
        file=sys.stdout,
        leave=True,
        position=0,
    )
    pbar.refresh()

    for idx, dt_value in enumerate(dt_values, start=1):
        pbar.set_description(f"DT Convergence Sweep ({idx}/{len(dt_values)}) | dt_day={dt_value:g}")
        df_subset = run_mc_for_dt(dt_value)
        iter_totals = _iter_totals_from_df(df_subset, result_col)

        if iter_totals.empty:
            if not silence_runs:
                tqdm.write(f"[DT Convergence] dt_day={dt_value:g}: no totals found; skipping.")
            pbar.update(1)
            pbar.refresh()
            continue

        mean_val = float(iter_totals.mean())
        std_val = float(iter_totals.std(ddof=0) if len(iter_totals) > 1 else 0.0)
        prev_mean_val = 0.0 if prev_mean is None else float(prev_mean)
        prev_std_val = 0.0 if prev_std is None else float(prev_std)
        mean_delta = 0.0 if prev_mean_val == 0.0 else abs(mean_val - prev_mean_val) / abs(prev_mean_val) * 100
        std_delta = 0.0 if prev_std_val == 0.0 else abs(std_val - prev_std_val) / abs(prev_std_val) * 100

        results.append(
            {
                "dt_day": float(dt_value),
                "mean_total_tons": mean_val,
                "std_total_tons": std_val,
                "mean_delta_pct": mean_delta,
                "std_delta_pct": std_delta,
            }
        )

        prev_mean = mean_val
        prev_std = std_val
        pbar.update(1)
        pbar.refresh()

    pbar.close()
    sweep_df = pd.DataFrame(results)

    if results_dir and sweep_df is not None and not sweep_df.empty:
        if bool(config.get("save_plot", True)):
            fig, (ax_mean, ax_std) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
            ax_mean.plot(sweep_df["dt_day"], sweep_df["mean_total_tons"], marker="o", label="Mean")
            ax_mean.set_ylabel("Mean H2 total [tons]")
            ax_mean.grid(True)
            ax_mean.legend(loc="best")
            ax_std.plot(sweep_df["dt_day"], sweep_df["std_total_tons"], marker="s", color="darkorange", label="Std")
            ax_std.set_ylabel("Std H2 total [tons]")
            ax_std.set_xlabel("dt_day")
            ax_std.grid(True)
            ax_std.legend(loc="best")
            fig.tight_layout()
            plot_path = os.path.join(results_dir, "dt_convergence_sweep")
            _save_plot_pair(plot_path, fig, dpi=300)
            plt.close(fig)

        csv_path = os.path.join(results_dir, "dt_convergence_sweep.csv")
        sweep_df.to_csv(csv_path, index=False)

    return sweep_df


# ======================================================== Univariate sweeps ==========================================================

def _run_no_sat_single(
    mc_kwargs: Dict[str, Any],
    deterministic_factors: Dict[str, float],
) -> float:
    results_temp_volumetric, *_ = compute_h2_production_no_saturation(
        **mc_kwargs,
        deterministic_factors=deterministic_factors,
        show_progress=False,
    )
    total_kg_day = sum(v for v in results_temp_volumetric.values() if np.isfinite(v))
    return (total_kg_day * 365.0) / 1000.0


def run_no_saturation_univariate_sweep(
    mc_kwargs: Dict[str, Any],
    base_config: Dict[str, Any],
    uni_config: Dict[str, Any],
    results_dir: Optional[str] = None,
    *,
    n_cores: Optional[int] = None,
    seed: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """One-at-a-time sweep for the no-saturation volumetric Monte Carlo."""

    params = [
        "v_ref_range",
        "prod_rate_range",
        "volume_range",
        "serp_correction_range",
        "surface_area_range",
    ]

    n_points = max(3, int(uni_config.get("n_points", 5)))
    n_rep = max(1, int(uni_config.get("n_rep", 5)))
    baseline_n_iter = max(1, int(uni_config.get("baseline_n_iter", n_rep)))

    _ = _clamp_worker_count(n_cores)
    rng = np.random.default_rng(seed)
    _ = rng

    def _range_for(param_name: str) -> Tuple[float, float]:
        lo, hi = base_config.get(param_name, (1.0, 1.0))
        return float(lo), float(hi)

    baseline_factors = {param: 1.0 for param in params}
    baseline_totals = [_run_no_sat_single(mc_kwargs, baseline_factors) for _ in range(baseline_n_iter)]
    baseline_mean = float(np.mean(baseline_totals)) if baseline_totals else 0.0

    rows: List[Dict[str, Any]] = []
    for param in params:
        lo, hi = _range_for(param)
        samples = np.linspace(lo, hi, n_points)
        totals: List[float] = []
        for value in samples:
            factors = dict(baseline_factors)
            factors[param] = float(value)
            for _ in range(n_rep):
                totals.append(_run_no_sat_single(mc_kwargs, factors))

        mean_val = float(np.mean(totals)) if totals else 0.0
        std_val = float(np.std(totals, ddof=0)) if len(totals) > 1 else 0.0
        rows.append(
            {
                "parameter": param,
                "range_min": lo,
                "range_max": hi,
                "mean_total_tons": mean_val,
                "std_total_tons": std_val,
                "delta_mean": mean_val - baseline_mean,
                "delta_mean_pct": (mean_val - baseline_mean) / baseline_mean * 100 if baseline_mean else 0.0,
            }
        )

    df_results = pd.DataFrame(rows)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        csv_path = os.path.join(results_dir, "univariate_no_saturation.csv")
        df_results.to_csv(csv_path, index=False)

    return df_results


def run_saturation_univariate_sweep(
    mc_kwargs: Dict[str, Any],
    base_config: Dict[str, Any],
    uni_config: Dict[str, Any],
    results_dir: Optional[str] = None,
    *,
    n_cores: Optional[int] = None,
    seed: Optional[int] = None,
    mc_saturation_config: Optional[Dict[str, Any]] = None,
) -> Optional[pd.DataFrame]:
    """One-at-a-time sweep for saturation Monte Carlo factors."""

    params = [
        "vol_range",
        "mean_press_range",
        "serp_deg_range",
        "spacing_range",
        "perm_range",
        "prod_rate_range",
        "d_range",
        "kg_rocks_range",
        "solubility_scaling_range",
    ]

    n_points = max(3, int(uni_config.get("n_points", 5)))
    n_rep = max(1, int(uni_config.get("n_rep", 5)))
    baseline_n_iter = max(1, int(uni_config.get("baseline_n_iter", n_rep)))

    _ = _clamp_worker_count(n_cores)
    _ = seed
    config_mc = mc_saturation_config or base_config

    def _range_for(param_name: str) -> Tuple[float, float]:
        lo, hi = config_mc.get(param_name, (1.0, 1.0))
        return float(lo), float(hi)

    baseline_factors = {param: 1.0 for param in params}
    baseline_factor_list = cast(List[Optional[Dict[str, float]]], [dict(baseline_factors) for _ in range(baseline_n_iter)])

    baseline_stats = run_saturation_monte_carlo(
        n_iter=baseline_n_iter,
        custom_factors_list=baseline_factor_list,
        **mc_kwargs,
    )
    baseline_totals = baseline_stats.attrs.get("iter_totals", []) if baseline_stats is not None else []
    baseline_mean = float(np.mean(baseline_totals)) if len(baseline_totals) else 0.0

    rows: List[Dict[str, Any]] = []
    for param in params:
        lo, hi = _range_for(param)
        samples = np.linspace(lo, hi, n_points)
        totals: List[float] = []
        for value in samples:
            factors = dict(baseline_factors)
            factors[param] = float(value)
            factors_list = cast(List[Optional[Dict[str, float]]], [dict(factors) for _ in range(n_rep)])
            stats = run_saturation_monte_carlo(
                n_iter=n_rep,
                custom_factors_list=factors_list,
                **mc_kwargs,
            )
            iter_totals = stats.attrs.get("iter_totals", []) if stats is not None else []
            totals.extend(iter_totals)

        mean_val = float(np.mean(totals)) if totals else 0.0
        std_val = float(np.std(totals, ddof=0)) if len(totals) > 1 else 0.0
        rows.append(
            {
                "parameter": param,
                "range_min": lo,
                "range_max": hi,
                "mean_total_tons": mean_val,
                "std_total_tons": std_val,
                "delta_mean": mean_val - baseline_mean,
                "delta_mean_pct": (mean_val - baseline_mean) / baseline_mean * 100 if baseline_mean else 0.0,
            }
        )

    df_results = pd.DataFrame(rows)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        csv_path = os.path.join(results_dir, "univariate_saturation.csv")
        df_results.to_csv(csv_path, index=False)

    return df_results


# ======================================================== Limiting factors ==========================================================

def analyze_limiting_factors_by_flow_target(
    FLOW_TARGET_FRACTURE_CONFIG: Dict[str, Any],
    volume_at_temperature: Dict[str, float],
    df_saturation_table: pd.DataFrame,
    mean_pressure_ranges: Dict[str, float],
    serpentinization_degree: float,
    int_fracture_spacing: float,
    permeability_fractures: float,
    production_rate_volumetric: Dict[str, Any],
    years: float,
    dist_x: float,
    dist_y: float,
    dist_z: float,
    kg_rocks_dict: Optional[Dict[str, float]],
    *,
    flow_target_log_min: float,
    flow_target_log_max: float,
    flow_target_n_samples: int,
    mc_flow_target_config: Dict[str, Any],
    mc_saturation_config: Dict[str, Any],
    porosity_front: float,
    density_serpentinite: float,
    results_path: Optional[str] = None,
    total_tons_no_sat: float = 0.0,
    n_cores: Optional[int] = None,
    seed: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """Sweep flow targets and record Monte Carlo H₂ totals plus limiting factors."""

    _ = FLOW_TARGET_FRACTURE_CONFIG
    flow_targets = np.logspace(
        np.log10(flow_target_log_min),
        np.log10(flow_target_log_max),
        int(flow_target_n_samples),
    ).flatten()

    n_iter = int(mc_flow_target_config.get("n_iter", 50))
    show_progress = bool(mc_flow_target_config.get("show_progress", False))
    save_timeseries_plots = bool(mc_flow_target_config.get("save_timeseries_plots", False))

    results: List[Dict[str, Any]] = []
    flow_stats: Dict[float, Dict[str, Any]] = {}

    progress_ctx = (
        _get_progress_bar(total=len(flow_targets), desc="Analyzing flow targets", file=sys.stdout)
        if show_progress
        else nullcontext()
    )

    with progress_ctx as pbar:
        for flow_target in flow_targets:
            stats = run_saturation_monte_carlo(
                n_iter=n_iter,
                volume_at_temperature=volume_at_temperature,
                df_saturation_table=df_saturation_table,
                mean_pressure_ranges=mean_pressure_ranges,
                serpentinization_degree=serpentinization_degree,
                int_fracture_spacing=int_fracture_spacing,
                permeability_fractures=permeability_fractures,
                flow_target=float(flow_target),
                production_rate_volumetric=production_rate_volumetric,
                years=int(years),
                dist_x=dist_x,
                dist_y=dist_y,
                dist_z=dist_z,
                kg_rocks_dict=kg_rocks_dict,
                show_progress=False,
                track_timeseries=save_timeseries_plots,
                mc_config=mc_saturation_config,
                n_cores=n_cores,
                seed=seed,
                porosity_front=porosity_front,
                density_serpentinite=density_serpentinite,
            )

            mean_col = ("H2 total [tons]", "mean")
            std_col = ("H2 total [tons]", "std")
            total_tons = float(stats[mean_col].sum()) if stats is not None and mean_col in stats.columns else 0.0
            std_total = float(stats[std_col].sum()) if stats is not None and std_col in stats.columns else 0.0
            limiter = stats.attrs.get("dominant_limiter", "-") if stats is not None else "-"
            efficiency = (total_tons / total_tons_no_sat * 100) if total_tons_no_sat > 0 else 0.0

            row = {
                "Flow target [L/day]": float(flow_target),
                "H2 total [tons]": total_tons,
                "Std total [tons]": std_total,
                "Efficiency [%]": efficiency,
                "Limiting factor": limiter,
            }
            results.append(row)
            flow_stats[float(flow_target)] = row

            if pbar is not None:
                pbar.update(1)

    if not results:
        return None

    df_results_limiting = pd.DataFrame(results).sort_values("Flow target [L/day]")
    df_results_limiting.attrs["flow_stats_per_target"] = flow_stats

    if results_path:
        os.makedirs(results_path, exist_ok=True)
        csv_path = os.path.join(results_path, "limiting_factor_vs_flow_mc.csv")
        df_results_limiting.to_csv(csv_path, index=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            df_results_limiting["Flow target [L/day]"],
            df_results_limiting["H2 total [tons]"],
            marker="o",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Flow Target [L/day]")
        ax.set_ylabel("H2 Total [tons]")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        _save_plot_pair(os.path.join(results_path, "limiting_factor_vs_flow"), fig, dpi=300)
        plt.close(fig)

    return df_results_limiting

