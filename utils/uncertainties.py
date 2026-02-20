from __future__ import annotations

import csv
import math
import os
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager, nullcontext, redirect_stdout
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils.helpers import _clamp_worker_count, _get_progress_bar, _sample_unit_hypercube
from utils.no_saturation import (
    _clear_no_saturation_worker_context,
    _init_no_saturation_worker_context,
    _run_no_saturation_sample,
    compute_h2_production_no_saturation,
)
from utils.plotting import _save_plot_pair, get_plot_save_svg
from utils.reporting import save_mc_convergence_sweep_report
from utils.config import FlowTargetLimitingFactorsParams
from utils.saturation import run_saturation_monte_carlo

# Keep SVG text as editable text (not paths).
matplotlib.rcParams["svg.fonttype"] = "none"


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
    reuse_from_single_run = bool(config.get("reuse_from_single_run", True))
    tolerance_mean = float(config.get("tolerance_mean_pct", 1.0))
    tolerance_std = float(config.get("tolerance_std_pct", 0.5))
    silence_runs = bool(config.get("silence_runs", True))

    df_source_single = base_df if reuse_from_single_run else None
    unique_iters_single: List[int] = []
    if reuse_from_single_run:
        if (
            df_source_single is not None
            and "iter" in df_source_single.columns
            and df_source_single["iter"].nunique() >= max(iter_values)
        ):
            unique_iters_single = sorted(df_source_single["iter"].unique())
        else:
            reuse_from_single_run = False
            df_source_single = None

    def run_mc_for_iterations(n_value: int, sweep_callback=None) -> Optional[pd.DataFrame]:
        local_kwargs = dict(mc_kwargs)
        local_kwargs.setdefault("verbose", False)
        local_kwargs.setdefault("show_progress", False)
        if not silence_runs:
            print(f"[MC Convergence] Running Monte Carlo with n_iter={n_value} ...")
        prev_flag = globals().get("MC_SWEEP_PROGRESS_ACTIVE", False)
        globals()["MC_SWEEP_PROGRESS_ACTIVE"] = True
        local_bar = _get_progress_bar(
            total=int(n_value),
            desc=f"MC n_iter={int(n_value)}",
            file=sys.stdout,
            position=1,
            leave=False,
        )

        def combined_callback(completed, total):
            if total > 0:
                local_bar.total = total
            local_bar.n = completed
            local_bar.refresh()
            if sweep_callback is not None:
                try:
                    sweep_callback(completed, total)
                except Exception:
                    pass

        devnull = None
        try:
            context_mgr = nullcontext()
            if silence_runs:
                devnull = open(os.devnull, "w")
                context_mgr = redirect_stdout(devnull)
            with context_mgr:
                stats_tmp = run_saturation_monte_carlo(
                    n_iter=n_value,
                    progress_desc=f"MC n_iter={n_value}",
                    progress_position=1,
                    progress_leave=False,
                    progress_callback=combined_callback,
                    **local_kwargs,
                )
        finally:
            globals()["MC_SWEEP_PROGRESS_ACTIVE"] = prev_flag
            local_bar.close()
            if devnull is not None:
                devnull.close()

        if stats_tmp is None:
            return None
        df_tmp = stats_tmp.attrs.get("df_all")
        if df_tmp is None or df_tmp.empty:
            return None
        return df_tmp

    results: List[Dict[str, Any]] = []
    prev_mean = None
    prev_std = None

    def _bounded_update(bar, amount: float) -> None:
        """Update tqdm without letting the counter exceed its total (prevents frac>1 warnings)."""
        if bar is None or amount is None:
            return
        try:
            amt = float(amount)
        except Exception:
            return
        if amt <= 0:
            return
        if bar.total is None:
            bar.update(amt)
            return
        remaining = bar.total - bar.n
        if remaining <= 0:
            return
        bar.update(min(amt, remaining))

    pbar = _get_progress_bar(
        total=int(len(iter_values)),
        desc="MC Convergence Sweep",
        file=sys.stdout,
        leave=True,
        position=0,
        bar_format="{l_bar}{bar}| [{elapsed}<{remaining}, {rate_fmt}]",
    )
    pbar.refresh()

    for idx, n in enumerate(iter_values, start=1):
        pbar.set_description(f"MC Convergence Sweep ({idx}/{len(iter_values)}) | n_iter={n}")
        if reuse_from_single_run:
            if len(unique_iters_single) < n:
                available = len(unique_iters_single)
                if not silence_runs:
                    tqdm.write(
                        f"[MC Convergence] Skipping n_iter={n}: only {available} iterations available for reuse."
                    )
                pbar.update(1)
                pbar.refresh()
                continue
            if df_source_single is None:
                if not silence_runs:
                    tqdm.write(f"[MC Convergence] n_iter={n}: missing data for reuse; skipping.")
                pbar.update(1)
                pbar.refresh()
                continue
            selected_iters = set(unique_iters_single[:n])
            df_subset = df_source_single[df_source_single["iter"].isin(selected_iters)]
        else:
            fraction_state = [0.0]

            def sweep_callback(completed, total):
                if total <= 0:
                    return
                frac = completed / total
                if frac < 0:
                    frac = 0.0
                elif frac > 1:
                    frac = 1.0
                delta = frac - fraction_state[0]
                if delta > 0:
                    _bounded_update(pbar, delta)
                    fraction_state[0] = frac
                    pbar.set_postfix_str(f"{completed}/{total}", refresh=False)

            df_subset = run_mc_for_iterations(n, sweep_callback=sweep_callback)
            if df_subset is None or "iter" not in df_subset.columns:
                if not silence_runs:
                    tqdm.write(
                        f"[MC Convergence] n_iter={n}: unable to retrieve Monte Carlo results; skipping."
                    )
                pbar.update(1)
                pbar.refresh()
                continue
            if fraction_state[0] < 1.0:
                _bounded_update(pbar, 1 - fraction_state[0])
            pbar.refresh()

        if df_subset is None or df_subset.empty or "iter" not in df_subset.columns:
            sample_count = 0
            mean_val = std_val = None
        else:
            iter_totals = df_subset.groupby("iter")[result_col].sum()
            if iter_totals.empty:
                sample_count = 0
                mean_val = std_val = None
            else:
                sample_count = len(iter_totals)
                mean_val = float(iter_totals.mean())
                std_val = float(iter_totals.std(ddof=0) if len(iter_totals) > 1 else 0.0)

        if not sample_count:
            if not silence_runs:
                tqdm.write(f"[MC Convergence] n_iter={n}: no totals found; skipping.")
            pbar.update(1)
            pbar.refresh()
            continue

        if mean_val is None or std_val is None:
            if not silence_runs:
                tqdm.write(f"[MC Convergence] n_iter={n}: no totals found; skipping.")
            pbar.update(1)
            pbar.refresh()
            continue

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
    n_rep = max(1, int(uni_config.get("n_rep", 1)))
    baseline_n_iter = max(1, int(uni_config.get("baseline_n_iter", n_rep)))
    seed_value = seed if seed is not None else uni_config.get("sampling_seed", base_config.get("sampling_seed"))
    quiet = bool(uni_config.get("quiet", False))
    show_progress = bool(uni_config.get("show_progress", True) and not quiet)
    requested_workers = n_cores if n_cores is not None else uni_config.get("worker_count", None)
    worker_count = _clamp_worker_count(requested_workers)
    metric = "no_saturation_total_tons"
    metric_desc = "H₂ without solubility cap"
    print_delta_lines = bool(uni_config.get("print_delta_lines", True))
    make_plot = bool(uni_config.get("make_plot", False))
    uni_curves: Dict[str, Dict[str, Any]] = {}

    if not quiet:
        sep_uni_nosat = "=" * 150
        print("\n" + sep_uni_nosat)
        print("UNIVARIATE (NO SATURATION)".center(150))
        print(sep_uni_nosat)
    _ = metric_desc

    baseline_overrides = {k: (1.0, 1.0) for k in params}

    def _clone_overrides(src: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
        return {k: (float(v[0]), float(v[1])) for k, v in src.items()}

    def _build_deterministic(overrides: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        return {
            "v_ref": overrides.get("v_ref_range", (1.0, 1.0))[0],
            "prod_rate": overrides.get("prod_rate_range", (1.0, 1.0))[0],
            "volume": overrides.get("volume_range", (1.0, 1.0))[0],
            "serp_correction": overrides.get("serp_correction_range", (1.0, 1.0))[0],
            "surface_area": overrides.get("surface_area_range", (1.0, 1.0))[0],
        }

    baseline_tasks: List[
        Tuple[str, Optional[str], Optional[float], Optional[Dict[str, float]], Dict[str, Tuple[float, float]]]
    ] = []
    param_task_map: Dict[
        str, List[Tuple[str, Optional[str], Optional[float], Optional[Dict[str, float]], Dict[str, Tuple[float, float]]]]
    ] = {}
    sample_order: Dict[str, List[float]] = {}

    for _ in range(baseline_n_iter):
        baseline_tasks.append(("baseline", None, None, None, _clone_overrides(baseline_overrides)))

    for param in params:
        if param not in base_config:
            continue
        a, b = base_config[param]
        samples = [float(x) for x in np.linspace(float(a), float(b), n_points)]
        sample_order[param] = samples
        param_tasks: List[
            Tuple[str, Optional[str], Optional[float], Optional[Dict[str, float]], Dict[str, Tuple[float, float]]]
        ] = []
        for value in samples:
            overrides = _clone_overrides(baseline_overrides)
            overrides[param] = (value, value)
            det = _build_deterministic(overrides)
            for _ in range(n_rep):
                param_tasks.append(("param", param, value, det.copy(), _clone_overrides(overrides)))
        if param_tasks:
            param_task_map[param] = param_tasks

    worker_context = {"mc_kwargs": mc_kwargs, "sampling_seed": seed_value}

    total_tasks = len(baseline_tasks) + sum(len(tasks) for tasks in param_task_map.values())
    overall_progress = None
    if show_progress and total_tasks > 0:
        print("\n\n", end="")
        overall_progress = _get_progress_bar(
            total=total_tasks,
            desc="Univariate sweep (no saturation)",
            file=sys.stdout,
            leave=False,
            bar_format=" {l_bar}{bar}|",
        )
        overall_progress.refresh()

    def _execute_tasks(task_list, desc):
        if not task_list:
            return []
        progress = overall_progress if show_progress else None
        results_local = []

        def _update():
            if progress is not None:
                progress.update(1)

        if worker_count == 1:
            _init_no_saturation_worker_context(worker_context)
            try:
                for task in task_list:
                    results_local.append(_run_no_saturation_sample(task))
                    _update()
            finally:
                _clear_no_saturation_worker_context()
        else:
            inflight_cap = max(1, worker_count * 2)
            with ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=_init_no_saturation_worker_context,
                initargs=(worker_context,),
            ) as executor:
                task_iter = iter(task_list)
                inflight = set()

                try:
                    for _ in range(inflight_cap):
                        task = next(task_iter)
                        inflight.add(executor.submit(_run_no_saturation_sample, task))
                except StopIteration:
                    pass

                while inflight:
                    done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                    for future in done:
                        results_local.append(future.result())
                        _update()

                    try:
                        for _ in range(inflight_cap - len(inflight)):
                            task = next(task_iter)
                            inflight.add(executor.submit(_run_no_saturation_sample, task))
                    except StopIteration:
                        pass

        if progress is not None:
            progress.refresh()
        return results_local

    def _display_param_name(name: str) -> str:
        return name[:-6] if name.endswith("_range") else name

    task_results: List[Tuple[str, Optional[str], Optional[float], float]] = []

    baseline_results = _execute_tasks(baseline_tasks, "Univar baseline")
    task_results.extend(baseline_results)
    baseline_vals: List[float] = [total for role, _, _, total in baseline_results if role == "baseline"]
    mean_base = float(np.mean(baseline_vals)) if baseline_vals else 0.0

    for param in params:
        param_task_bucket = param_task_map.get(param)
        if not param_task_bucket:
            continue
        pname = _display_param_name(param)
        desc = f"Univar {pname}"
        task_results.extend(_execute_tasks(param_task_bucket, desc))

    if overall_progress is not None:
        overall_progress.close()

    param_results: Dict[str, Dict[float, List[float]]] = {}

    for role, param_name, sample_value, total in task_results:
        if role == "param" and param_name is not None and sample_value is not None:
            param_results.setdefault(param_name, {}).setdefault(float(sample_value), []).append(total)

    results = []
    rows = []

    for param in params:
        samples = sample_order.get(param, [])
        if not samples:
            continue
        buckets = param_results.get(param, {})
        x_vals: List[float] = []
        y_vals: List[float] = []
        y_min_vals: List[float] = []
        y_max_vals: List[float] = []

        a, b = base_config.get(param, (1.0, 1.0))
        for v in samples:
            vals = buckets.get(float(v), [])
            if not vals:
                continue
            vals_arr = np.asarray(vals, dtype=float)
            y_mean = float(vals_arr.mean())
            y_min = float(vals_arr.min())
            y_max = float(vals_arr.max())
            x_vals.append(v)
            y_vals.append(y_mean)
            y_min_vals.append(y_min)
            y_max_vals.append(y_max)

        if not x_vals:
            continue

        x_arr = np.asarray(x_vals, dtype=float)
        y_arr = np.asarray(y_vals, dtype=float)
        mean_low = float(y_arr[0])
        mean_high = float(y_arr[-1])
        mean_val = float(y_arr.mean())
        std_val = float(y_arr.std(ddof=0)) if len(y_arr) > 1 else 0.0
        delta_mean = mean_high - mean_low
        delta_mean_pct = (delta_mean / mean_base * 100.0) if mean_base not in (0.0, -0.0) else 0.0

        pct_std = (100.0 * std_val / mean_val) if mean_val not in (0.0, -0.0) else 0.0
        span_factor = float(b) - float(a)
        sensitivity_tpy_per_factor = delta_mean / span_factor if span_factor not in (0.0, -0.0) else 0.0
        factor_span_pct = span_factor * 100.0
        elasticity_pct_per_pctfactor = delta_mean_pct / factor_span_pct if factor_span_pct not in (0.0, -0.0) else 0.0
        dep_note: List[str] = []

        results.append(
            (
                param,
                mean_val,
                std_val,
                pct_std,
                mean_low,
                mean_high,
                delta_mean,
                delta_mean_pct,
                sensitivity_tpy_per_factor,
                elasticity_pct_per_pctfactor,
                (float(a), float(b)),
                dep_note,
            )
        )
        rows.append(
            {
                "parameter": param,
                "mean_total_tons": mean_val,
                "std_total_tons": std_val,
                "pct_std": pct_std,
                "sensitivity_tpy_per_factor": sensitivity_tpy_per_factor,
                "elasticity_pct_per_pctfactor": elasticity_pct_per_pctfactor,
                "std_share_pct": None,
                "mean_low": mean_low,
                "mean_high": mean_high,
                "delta_mean": delta_mean,
                "delta_mean_pct_of_baseline": delta_mean_pct,
                "sampled_range": f"[{a}, {b}]",
                "dependencies": "; ".join(dep_note) if dep_note else "independent",
                "metric": metric,
            }
        )

        if make_plot and x_vals:
            order = np.argsort(x_vals)
            uni_curves[param] = {
                "x": np.asarray(x_vals, dtype=float)[order],
                "y_mean": np.asarray(y_vals, dtype=float)[order],
                "y_min": np.asarray(y_min_vals, dtype=float)[order],
                "y_max": np.asarray(y_max_vals, dtype=float)[order],
                "range": (float(a), float(b)),
                "ylabel": "H₂ Total [tons/yr]",
                "xlabel": f"{param} factor",
            }

        _ = print_delta_lines

    if not results:
        return None

    total_std = sum(r[2] for r in results) or 0.0
    std_map = {param: std_val for param, _, std_val, *_ in results}
    for row in rows:
        param_key = str(row.get("parameter", ""))
        std_val_row = std_map.get(param_key, 0.0)
        row["std_share_pct"] = 100.0 * std_val_row / total_std if total_std not in (0.0, -0.0) else 0.0

    results.sort(key=lambda x: x[7], reverse=True)
    metric_label = "H₂ final (no cap)"
    pbar = _get_progress_bar(
        total=1,
        desc="Univariate table (no saturation)",
        file=sys.stdout,
        bar_format=" {l_bar}{bar}|",
    )
    pbar.update(1)
    pbar.close()
    print(f"\nUnivariate sensitivity ({metric}) — metric={metric_label} (descending Δmean % of base):")
    print("-" * 150)
    print(
        f"{'Parameter':<18}"
        f"{'Range':<16}"
        f"{'Mean [t]':>14}"
        f"{'Mean min':>14}"
        f"{'Mean max':>14}"
        f"{'Δmean':>14}"
        f"{'Δ% base':>12}"
        f"{'Elasticity ':>14}"
        f"{'Deps' :>16}"
    )
    print("-" * 150)
    for (
        param,
        mean_val,
        std_val,
        pct_std,
        mean_low,
        mean_high,
        delta_mean,
        delta_mean_pct,
        sensitivity_tpy_per_factor,
        elasticity_pct_per_pctfactor,
        sampled_range,
        dep_note,
    ) in results:
        pname = param[:-6] if param.endswith("_range") else param
        dep_txt = ", ".join(dep_note) if dep_note else "independent"
        print(
            f"{pname:<18}"
            f"{sampled_range!s:<16}"
            f"{mean_val:>14.3f}"
            f"{mean_low:>14.3f}"
            f"{mean_high:>14.3f}"
            f"{delta_mean:>14.3f}"
            f"{delta_mean_pct:>12.2f}"
            f"{elasticity_pct_per_pctfactor:>16.4f}"
            f"{dep_txt:>16}"
        )
    print("-" * 150)
    print("Column meanings (table):")
    print("  Mean [t]: Mean no-cap H₂ when only that factor moves.")
    print("  Mean min / Mean max: No-cap H₂ with the factor at range min / max (others at 1.0).")
    print("  Δmean: Difference (max - min) in no-cap H₂ due only to that factor.")
    print("  Δ% base: Δmean divided by the baseline mean (all factors = 1.0).")
    print("  Elasticity: % change in H₂ per 1% change in the factor (normalized slope).")
    print("  Range: Factor interval used in the sweep.")
    print("  Deps: Linked variables also affected by that factor (if any).")

    if not results_dir:
        results_dir = globals().get("results_path", None)

    if make_plot and len(uni_curves):
        if results_dir:
            os.makedirs(results_dir, exist_ok=True)

        def _plot_one(ax, param: str, curve: Dict[str, Any]):
            x = curve["x"]
            y_mean = curve["y_mean"]
            ax.plot(x, y_mean, lw=2.0, color="tab:blue")
            try:
                ax.fill_between(x, curve["y_min"], curve["y_max"], color="tab:blue", alpha=0.20)
            except Exception:
                pass
            ax.axvline(1.0, color="k", lw=1.0, ls="--", alpha=0.6)
            ax.set_title(param[:-6] if param.endswith("_range") else param)
            ax.set_xlabel("factor")
            ax.set_ylabel("H₂ total [tons/yr]")
            ax.grid(True, alpha=0.3)
            try:
                ax.set_xlim(min(curve["range"]), max(curve["range"]))
            except Exception:
                pass

        n = len(uni_curves)
        ncols = 3 if n >= 3 else n
        nrows = int(math.ceil(n / max(ncols, 1)))
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(5.4 * ncols, 3.6 * nrows),
            squeeze=False,
        )
        axes_flat = axes.ravel()
        for ax in axes_flat:
            ax.set_visible(False)
        for i, (param, curve) in enumerate(uni_curves.items()):
            ax = axes_flat[i]
            ax.set_visible(True)
            _plot_one(ax, param, curve)

        fig.suptitle(f"Univariate parameter sweep: factor -> H₂ total (metric={metric})", y=0.995)
        fig.tight_layout()

        if results_dir:
            base = os.path.join(results_dir, f"univariate_curves_{metric}")
            try:
                fig.savefig(base + ".png", dpi=300)
            except Exception:
                pass
            if get_plot_save_svg():
                try:
                    fig.savefig(base + ".svg")
                except Exception:
                    pass
        plt.close(fig)

    df_results = pd.DataFrame(rows)

    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        out_path = os.path.join(results_dir, "univariate_sensitivity_no_saturation.csv")
        df_results.to_csv(out_path, index=False)

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
    """One-at-a-time sweep of saturation Monte Carlo factors to rank their impact on H₂ totals."""
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
    dep_map = {
        "d_range": ["dist_x_y_z"],
        "kg_rocks_range": ["Serp front vel"],
    }

    n_points = max(3, int(uni_config.get("n_points", 15)))
    n_rep = max(1, int(uni_config.get("n_rep", uni_config.get("n_iter", 100))))
    baseline_n_iter = max(1, int(uni_config.get("baseline_n_iter", n_rep)))
    seed_value = seed if seed is not None else uni_config.get("sampling_seed", base_config.get("sampling_seed"))
    show_progress = bool(uni_config.get("show_progress", True))
    print_delta_lines = bool(uni_config.get("print_delta_lines", True))
    metric = iter_total_metric = "final"
    metric_desc = "delivered H2 (capped by solubility)"

    worker_count_uni = _clamp_worker_count(n_cores)

    mc_kwargs_clean = dict(mc_kwargs)
    for drop_key in [
        "show_progress",
        "progress_desc",
        "progress_position",
        "custom_factors_list",
        "progress_leave",
        "progress_callback",
    ]:
        mc_kwargs_clean.pop(drop_key, None)
    mc_kwargs_clean["worker_count"] = worker_count_uni

    make_plot = bool(uni_config.get("make_plot", False))
    results: List[
        Tuple[str, float, float, float, List[str], float, float, float, float, float, float, Tuple[float, float]]
    ] = []
    csv_rows: List[Dict[str, Any]] = []
    uni_curves: Dict[str, Dict[str, Any]] = {}

    sep_uni_sat = "=" * 150
    print("\n" + sep_uni_sat)
    print("UNIVARIATE SENTITIVITY ANALYSIS".center(150))
    print(sep_uni_sat)
    _ = metric_desc

    config_mc = mc_saturation_config or base_config

    @contextmanager
    def _force_random_sampling(seed_val):
        prev_sampling = config_mc.get("sampling", "random")
        prev_seed = config_mc.get("sampling_seed", None)
        config_mc["sampling"] = "random"
        config_mc["sampling_seed"] = seed_val
        try:
            yield
        finally:
            config_mc["sampling"] = prev_sampling
            config_mc["sampling_seed"] = prev_seed

    def _run_mc(desc: str, custom_factors_list, n_iter, disable_scale_factor_ws=True):
        return run_saturation_monte_carlo(
            n_iter=n_iter,
            progress_desc=desc,
            show_progress=show_progress or desc == "Univar baseline",
            progress_position=0,
            progress_leave=False,
            custom_factors_list=custom_factors_list,
            iter_total_metric=iter_total_metric,
            seed_base=int(seed_value) if seed_value is not None else None,
            disable_scale_factor_ws=disable_scale_factor_ws,
            **mc_kwargs_clean,
        )

    def _aggregate_curve(x_vals: List[float], iter_totals: pd.Series):
        y_all = np.asarray(iter_totals, dtype=float)
        x_all = np.asarray(x_vals, dtype=float)
        n = int(min(len(x_all), len(y_all)))
        x_all = x_all[:n]
        y_all = y_all[:n]
        df_curve = pd.DataFrame({"x": x_all, "y": y_all})
        grp = df_curve.groupby("x")
        x_unique = grp["y"].mean().index.to_numpy(dtype=float)
        y_mean = grp["y"].mean().to_numpy(dtype=float)
        y_std = grp["y"].std(ddof=0).fillna(0.0).to_numpy(dtype=float)
        y_min = grp["y"].min().to_numpy(dtype=float)
        y_max = grp["y"].max().to_numpy(dtype=float)
        mean_low = float(y_mean[0]) if len(y_mean) else 0.0
        mean_high = float(y_mean[-1]) if len(y_mean) else 0.0
        mean_val = float(np.mean(y_mean)) if len(y_mean) else 0.0
        std_val = float(np.std(y_mean, ddof=0)) if len(y_mean) > 1 else 0.0
        return x_unique, y_mean, y_min, y_max, mean_low, mean_high, mean_val, std_val

    baseline_keys = params
    baseline_factor_dict = {k: 1.0 for k in baseline_keys}
    baseline_factors = cast(
        List[Optional[Dict[str, float]]],
        [dict(baseline_factor_dict) for _ in range(baseline_n_iter)],
    )

    with _force_random_sampling(seed_value):
        stats_base = _run_mc("Univar baseline", baseline_factors, baseline_n_iter)
        iter_totals_base = (
            stats_base.attrs.get("iter_totals", pd.Series(dtype=float)) if stats_base is not None else pd.Series(dtype=float)
        )
        mean_base = float(iter_totals_base.mean()) if len(iter_totals_base) else 0.0
        dominant_limiter = stats_base.attrs.get("dominant_limiter", "-") if stats_base is not None else "-"
        _ = dominant_limiter

        for param in params:
            if param not in base_config:
                continue

            a, b = base_config[param]
            samples = np.linspace(float(a), float(b), n_points)
            x_vals: List[float] = []
            custom_factors: List[Optional[Dict[str, float]]] = []
            for v in samples:
                for _ in range(n_rep):
                    x_vals.append(float(v))
                    d = dict(baseline_factor_dict)
                    d[param] = float(v)
                    custom_factors.append(d)

            stats_uni = _run_mc(
                f"Univar {param}",
                custom_factors,
                len(custom_factors),
                disable_scale_factor_ws=bool(uni_config.get("disable_ws_random", True)),
            )
            if stats_uni is None or stats_uni.empty:
                continue

            iter_totals = stats_uni.attrs.get("iter_totals", pd.Series(dtype=float))
            (
                x_unique,
                y_mean,
                y_min,
                y_max,
                mean_low,
                mean_high,
                mean_val,
                std_val,
            ) = _aggregate_curve(x_vals, iter_totals)

            delta_mean = mean_high - mean_low
            delta_mean_pct = (delta_mean / mean_base * 100.0) if mean_base not in (0.0, -0.0) else 0.0
            span_factor = float(b) - float(a)
            sensitivity_tons_per_factor = delta_mean / span_factor if span_factor not in (0.0, -0.0) else 0.0
            factor_span_pct = span_factor * 100.0
            elasticity_pct_per_pctfactor = delta_mean_pct / factor_span_pct if factor_span_pct not in (0.0, -0.0) else 0.0
            dep_note = dep_map.get(param, [])
            sampled_range = (float(a), float(b))
            pct_std = (100.0 * std_val / mean_val) if mean_val not in (0.0, -0.0) else 0.0

            results.append(
                (
                    param,
                    mean_val,
                    std_val,
                    pct_std,
                    dep_note,
                    mean_low,
                    mean_high,
                    delta_mean,
                    delta_mean_pct,
                    sensitivity_tons_per_factor,
                    elasticity_pct_per_pctfactor,
                    sampled_range,
                )
            )
            csv_rows.append(
                {
                    "parameter": param,
                    "mean_total_tons": mean_val,
                    "std_total_tons": std_val,
                    "pct_std": pct_std,
                    "sensitivity_tons_per_factor": sensitivity_tons_per_factor,
                    "elasticity_pct_per_pctfactor": elasticity_pct_per_pctfactor,
                    "std_share_pct": None,
                    "mean_low": mean_low,
                    "mean_high": mean_high,
                    "delta_mean": delta_mean,
                    "delta_mean_pct_of_baseline": delta_mean_pct,
                    "sampled_range": f"[{sampled_range[0]}, {sampled_range[1]}]",
                    "dependencies": "; ".join(dep_note) if dep_note else "independent",
                    "metric": metric,
                }
            )

            if make_plot and len(x_unique):
                order = np.argsort(x_unique)
                uni_curves[param] = {
                    "x": x_unique[order],
                    "y_mean": y_mean[order],
                    "y_min": y_min[order],
                    "y_max": y_max[order],
                    "range": sampled_range,
                    "ylabel": "H₂ Total [tons]",
                    "xlabel": f"{param} factor",
                }

            _ = print_delta_lines

    if not results:
        return None

    results.sort(key=lambda x: x[8], reverse=True)
    total_std = sum(r[2] for r in results) or 0.0
    std_map = {param: std_val for param, _, std_val, *_ in results}
    for row in csv_rows:
        param_key = str(row.get("parameter", ""))
        std_val_row = std_map.get(param_key, 0.0)
        row["std_share_pct"] = 100.0 * std_val_row / total_std if total_std not in (0.0, -0.0) else 0.0

    def _display_param_name(name: str) -> str:
        if name == "solubility_scaling_range":
            return "solubility"
        return name[:-6] if name.endswith("_range") else name

    metric_label = "H₂ final (capped)"
    pbar = _get_progress_bar(
        total=1,
        desc="Univariate table (saturation)",
        file=sys.stdout,
        bar_format=" {l_bar}{bar}|",
    )
    pbar.update(1)
    pbar.close()
    print(f"\nUnivariate sensitivity ({metric}) — metric={metric_label} (descending Δmean % of base):")
    print("-" * 150)
    print(
        f"{'Parameter':<18}"
        f"{'Range':<16}"
        f"{'Mean [t]':>14}"
        f"{'Mean min':>14}"
        f"{'Mean max':>14}"
        f"{'Δmean':>14}"
        f"{'Δ% base':>12}"
        f"{'Elasticity ':>14}"
        f"{'Deps':>16}"
    )
    print("-" * 150)
    for (
        param,
        mean_val,
        std_val,
        pct_std,
        dep_note,
        mean_low,
        mean_high,
        delta_mean,
        delta_mean_pct,
        sensitivity_tons_per_factor,
        elasticity_pct_per_pctfactor,
        sampled_range,
    ) in results:
        dep_txt = ", ".join(dep_note) if dep_note else "independent"
        pname = _display_param_name(param)
        print(
            f"{pname:<18}"
            f"{sampled_range!s:<16}"
            f"{mean_val:>14.3f}"
            f"{mean_low:>14.3f}"
            f"{mean_high:>14.3f}"
            f"{delta_mean:>14.3f}"
            f"{delta_mean_pct:>12.2f}"
            f"{elasticity_pct_per_pctfactor:>16.4f}"
            f"{dep_txt:>16}"
        )
    print("-" * 150)
    print("Column meanings (table):")
    print("  Mean [t]: Mean capped H₂ when only that factor moves.")
    print("  Mean min / Mean max: Capped H₂ with the factor at range min / max (others at 1.0).")
    print("  Δmean: Difference (max - min) in capped H₂ due only to that factor.")
    print("  Δ% base: Δmean divided by the baseline mean (all factors = 1.0).")
    print("  Elasticity: % change in capped H₂ per 1% change in the factor (normalized slope).")
    print("  Range: Factor interval used in the sweep.")
    print("  Deps: Linked variables also affected by that factor (if any).")
    print(" ")
    print(" ")

    if make_plot:
        if results_dir:
            os.makedirs(results_dir, exist_ok=True)
        else:
            results_dir = os.getcwd()

    if make_plot and len(uni_curves):

        def _plot_one(ax, param: str, curve: Dict[str, Any]):
            x = curve["x"]
            y_mean = curve["y_mean"]
            ax.plot(x, y_mean, lw=2.0, color="tab:blue")
            try:
                ax.fill_between(x, curve["y_min"], curve["y_max"], color="tab:blue", alpha=0.20)
            except Exception:
                pass
            ax.axvline(1.0, color="k", lw=1.0, ls="--", alpha=0.6)
            ax.set_title(_display_param_name(param))
            ax.set_xlabel("factor")
            ax.set_ylabel("H₂ total [tons]")
            ax.grid(True, alpha=0.3)
            try:
                ax.set_xlim(min(curve["range"]), max(curve["range"]))
            except Exception:
                pass

        n = len(uni_curves)
        ncols = 3 if n >= 3 else n
        nrows = int(math.ceil(n / max(ncols, 1)))
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(5.4 * ncols, 3.6 * nrows),
            squeeze=False,
        )
        axes_flat = axes.ravel()
        for ax in axes_flat:
            ax.set_visible(False)
        for i, (param, curve) in enumerate(uni_curves.items()):
            ax = axes_flat[i]
            ax.set_visible(True)
            _plot_one(ax, param, curve)

        fig.suptitle(f"Univariate parameter sweep: factor -> H₂ total (metric={metric})", y=0.995)
        fig.tight_layout()

        if results_dir:
            base = os.path.join(results_dir, f"univariate_curves_{metric}")
            try:
                fig.savefig(base + ".png", dpi=300)
            except Exception:
                pass
            if get_plot_save_svg():
                try:
                    fig.savefig(base + ".svg")
                except Exception:
                    pass
        plt.close(fig)

    df_results = pd.DataFrame(csv_rows)
    if not results_dir:
        results_dir = globals().get("results_path", None)

    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        out_path = os.path.join(results_dir, "univariate_sensitivity_saturation.csv")
        df_results.to_csv(out_path, index=False)
    return df_results


# ======================================================== Limiting factors ==========================================================

def analyze_limiting_factors_by_flow_target(
    params: FlowTargetLimitingFactorsParams,
) -> Optional[pd.DataFrame]:
    """Sweep flow targets and record Monte Carlo H₂ totals plus limiting factors."""

    _ = params.flow_target_fracture_config
    try:
        flow_target_log_min = float(params.flow_target_log_min)
        flow_target_log_max = float(params.flow_target_log_max)
        flow_target_n_samples = int(params.flow_target_n_samples)
    except Exception:
        return None

    if flow_target_log_min <= 0 or flow_target_log_max <= 0 or flow_target_n_samples <= 0:
        return None

    summaries: List[Dict[str, Any]] = []
    flow_stats: Dict[float, Dict[str, Any]] = {}
    print(" ")
    flow_targets = np.logspace(
        np.log10(flow_target_log_min),
        np.log10(flow_target_log_max),
        flow_target_n_samples,
    ).flatten()
    total_flows = len(flow_targets)
    progress_ctx = (
        _get_progress_bar(range(total_flows), desc="Analyzing flow targets", file=sys.stdout)
        if total_flows > 0
        else nullcontext()
    )
    config = params.mc_flow_target_config
    mc_iterations = int(config.get("n_iter", 50))
    verbose = bool(config.get("verbose", False))
    show_progress = bool(config.get("show_progress", False))
    track_timeseries = bool(config.get("save_timeseries_plots", False))

    total_pore_volume_m3 = float("nan")
    porosity_used = params.porosity_front if params.porosity_front <= 1 else params.porosity_front / 100.0
    if params.kg_rocks_dict:
        try:
            total_pore_volume_m3 = sum(
                (val / params.density_serpentinite) * porosity_used for val in params.kg_rocks_dict.values()
            )
        except Exception:
            total_pore_volume_m3 = float("nan")

    outer_total_tons_no_sat = params.total_tons_no_sat if params.total_tons_no_sat else None

    with progress_ctx as pbar:
        for i, ft in enumerate(flow_targets):
            flow_l_day = float(ft)
            flow_m3_day = flow_l_day / 1000.0 if flow_l_day > 0 else float("nan")
            turnover_days_total = (
                total_pore_volume_m3 / flow_m3_day
                if (flow_m3_day and flow_m3_day > 0 and math.isfinite(total_pore_volume_m3))
                else float("nan")
            )
            config_dt_cap = params.mc_saturation_config.get("dt_day", 1.0)
            if math.isfinite(turnover_days_total) and turnover_days_total > 0:
                dt_for_flow = max(1e-4, min(config_dt_cap, turnover_days_total / 2.0))
            else:
                dt_for_flow = max(1e-4, float(config_dt_cap))

            stats_mc = run_saturation_monte_carlo(
                n_iter=mc_iterations,
                volume_at_temperature=params.volume_at_temperature,
                df_saturation_table=params.df_saturation_table,
                mean_pressure_ranges=params.mean_pressure_ranges,
                serpentinization_degree=params.serpentinization_degree,
                int_fracture_spacing=params.int_fracture_spacing,
                permeability_fractures=params.permeability_fractures,
                flow_target=ft,
                production_rate_volumetric=params.production_rate_volumetric,
                years=int(params.years),
                dist_x=params.dist_x,
                dist_y=params.dist_y,
                dist_z=params.dist_z,
                kg_rocks_dict=params.kg_rocks_dict,
                verbose=verbose,
                show_progress=show_progress,
                track_timeseries=track_timeseries,
                dt_day=dt_for_flow,
                mc_config=params.mc_saturation_config,
                n_cores=params.n_cores,
                seed=params.seed,
                porosity_front=params.porosity_front,
                density_serpentinite=params.density_serpentinite,
            )

            if stats_mc is None or stats_mc.empty:
                if pbar is not None:
                    pbar.update(1)
                continue

            total_tons_sat = 0.0
            std_total_mc = 0.0
            mean_efficiency = 0.0
            mean_col = ("H2 total [tons]", "mean")
            std_col = ("H2 total [tons]", "std")
            if mean_col in stats_mc.columns:
                total_tons_sat = float(stats_mc[mean_col].sum())

            water_cols = {
                "diff": ("daily_diffused_H2O [kg/day]", "mean"),
                "frac": ("daily_fractured_H2O [kg/day]", "mean"),
            }
            delivered_water_kg_day = 0.0
            for col in water_cols.values():
                if col in stats_mc.columns:
                    delivered_water_kg_day += float(stats_mc[col].sum())

            water_kg_total = delivered_water_kg_day * float(params.years) * 365
            raw_total_mol_per_kg = (
                (total_tons_sat * 1000.0) / water_kg_total / 0.002016
                if water_kg_total > 0 else 0.0
            )

            if outer_total_tons_no_sat is not None:
                mean_efficiency = (total_tons_sat / outer_total_tons_no_sat) * 100 if outer_total_tons_no_sat > 0 else 0.0
            else:
                mean_efficiency = 0.0
            dominant_limiter = stats_mc.attrs.get("dominant_limiter", "-")
            iter_totals_attr = stats_mc.attrs.get("iter_totals")
            if iter_totals_attr is not None:
                iter_totals_arr = np.asarray(iter_totals_attr, dtype=float).ravel()
                if iter_totals_arr.size > 1:
                    finite_vals = iter_totals_arr[np.isfinite(iter_totals_arr)]
                    if finite_vals.size > 1:
                        std_total_mc = float(np.std(finite_vals, ddof=0))
            n_iter_effective = int(getattr(iter_totals_attr, "shape", [mc_iterations])[0] or mc_iterations)
            df_all = stats_mc.attrs.get("df_all")
            if isinstance(df_all, pd.DataFrame) and "dt_day_used [day]" in df_all.columns:
                avg_dt_used = float(df_all["dt_day_used [day]"].mean())
            else:
                avg_dt_used = float("nan")
            if isinstance(df_all, pd.DataFrame) and "Pore volume turnover [days]" in df_all.columns:
                mean_turnover_days = float(df_all["Pore volume turnover [days]"].mean())
            else:
                mean_turnover_days = float("nan")
            if isinstance(df_all, pd.DataFrame) and "Pore saturation time [days]" in df_all.columns:
                mean_sat_time_days = float(df_all["Pore saturation time [days]"].mean())
            else:
                mean_sat_time_days = float("nan")
            flow_m3_day = float(ft) / 1000.0 if ft and np.isfinite(ft) else float("nan")
            turnover_days_total = (
                total_pore_volume_m3 / flow_m3_day
                if (flow_m3_day and flow_m3_day > 0 and math.isfinite(total_pore_volume_m3))
                else float("nan")
            )
            std_total_report = std_total_mc

            daily_turnover_pct = 100.0 / turnover_days_total if math.isfinite(turnover_days_total) and turnover_days_total > 0 else float("nan")

            flow_stats[float(ft)] = {
                "total_tons_sat": total_tons_sat,
                "std_total": std_total_report,
                "std_total_mc": std_total_mc,
                "mean_efficiency": mean_efficiency,
                "limiting_factor": dominant_limiter,
                "n_iter": n_iter_effective,
                "flow_rate_L_day": float(ft),
                "total_flow_L_day": float(ft),
                "Conc [mol/kg]": raw_total_mol_per_kg,
                "pore_volume_turnover_days": turnover_days_total,
                "daily_turnover_pct": daily_turnover_pct,
                "pore_volume_mean_days_ranges": mean_turnover_days,
                "pore_saturation_time_days": mean_sat_time_days,
                "dt_day_used_mean": avg_dt_used if math.isfinite(avg_dt_used) else dt_for_flow,
            }
            summaries.append({
                "Flow target [L/day]": ft,
                "H2 total [tons]": total_tons_sat,
                "Std total [tons]": std_total_report,
                "Efficiency [%]": mean_efficiency,
                "Limiting factor": dominant_limiter,
                "Iterations": n_iter_effective,
                "Conc [mol/kg]": raw_total_mol_per_kg,
                "Sat time [days]": mean_sat_time_days,
                "Turnover [days]": turnover_days_total,
                "Turnover [%/day]": daily_turnover_pct,
                "dt [day]": avg_dt_used if math.isfinite(avg_dt_used) else dt_for_flow,
            })
            if pbar is not None:
                pbar.update(1)

    if not summaries:
        print("Could not generate Monte Carlo statistics for the requested flows.")
        return pd.DataFrame()

    df_results_limiting = pd.DataFrame(summaries).sort_values("Flow target [L/day]")
    df_results_limiting.attrs["flow_stats_per_target"] = flow_stats

    if flow_stats:
        print("\nSummary of H₂ total per flow target:")
        header_fmt = (
            "{flow:<13} {iters:>5} {total_flow:>15} {h2tot:>12} {std:>10} "
            "{conc:>12} {sat_time:>12} {turn:>12} {recambio:>15} {dt:>9} {limit:>7}"
        )
        header_line = header_fmt.format(
            flow="Flow", iters="Iter", total_flow="Total flow",
            h2tot="H₂ total", std="Std", conc="Conc",
            sat_time="SatTime", turn="Turnover days", recambio="Turnover", dt="dt", limit="Limit"
        )
        units_line = header_fmt.format(
            flow="[L/day]", iters="[-]", total_flow="[L/day]",
            h2tot="[tons]", std="[tons]", conc="[mol/kg]",
            sat_time="[days]", turn="[days]", recambio="[%/day]", dt="[day]", limit=""
        )
        dash_len = max(len(header_line), len(units_line))
        print("-" * dash_len)
        print(header_line)
        print(units_line)
        print("-" * dash_len)
        seen_non_water = False
        for ft in sorted(flow_stats.keys()):
            stats_ft = flow_stats[ft]
            iter_count_val = stats_ft.get("n_iter", mc_iterations)
            try:
                iter_count = int(iter_count_val)
            except Exception:
                iter_count = int(mc_iterations)
            total_tons_for_flow = stats_ft.get("total_tons_sat", float("nan"))
            sat_time_days = stats_ft.get("pore_saturation_time_days", float("nan"))
            turnover_days = stats_ft.get("pore_volume_turnover_days", float("nan"))
            if not seen_non_water and math.isfinite(total_tons_for_flow) and total_tons_for_flow < 10.0:
                limit_label = "water"
            elif math.isfinite(sat_time_days) and math.isfinite(turnover_days):
                limit_label = "sat" if sat_time_days < turnover_days else "rate"
                if limit_label in ("sat", "rate"):
                    seen_non_water = True
            else:
                limit_label = "-"
            stats_ft["limiting_factor"] = limit_label
            print(
                header_fmt.format(
                    flow=f"{ft:.2e}",
                    iters=str(iter_count),
                    total_flow=f"{stats_ft.get('total_flow_L_day', ft):.2f}",
                    h2tot=f"{stats_ft['total_tons_sat']:.2f}",
                    std=f"{stats_ft['std_total']:.2f}",
                    conc=f"{stats_ft.get('Conc [mol/kg]', stats_ft.get('avg_concentration_total_mol_kg', 0.0)):.4f}",
                    sat_time=f"{stats_ft.get('pore_saturation_time_days', float('nan')):.2f}",
                    turn=f"{stats_ft.get('pore_volume_turnover_days', float('nan')):.2f}",
                    recambio=f"{stats_ft.get('daily_turnover_pct', float('nan')):.2f}",
                    dt=f"{stats_ft.get('dt_day_used_mean', float('nan')):.4f}",
                    limit=limit_label,
                )
            )
        print("-" * dash_len)

        col_meanings = [
            "Flow: water-delivery target evaluated (log sweep).",
            "Iter: number of Monte Carlo iterations used for that target.",
            "Total flow: equivalent volumetric delivery in L/day.",
            "H₂ total: mean dissolved H₂ produced under saturation limits (tons).",
            "Std: standard deviation of H₂ total (tons).",
            "Conc: dissolved H₂ concentration per kg of delivered water (mol/kg).",
            "SatTime: days until the pore water reaches 99% average saturation.",
            "Turnover days: bulk pore-volume turnover time (days).",
            "Turnover [%/day]: percentage of pore water replaced per day.",
            "Limit: dominant limiting factor (water / sat / rate / -).",
            "dt [day]: Mean integration timestep used during the Monte Carlo runs for that flow target.",
        ]
        for desc in col_meanings:
            print(f"  - {desc}")

        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=(10, 9),
            sharex=True,
            gridspec_kw={"height_ratios": [2.5, 1.2]},
        )

        flows_plot = []
        mean_plot = []
        min_plot = []
        max_plot = []
        factor_plot = []
        turnover_plot = []
        sat_time_plot = []
        conc_plot = []
        for ft in sorted(flow_stats.keys()):
            stats_ft = flow_stats[ft]
            mean_val = stats_ft["total_tons_sat"]
            std_val = stats_ft["std_total"]
            flows_plot.append(ft)
            mean_plot.append(mean_val)
            min_plot.append(mean_val - std_val)
            max_plot.append(mean_val + std_val)
            factor_plot.append(stats_ft["limiting_factor"])
            turnover_plot.append(stats_ft.get("pore_volume_turnover_days", float("nan")))
            sat_time_plot.append(stats_ft.get("pore_saturation_time_days", float("nan")))
            conc_plot.append(stats_ft.get("Conc [mol/kg]", float("nan")))

        colors = {"water": "green", "rate": "blue", "sat": "red", "-": "gray"}

        for factor, color in colors.items():
            idxs = [i for i, f in enumerate(factor_plot) if f == factor]
            if not idxs:
                continue
            flows_f = np.array([flows_plot[i] for i in idxs], dtype=float)
            mean_f = np.array([mean_plot[i] for i in idxs], dtype=float)
            min_f = np.array([min_plot[i] for i in idxs], dtype=float)
            max_f = np.array([max_plot[i] for i in idxs], dtype=float)
            ax1.plot(flows_f, mean_f, color=color, marker="o", label=factor)
            ax1.fill_between(flows_f.tolist(), min_f.tolist(), max_f.tolist(), color=color, alpha=0.15)

        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_ylabel("H₂ Total [tons]")
        ax1.set_title("Monte Carlo H₂ totals vs Flow target")
        ax1.grid(True)
        ax1.legend(title="Limiting factor", loc="best")

        flows_arr = np.asarray(flows_plot, dtype=float)
        turnover_arr = np.asarray(turnover_plot, dtype=float)
        sat_time_arr = np.asarray(sat_time_plot, dtype=float)
        conc_arr = np.asarray(conc_plot, dtype=float)

        ratio_arr = turnover_arr / sat_time_arr
        mask_ratio = np.isfinite(ratio_arr) & (ratio_arr > 0)
        mask_conc = np.isfinite(conc_arr) & (conc_arr > 0)

        if mask_ratio.any():
            ax2.plot(
                flows_arr[mask_ratio],
                ratio_arr[mask_ratio],
                color="tab:purple",
                marker="o",
                linestyle="-",
                label="Turnover / SatTime",
            )

        ax2.set_ylabel("Turnover / SatTime [-]")
        ax2.set_xlabel("Flow Target [L/day]")
        ax2.grid(True, which="both", alpha=0.3)
        ax2_right = ax2.twinx()
        if mask_conc.any():
            ax2_right.plot(
                flows_arr[mask_conc],
                conc_arr[mask_conc],
                color="tab:orange",
                marker="s",
                linestyle="--",
                label="Conc [mol/kg]",
            )
        ax2_right.set_ylabel("Conc [mol/kg]")

        handles_left, labels_left = ax2.get_legend_handles_labels()
        handles_right, labels_right = ax2_right.get_legend_handles_labels()
        if handles_left or handles_right:
            ax2.legend(handles_left + handles_right, labels_left + labels_right, loc="best")

        fig.tight_layout()

        if params.results_path:
            os.makedirs(params.results_path, exist_ok=True)
            base_path = os.path.join(params.results_path, "limiting_factor_vs_flow_mc")
            plt.savefig(base_path + ".png", dpi=300)
            if get_plot_save_svg():
                plt.savefig(base_path + ".svg")

            csv_path = base_path + ".csv"
            df_results_limiting.to_csv(csv_path, index=False)
            column_meanings = [
                "Flow target [L/day]: Water delivery target evaluated (log-spaced sweep).",
                "H2 total [tons]: Mean total H₂ produced over the reporting period (includes solubility cap).",
                "Std total [tons]: Standard deviation of total H₂ across Monte Carlo iterations.",
                "Efficiency [%]: Percent of the no-saturation total achieved (uses outer run total when available).",
                "Limiting factor: Dominant limiting factor across ranges (water / sat / rate / -).",
                "Iterations: Monte Carlo iterations effectively used for this flow target.",
                "Conc [mol/kg]: Dissolved H₂ concentration per kg of delivered water (excluding free gas).",
                "Sat time [days]: Time required for pore water to reach 99% average saturation.",
                "dt: mean simulation timestep used for that target (day).",
                "Turnover [days]: Bulk pore-volume turnover time using total pore volume ÷ flow rate.",
            ]
            with open(csv_path, "a", newline="") as f_meaning:
                writer = csv.writer(f_meaning)
                writer.writerow([])
                writer.writerow(["# Column meanings:"])
                for desc in column_meanings:
                    writer.writerow([f"# {desc}"])

        plt.show()

    return df_results_limiting

