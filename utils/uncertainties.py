from __future__ import annotations

import math
import os
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager, nullcontext, redirect_stdout
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils.helpers import _clamp_worker_count, _get_progress_bar, _sample_unit_hypercube
from utils.no_saturation import (
    _clear_no_saturation_worker_context,
    _init_no_saturation_worker_context,
    _run_no_saturation_sample,
    compute_h2_production_no_saturation,
)
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
        bar_format="{l_bar}{bar}| [{elapsed}<{remaining}, {rate_fmt}]",
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
    n_rep = max(1, int(uni_config.get("n_rep", 1)))
    baseline_n_iter = max(1, int(uni_config.get("baseline_n_iter", n_rep)))
    seed_value = seed if seed is not None else uni_config.get("sampling_seed", base_config.get("sampling_seed"))
    quiet = bool(uni_config.get("quiet", False))
    show_progress = bool(uni_config.get("show_progress", False) and not quiet)
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

    def _execute_tasks(task_list, desc):
        if not task_list:
            return []
        progress = (
            _get_progress_bar(total=len(task_list), desc=desc, file=sys.stdout, leave=False)
            if show_progress
            else None
        )
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
            progress.close()
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
    print("UNIVARIATE (SATURATION)".center(150))
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
    # Interpretation note intentionally omitted to keep output minimal.

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
        x_vals = df_results_limiting["Flow target [L/day]"].to_numpy(dtype=float)
        y_vals = df_results_limiting["H2 total [tons]"].to_numpy(dtype=float)
        ax.plot(x_vals, y_vals, marker="o")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Flow Target [L/day]")
        ax.set_ylabel("H2 Total [tons]")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        _save_plot_pair(os.path.join(results_path, "limiting_factor_vs_flow"), fig, dpi=300)
        plt.close(fig)

    return df_results_limiting

