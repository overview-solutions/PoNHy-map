from __future__ import annotations

import ast
import math
import os
import sys
import warnings
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from tqdm import tqdm

_HEADER_PRINTED = False


def _normalize_unit_parameters(
    labels: Optional[Sequence[str]],
    dens_adj: Sequence[float],
    magsus: Sequence[float],
    dens_disp: Sequence[float],
    magsus_disp: Sequence[float],
    weights: Sequence[float],
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if dens_adj is None or magsus is None or dens_disp is None or magsus_disp is None or weights is None:
        raise ValueError("UNIT_*_LIST parameters must be defined to configure the GMM units.")

    lengths = {len(dens_adj), len(magsus), len(dens_disp), len(magsus_disp), len(weights)}
    if len(lengths) != 1:
        raise ValueError(
            "Unit parameter lists must have the same length. "
            "Check unit_dens_adj_list, unit_magsus_list, unit_dens_disp_list, "
            "unit_magsus_disp_list, vol_unit_list."
        )

    n_units = len(dens_adj)
    resolved_labels = list(labels) if labels else []
    if not resolved_labels or len(resolved_labels) != n_units:
        resolved_labels = [f"Unit {i + 1}" for i in range(n_units)]

    dens_adj_arr = np.asarray(dens_adj, dtype=float)
    magsus_arr = np.asarray(magsus, dtype=float)
    dens_disp_arr = np.asarray(dens_disp, dtype=float)
    magsus_disp_arr = np.asarray(magsus_disp, dtype=float)
    weights_arr = np.asarray(weights, dtype=float)

    weight_sum = float(np.sum(weights_arr)) if weights_arr.size else 0.0
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("vol_unit_list must contain positive values that sum to 1.")
    if not np.isclose(weight_sum, 1.0):
        warnings.warn("vol_unit_list does not sum to 1. Normalizing weights.")
        weights_arr = weights_arr / weight_sum

    return (
        resolved_labels,
        dens_adj_arr,
        magsus_arr,
        dens_disp_arr,
        magsus_disp_arr,
        weights_arr,
    )


def _clamp_worker_count(requested: Optional[int] = None) -> int:
    """Clamp the requested worker count to the available CPUs (>=1)."""
    cpu_total = os.cpu_count() or 1
    if requested is None:
        return cpu_total
    try:
        value = int(requested)
    except (TypeError, ValueError):
        value = cpu_total
    if value <= 0:
        return cpu_total
    return max(1, min(value, cpu_total))


def _compute_mc_chunk_size(n_iter: int, range_count: int, target_updates: int = 200) -> int:
    """Derive a sensible chunk size for Monte Carlo batching without user tuning."""
    if n_iter <= 0:
        return 1
    total_iter = max(1, int(n_iter))
    range_count = max(1, int(range_count))
    target_updates = max(1, int(target_updates))
    per_range_target = max(1.0, target_updates / float(range_count))
    chunk_estimate = int(math.ceil(total_iter / per_range_target))
    return max(1, min(total_iter, chunk_estimate))


def _get_progress_bar(*args, file=sys.stdout, **kwargs):
    """Return a tqdm progress bar that auto-disables when stdout is redirected."""
    if "disable" in kwargs:
        return tqdm(*args, file=file, **kwargs)

    target_file = file
    try:
        if file is sys.stdout and hasattr(sys.stdout, "terminal"):
            orig = getattr(sys.stdout, "terminal")
            if orig is not None:
                target_file = orig
    except Exception:
        target_file = file

    try:
        is_tty = bool(getattr(target_file, "isatty", lambda: False)())
    except Exception:
        is_tty = False

    if not is_tty:
        try:
            tty = open("/dev/tty", "w")
            target_file = tty
            is_tty = True
        except Exception:
            pass
    kwargs["disable"] = not is_tty
    return tqdm(*args, file=target_file, **kwargs)


def _get_convergence_status(opt_instance):
    stop_reason = None
    for attr in (
        "stop_reason",
        "stopReason",
        "stopping_criteria",
        "stop_criteria",
        "stop_criterion",
        "stopping_criterion",
    ):
        value = getattr(opt_instance, attr, None)
        if value:
            stop_reason = value
            break

    iter_count = getattr(opt_instance, "iter", None)
    if iter_count is None:
        iter_count = getattr(opt_instance, "iterNum", None)
    max_iter = getattr(opt_instance, "maxIter", None)

    reason_text = str(stop_reason) if stop_reason else ""
    reason_text_lower = reason_text.lower()
    hit_max_iter = False
    if "maxiter" in reason_text_lower or "max iter" in reason_text_lower:
        hit_max_iter = True
    if not hit_max_iter and iter_count is not None and max_iter is not None:
        hit_max_iter = iter_count >= max_iter

    if stop_reason:
        status = reason_text
    elif hit_max_iter:
        status = "Reached maxIter"
    else:
        status = "Unknown (optimizer did not report a stop reason)"

    converged = bool(stop_reason) and not hit_max_iter
    return status, converged, iter_count, max_iter


def _sample_unit_hypercube(n: int, d: int, mode: str = "uniform", seed: Optional[int] = None) -> np.ndarray:
    """Generate unit-cube samples with unified Sobol/LHS/Uniform."""
    if n <= 0 or d <= 0:
        return np.zeros((0, d), dtype=float)
    mode = (mode or "uniform").lower()

    if mode == "sobol":
        try:
            from scipy.stats import qmc

            m = int(math.ceil(math.log2(max(1, n))))
            sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
            U = sampler.random_base2(m=m)
            return U[:n, :]
        except Exception:
            pass

    if mode == "lhs":
        try:
            from scipy.stats import qmc

            engine = qmc.LatinHypercube(d=d, seed=seed)
            return engine.random(n)
        except Exception:
            pass

        rng = np.random.default_rng(seed)
        U = np.empty((n, d), dtype=float)
        for j in range(d):
            perm = rng.permutation(n)
            U[:, j] = (perm + rng.random(n)) / n
        return U

    rng = np.random.default_rng(seed)
    return rng.random((n, d))


def _scale_samples_to_ranges(unit: np.ndarray, range_specs: Sequence[Tuple[str, Tuple[float, float]]]) -> np.ndarray:
    """Scale unit-cube samples to provided (lo, hi) ranges."""
    if unit.size == 0:
        return unit

    out = np.empty_like(unit, dtype=float)
    for j, (_, (lo, hi)) in enumerate(range_specs):
        out[:, j] = lo + (hi - lo) * unit[:, j]
    return out


def _coerce_config_value(key: str, value: Any) -> Any:
    if key == "kappa":
        return np.array(value)
    if key == "chi0_ratio":
        return np.array(value)
    if key == "n_cores" and (value == 0 or value == "auto"):
        return None
    if key in {"serpentinization_data", "serp_corr_percentage"} and isinstance(value, dict):
        return {k: np.array(v) for k, v in value.items()}
    return value


def _apply_text_config(
    config: Dict[str, Any],
    config_param_names: Sequence[str],
    globals_dict: Dict[str, Any],
) -> None:
    unknown = [key for key in config.keys() if key not in config_param_names]
    if unknown:
        print(f"ERROR: Unknown config keys in text file: {', '.join(sorted(unknown))}")
        sys.exit(1)
    for key, value in config.items():
        globals_dict[key] = _coerce_config_value(key, value)
    globals_dict["DEFAULT_SAMPLING_SEED"] = (
        globals_dict["seed"] if globals_dict.get("use_global_seed") else None
    )


def _remove_inline_comment(value: str) -> str:
    in_single = False
    in_double = False
    for idx, ch in enumerate(value):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            return value[:idx].rstrip()
    return value.strip()


def _parse_value(raw_value: str) -> Any:
    value = _remove_inline_comment(raw_value).strip()
    if value == "":
        return ""
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return ast.literal_eval(value)
    except Exception:
        return value.strip("\"'")


def _load_text_config(path: str, top_level_keys: Optional[Sequence[str]] = None) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    data: Dict[str, Any] = {}
    current_section: Optional[str] = None
    top_level = set(top_level_keys or [])
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if stripped.startswith("[") and stripped.endswith("]"):
                    section = stripped[1:-1].strip()
                    if not section:
                        raise ValueError(f"Empty section header at line {line_no}.")
                    current_section = section
                    data.setdefault(section, {})
                    continue
                if "=" not in stripped:
                    raise ValueError(f"Invalid line {line_no}: {line.rstrip()}")
                key, raw_value = stripped.split("=", 1)
                key = key.strip()
                parsed_value = _parse_value(raw_value)
                if current_section and key in top_level:
                    target = data
                else:
                    target = data[current_section] if current_section else data
                if key in target:
                    raise ValueError(f"Duplicate key '{key}' at line {line_no}.")
                target[key] = parsed_value
    except Exception as exc:
        print(f"ERROR: Failed to read config file '{path}': {exc}")
        sys.exit(1)
    return data


def _load_yaml_config(path: str, top_level_keys: Optional[Sequence[str]] = None) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except Exception as exc:
        print(f"ERROR: Failed to read config file '{path}': {exc}")
        sys.exit(1)
    if data is None:
        return None
    if not isinstance(data, dict):
        print(f"ERROR: Config file '{path}' must define a top-level mapping.")
        sys.exit(1)
    return data


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) == 0
    return False


def _report_missing_params(missing: Sequence[str], routine: str) -> None:
    for name in missing:
        print(f"ERROR: Missing required parameter '{name}' for {routine}.")


def _print_header() -> None:
    print("\n" + "=" * 150)
    print("  PoNHy - Potential for Natural Hydrogen")
    print("  Developed by: Rodolfo Christiansen (LIAG- Institute for Applied Geophysics) and the PoNHy Team")
    print("  Contact:  Rodolfo.Christiansen@liag-institut.de - rodo_christiansen@hotmail.com")
    print("  MIT License: Free to use, modify, and redistribute with attribution")
    print("=" * 150)


def _print_header_once() -> None:
    global _HEADER_PRINTED
    if not _HEADER_PRINTED:
        _print_header()
        _HEADER_PRINTED = True


def _format_value_for_filename(val: float) -> str:
    """Return a compact, filesystem-friendly string for numeric values (keeps scientific 'e')."""
    try:
        s = f"{val:.6g}"
    except Exception:
        s = str(val)
    return s.replace("+", "")


def _resolve_base_dir_path(base_dir: str, path: Optional[str]) -> Optional[str]:
    """Resolve a path relative to base_dir, treating /Data/... as base_dir/Data/... when needed."""
    if path is None:
        return None
    if not isinstance(path, str):
        return str(path)
    if not base_dir:
        return path
    if os.path.isabs(path):
        candidate = os.path.join(base_dir, path.lstrip(os.sep))
        if os.path.exists(candidate) and not os.path.exists(path):
            return candidate
        if path.startswith(os.sep) and not os.path.exists(path):
            return candidate
        return path
    return os.path.join(base_dir, path)


def _moving_average(series: List[float], min_window: int = 10, max_window: int = 200) -> List[float]:
    """Return a moving-average smoothed copy of the series (no padding)."""
    if not series:
        return series
    arr = np.asarray(series, dtype=float)
    window = max(min_window, min(max_window, max(1, len(arr) // 40)))
    if len(arr) < window:
        return arr.tolist()
    kernel = np.ones(window, dtype=float) / window
    smoothed = np.convolve(arr, kernel, mode="same")
    return smoothed.tolist()


def _trim_trailing_dropoff(series: List[float], drop_ratio: float = 0.8, window: int = 30) -> List[float]:
    """Trim trailing segment if the last window falls below a ratio of the preceding window mean."""
    if not series or len(series) < 2 * window:
        return series
    arr = list(series)
    while len(arr) >= 2 * window:
        tail = arr[-window:]
        prev = arr[-2 * window : -window]
        tail_mean = float(np.mean(tail)) if np.isfinite(np.mean(tail)) else tail[-1]
        prev_mean = float(np.mean(prev)) if np.isfinite(np.mean(prev)) else prev[-1]
        if prev_mean > 0 and tail_mean < drop_ratio * prev_mean:
            arr = arr[:-window]
        else:
            break
    return arr


def _select_stable_window(
    series: List[float],
    target_len: int,
    warmup_days: int = 25,
    min_window: int = 10,
    max_window: int = 200,
) -> Dict[str, Any]:
    """Pick the most stable interval after warm-up; extend by scaling if not enough length."""
    if not series:
        return {
            "smoothed": [],
            "mean": 0.0,
            "start": 0,
            "end": -1,
            "len_used": 0,
            "extend_needed": True,
            "target_len": target_len,
        }

    trimmed = _trim_trailing_dropoff(series)
    smoothed = _moving_average(trimmed, min_window=min_window, max_window=max_window)
    n = len(smoothed)
    target_len = max(1, int(target_len))
    start_idx = min(warmup_days, n)
    if start_idx >= n:
        start_idx = max(0, n - 1)
    candidate = smoothed[start_idx:]
    if not candidate:
        return {
            "smoothed": smoothed,
            "mean": float(np.mean(smoothed)) if smoothed else 0.0,
            "start": 0,
            "end": n - 1,
            "len_used": n,
            "extend_needed": True,
            "target_len": target_len,
        }

    if len(candidate) >= target_len:
        win_len = target_len
        best_idx = 0
        best_metric = float("inf")
        for i in range(0, len(candidate) - win_len + 1):
            window_vals = candidate[i : i + win_len]
            mean_w = float(np.mean(window_vals)) if window_vals else 0.0
            std_w = float(np.std(window_vals)) if window_vals else 0.0
            metric = std_w / max(abs(mean_w), 1e-12)
            if metric < best_metric:
                best_metric = metric
                best_idx = i
        sel_start = start_idx + best_idx
        sel_end = sel_start + win_len - 1
        mean_sel = float(np.mean(smoothed[sel_start : sel_end + 1])) if win_len > 0 else 0.0
        return {
            "smoothed": smoothed,
            "mean": mean_sel,
            "start": sel_start,
            "end": sel_end,
            "len_used": win_len,
            "extend_needed": False,
            "target_len": target_len,
        }

    win_len = len(candidate)
    mean_sel = float(np.mean(candidate)) if candidate else 0.0
    sel_start = start_idx
    sel_end = n - 1
    return {
        "smoothed": smoothed,
        "mean": mean_sel,
        "start": sel_start,
        "end": sel_end,
        "len_used": win_len,
        "extend_needed": True,
        "target_len": target_len,
    }


def _as_scalar_value(val: Any) -> float:
    """Coerce lists/tuples/strings to a numeric scalar."""
    while isinstance(val, (list, tuple)):
        val = val[0] if val else 0.0
    if isinstance(val, str):
        try:
            val = float(val.strip())
        except ValueError:
            return 0.0
    try:
        return float(val)
    except Exception:
        return 0.0
