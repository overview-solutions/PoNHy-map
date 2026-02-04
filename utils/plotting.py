from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Sequence, cast

import matplotlib
from matplotlib import cm, pyplot as plt
from matplotlib.colors import Colormap
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd

from utils.general import extract_column
from utils.helpers import _format_value_for_filename, _moving_average, _select_stable_window

_SAVE_SVG: bool = True


def set_plot_save_svg(enabled: bool) -> None:
    """Enable or disable saving SVG files for plot outputs."""
    global _SAVE_SVG
    _SAVE_SVG = bool(enabled)


def _save_plot_pair(
    path: str,
    fig=None,
    dpi: int = 300,
    tight: bool = True,
    save_svg: Optional[bool] = None,
):
    """Save the provided figure (or current figure) as PNG (and optional SVG) files."""
    fig = fig or plt.gcf()
    base, _ = os.path.splitext(path)
    if not base:
        base = path

    png_kwargs: dict[str, Any] = {"dpi": dpi}
    if tight:
        png_kwargs["bbox_inches"] = "tight"

    svg_kwargs: dict[str, Any] = {}
    if tight:
        svg_kwargs["bbox_inches"] = "tight"

    fig.savefig(f"{base}.png", **png_kwargs)

    if save_svg is None:
        save_svg = _SAVE_SVG

    svg_path = None
    if save_svg:
        svg_path = f"{base}.svg"
        fig.savefig(svg_path, **svg_kwargs)
    return f"{base}.png", svg_path


_cm_registry = getattr(matplotlib, "colormaps", None)
if _cm_registry is None:
    _cm_registry = cm


def _get_cmap(name: str):
    if _cm_registry is not None and hasattr(_cm_registry, "get_cmap"):
        return _cm_registry.get_cmap(name)
    return cm.get_cmap(name)


def _get_discrete_colors(name: str, n: int) -> List[Any]:
    n = max(1, int(n))
    cmap = _get_cmap(name)

    resample_fn = getattr(cmap, "resampled", None)
    if callable(resample_fn):
        try:
            cmap = resample_fn(n)
        except Exception:
            pass

    if not callable(cmap) or getattr(cmap, "N", None) != n:
        try:
            cmap = cm.get_cmap(name, n)
        except Exception:
            pass

    cmap_callable = cast(Colormap, cmap)
    colors = [cmap_callable(i / max(1, n - 1)) for i in range(n)]
    return colors


def plot_misfit_evolution(save_values_directive, *, results_path: str):
    """Plot the evolution of gravity and magnetic misfits over inversion iterations."""
    if save_values_directive is None:
        return None
    phi_d_list = getattr(save_values_directive, "phi_d_list", [])
    if not phi_d_list:
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    iterations = range(len(phi_d_list))

    ax1.plot(iterations, getattr(save_values_directive, "phi_d_gravity_list", []), "b.-", label="Gravity Data Misfit")
    ax1.set_ylabel("Gravity Misfit")
    ax1.grid(True)
    ax1.legend(loc="upper right")

    ax2.plot(iterations, getattr(save_values_directive, "phi_d_magnetic_list", []), "g.-", label="Magnetic Data Misfit")
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Magnetic Misfit")
    ax2.grid(True)
    ax2.legend(loc="upper right")

    fig.suptitle("Evolution of Misfits over Iterations")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    file_base = os.path.join(results_path, "misfit_evolution")
    _save_plot_pair(file_base, fig, tight=False)
    plt.close(fig)
    return fig


def plot_learned_gmm(
    learned_gmm,
    density_model_no_info,
    magsus_model_no_info,
    ind_active,
    *,
    results_path: str,
    true_means: Optional[np.ndarray] = None,
    model_colors=None,
):
    """Plot the learned Gaussian Mixture Model (GMM) distributions for density and susceptibility."""
    if learned_gmm is None:
        return None

    if model_colors is not None:
        units_in_model = np.unique(model_colors[ind_active])
    else:
        units_in_model = np.array([])

    if true_means is not None:
        units_in_true_means = np.arange(len(true_means))
        units = np.union1d(units_in_model, units_in_true_means)
    else:
        units = units_in_model

    num_units = len(units)
    color_list = _get_discrete_colors("tab10", num_units if num_units > 0 else 1)
    color_mapping = dict(zip(units, color_list))
    color_names = {unit: f"Geounit {unit}" for unit in units}

    fig = plt.figure(figsize=(19, 10))
    ax0 = plt.subplot2grid((4, 4), (3, 1), colspan=3)
    ax1 = plt.subplot2grid((4, 4), (0, 1), colspan=3, rowspan=3)
    ax2 = plt.subplot2grid((4, 4), (0, 0), rowspan=3)
    ax = [ax0, ax1, ax2]
    learned_gmm.plot_pdf(flag2d=True, ax=ax, padding=2.5, plotting_precision=100)
    ax[0].set_xlabel("Density contrast [g/cc]")
    ax[2].set_ylabel("Magnetic Susceptibility [SI]")

    if model_colors is not None:
        scatter_colors = [color_mapping[val] for val in model_colors[ind_active]]
    else:
        scatter_colors = "blue"

    ax[1].scatter(
        density_model_no_info[ind_active],
        magsus_model_no_info[ind_active],
        c=scatter_colors,
        edgecolors="k",
        linewidths=0.3,
        marker=MarkerStyle("o"),
        label="Recovered PGI model",
        alpha=0.5,
    )

    if true_means is not None:
        for idx, (density, susceptibility) in enumerate(true_means):
            unit = idx
            ax[1].scatter(
                density,
                susceptibility,
                label=f"{color_names[unit]}",
                c=[color_mapping[unit]],
                marker=MarkerStyle("v"),
                edgecolors="k",
                s=200,
            )

    ax[0].hist(density_model_no_info[ind_active], density=True, bins=50)
    ax[2].hist(magsus_model_no_info[ind_active], density=True, bins=50, orientation="horizontal")
    ax[1].legend(loc="lower right")

    file_base = os.path.join(results_path, "learned_GMM")
    _save_plot_pair(file_base, fig, dpi=300)
    plt.close(fig)
    return fig


def plot_observed_predicted_residual(
    receiver_locations,
    observed_data,
    predicted_data,
    residual_data,
    titles,
    cbar_label,
    file_name,
):
    """Plot observed, predicted, and residual data grids side by side and save the figure."""
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    for ax, data_f, title in zip(axs, [observed_data, predicted_data, residual_data], titles):
        cont_f = ax.tricontourf(receiver_locations[:, 0], receiver_locations[:, 1], data_f, 100, cmap="RdBu_r")
        ax.tricontour(
            receiver_locations[:, 0],
            receiver_locations[:, 1],
            data_f,
            levels=25,
            colors="black",
            linewidths=0.3,
        )
        fig.colorbar(cont_f, ax=ax, label=cbar_label)
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal")
    plt.tight_layout()
    base_path = os.path.splitext(file_name)[0]
    _save_plot_pair(base_path, fig)
    plt.close(fig)


def plot_gravity_observed_predicted_residual(
    receiver_locations,
    observed_grav,
    d_pred_grav,
    residual_grav,
    *,
    results_path,
):
    """Plot observed, predicted, and residual gravity data using standardized layouts."""
    titles = ["Observed Gravity Data", "Predicted Gravity Data", "Gravity Residual"]
    file_name = os.path.join(results_path, "gravity_data.png")
    plot_observed_predicted_residual(
        receiver_locations,
        observed_grav,
        d_pred_grav,
        residual_grav,
        titles,
        "mGal",
        file_name,
    )


def plot_magnetic_observed_predicted_residual(
    receiver_locations,
    observed_mag,
    d_pred_mag,
    residual_mag,
    *,
    results_path,
):
    """Plot observed, predicted, and residual magnetic data using standardized layouts."""
    titles = ["Observed Magnetic Data", "Predicted Magnetic Data", "Magnetic Residual"]
    file_name = os.path.join(results_path, "magnetic_data.png")
    plot_observed_predicted_residual(
        receiver_locations,
        observed_mag,
        d_pred_mag,
        residual_mag,
        titles,
        "nT",
        file_name,
    )


def plot_residual_histograms(residual_grav, residual_mag, *, results_path):
    """Plot histograms comparing gravity and magnetic residual distributions side by side."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    mean_grav = np.mean(residual_grav)
    std_grav = np.std(residual_grav)
    axs[0].hist(residual_grav, bins="auto", color="blue", alpha=0.75, density=True)
    axs[0].set_title("Gravity Residuals")
    axs[0].set_xlabel("Residual mGal")
    axs[0].set_ylabel("Frequency")
    axs[0].annotate(
        f"Mean: {mean_grav:.2f}\nSTD: {std_grav:.2f}",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        verticalalignment="top",
        horizontalalignment="left",
    )

    mean_mag = np.mean(residual_mag)
    std_mag = np.std(residual_mag)
    axs[1].hist(residual_mag, bins="auto", color="red", alpha=0.75, density=True)
    axs[1].set_title("Magnetic Residuals")
    axs[1].set_xlabel("Residual nT")
    axs[1].set_ylabel("Frequency")
    axs[1].annotate(
        f"Mean: {mean_mag:.2f}\nSTD: {std_mag:.2f}",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        verticalalignment="top",
        horizontalalignment="left",
    )
    plt.tight_layout()
    file_name = os.path.join(results_path, "histograms_side_by_side.png")
    base_path = os.path.splitext(file_name)[0]
    _save_plot_pair(base_path, fig)
    plt.close(fig)


def save_total_timeseries_plots(
    timeseries_by_range: Dict[str, Dict[str, List[float]]],
    flow_target: float,
    results_path: str,
    flow_order: int = 1,
) -> None:
    """Save the total daily H₂ production (sum of all temperature ranges) as plot + CSV."""
    if not timeseries_by_range:
        return

    flow_label = _format_value_for_filename(flow_target)
    out_dir = os.path.join(results_path, "timeseries_flows")
    os.makedirs(out_dir, exist_ok=True)

    total_prod: List[float] = []
    total_prod_full: List[float] = []
    for series in timeseries_by_range.values():
        if not isinstance(series, dict) or "H2_prod_mol_day" not in series:
            continue
        y = series.get("H2_prod_mol_day", [])
        y_full = series.get("H2_prod_mol_day_full", y)
        if y:
            if len(total_prod) < len(y):
                total_prod.extend([0.0] * (len(y) - len(total_prod)))
            for i, val in enumerate(y):
                total_prod[i] += float(val) if val is not None else 0.0
        if y_full:
            if len(total_prod_full) < len(y_full):
                total_prod_full.extend([0.0] * (len(y_full) - len(total_prod_full)))
            for i, val in enumerate(y_full):
                total_prod_full[i] += float(val) if val is not None else 0.0

    if total_prod:
        warmup_days = 25
        meta = None
        for series in timeseries_by_range.values():
            if isinstance(series, dict) and "__meta" in series:
                meta = series["__meta"]
                break

        cooldown_default = 25
        cooldown_days = int(meta.get("cooldown_days", cooldown_default)) if isinstance(meta, dict) else cooldown_default
        series_for_window = total_prod[:-cooldown_days] if (cooldown_days > 0 and len(total_prod) > cooldown_days) else total_prod

        series_for_plot = total_prod_full if total_prod_full else total_prod

        stable = _select_stable_window(series_for_window, target_len=len(series_for_window), warmup_days=warmup_days)
        smoothed_window = np.asarray(stable.get("smoothed", series_for_window), dtype=float)
        smoothed_full = np.asarray(_moving_average(series_for_plot), dtype=float)
        offset = warmup_days
        days_full = list(range(len(smoothed_full)))
        consider_start = offset + stable.get("start", 0)
        consider_end = offset + stable.get("end", len(smoothed_window) - 1)
        cooldown_start = offset + len(series_for_window)
        cooldown_end = offset + len(total_prod) - 1 if len(total_prod) > 0 else cooldown_start

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(days_full, smoothed_full, color="purple", label="H₂ total (moving average)")
        if offset > 0:
            ax.axvspan(0, offset, color="lightgray", alpha=0.15, label="Warm-up")
        if consider_start < consider_end:
            ax.axvspan(consider_start, consider_end, color="orange", alpha=0.28, label="Stable window")
        if cooldown_days > 0 and cooldown_end > cooldown_start:
            ax.axvspan(cooldown_start, cooldown_end, color="lightblue", alpha=0.15, label="Cooldown")
        ax.set_xlabel("Days")
        ax.set_ylabel("H₂ produced [mol/day]")
        ax.set_title(f"Flow {flow_target:.3g} L/day - Total")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        flow_value = float(flow_target)
        flow_int = int(round(flow_value)) if math.isfinite(flow_value) else flow_label
        fname_tot = f"flow_total_{flow_order}_{flow_int}.png"
        fpath_tot = os.path.join(out_dir, fname_tot)
        try:
            fig.savefig(fpath_tot, dpi=300)
        finally:
            plt.close(fig)


def prepare_lab_plot_inputs(
    production_rate_volumetric,
    serpentinization_front_velocities,
    serp_corr_percentage,
    mean_pressure_ranges,
    mean_depths,
    saturation_df,
    h2o_incorporated=None,
    inflow_diff=None,
    inflow_frac=None,
    volume_at_temperature=None,
):
    """Collect and normalize lab/field data for plotting."""

    def temp_midpoint(tr):
        a, b = tr.split("_")
        return (float(a) + float(b)) / 2

    def temp_lower(tr):
        return float(tr.split("_")[0])

    temp_ranges = sorted(mean_pressure_ranges.keys(), key=temp_lower)
    temp_midpoints = [temp_midpoint(k) for k in temp_ranges]
    pressures = [mean_pressure_ranges[k] / 10.0 for k in temp_ranges]

    if pressures:
        pmin = min(pressures)
        pmax = max(pressures)
        if pmin == pmax:
            pmin -= 0.1
            pmax += 0.1
    else:
        pmin = 0.0
        pmax = 1.0

    volume_map = volume_at_temperature or {}
    volume_enabled = bool(volume_map)
    volume_sequence = [float(volume_map.get(tr, 0.0)) for tr in temp_ranges] if volume_enabled else []

    if len(temp_midpoints) > 1:
        diffs = []
        s = sorted(set(temp_midpoints))
        for i in range(len(s) - 1):
            d = s[i + 1] - s[i]
            if d > 0:
                diffs.append(d)
        bar_width_temp = (min(diffs) * 0.6) if diffs else 0.5
    else:
        bar_width_temp = 0.5

    pressure_unique = sorted(set(pressures))
    pressure_position_lookup = {}
    pressure_tick_positions = []
    pressure_tick_labels = []

    if pressure_unique:
        if len(pressure_unique) == 1:
            pressure_position_lookup = {pressure_unique[0]: 0.5}
            p_axis_min = 0.0
            p_axis_max = 1.0
        else:
            positions = [i / (len(pressure_unique) - 1) for i in range(len(pressure_unique))]
            pressure_position_lookup = dict(zip(pressure_unique, positions))
            p_axis_min = 0.0
            p_axis_max = 1.0

        max_ticks = 6
        if len(pressure_unique) <= max_ticks:
            tick_values = pressure_unique
        else:
            step = max(1, len(pressure_unique) // (max_ticks - 1))
            tick_values = [pressure_unique[i] for i in range(0, len(pressure_unique), step)]
            if tick_values[-1] != pressure_unique[-1]:
                tick_values.append(pressure_unique[-1])

        for val in tick_values:
            pressure_tick_positions.append(pressure_position_lookup[val])
            pressure_tick_labels.append(f"{val:.1f}")

        bar_width_pressure = (1.0 / len(pressure_unique)) * 0.6
    else:
        p_axis_min = 0.0
        p_axis_max = 1.0
        bar_width_pressure = 0.5

    depth_sorted_keys = sorted(mean_pressure_ranges.keys(), key=lambda k: float(k.split("_")[0]))
    depths = [mean_depths[k] for k in depth_sorted_keys]
    temp_midpoints_sorted = [temp_midpoint(k) for k in depth_sorted_keys]

    wr_key = list(production_rate_volumetric.keys())[0]
    prod_data = sorted([(temp_midpoint(tr), val) for tr, val in production_rate_volumetric[wr_key]])

    h2o_clean = []
    if h2o_incorporated:
        items = []
        if isinstance(h2o_incorporated, dict):
            for k, v in h2o_incorporated.items():
                items.append((k, v))
        elif isinstance(h2o_incorporated, (list, tuple)):
            for item in h2o_incorporated:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    items.append(item)
                elif isinstance(item, dict):
                    for k, v in item.items():
                        items.append((k, v))
        for x, y in items:
            try:
                h2o_clean.append((float(x), float(y)))
            except Exception:
                pass

    corr_percentages = serp_corr_percentage["percentage"]
    corr_values = serp_corr_percentage["correction"]

    vel_data = []
    for tr, values in serpentinization_front_velocities.items():
        t_mid = temp_midpoint(tr)
        avg = None
        if isinstance(values, dict) and "avg" in values:
            avg = float(values["avg"])
        elif isinstance(values, (list, tuple)):
            for item in values:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    label, val = item
                    if isinstance(label, str) and label.lower() == "avg":
                        avg = float(val)
                    elif isinstance(label, (int, float)):
                        avg = float(val)
        if avg is not None:
            vel_data.append((t_mid, avg))
    vel_data.sort()

    solubility_lookup = {}
    if saturation_df is not None and not saturation_df.empty:
        if "Temperature Range" in saturation_df.columns and "Solubility [mol/kg]" in saturation_df.columns:
            for tr, sol in zip(saturation_df["Temperature Range"], saturation_df["Solubility [mol/kg]"]):
                solubility_lookup[tr] = sol

    solubility_series = [(mean_pressure_ranges.get(tr, 0.0) / 10.0, solubility_lookup.get(tr)) for tr in temp_ranges]

    return dict(
        temp_midpoints=temp_midpoints,
        temp_midpoints_sorted=temp_midpoints_sorted,
        depths=depths,
        prod_data=prod_data,
        h2o_clean=h2o_clean,
        corr_percentages=corr_percentages,
        corr_values=corr_values,
        vel_data=vel_data,
        solubility_series=solubility_series,
        pressures=pressures,
        pmin=pmin,
        pmax=pmax,
        p_axis_min=p_axis_min,
        p_axis_max=p_axis_max,
        pressure_position_lookup=pressure_position_lookup,
        pressure_tick_positions=pressure_tick_positions,
        pressure_tick_labels=pressure_tick_labels,
        volume_enabled=volume_enabled,
        volume_sequence=volume_sequence,
        bar_width_temp=bar_width_temp,
        bar_width_pressure=bar_width_pressure,
        inflow_diff=inflow_diff,
        inflow_frac=inflow_frac,
    )


def plot_lab_data_panels(p, *, results_path: Optional[str] = None):
    """Plot all eight lab-data panels using precomputed values."""

    def add_volume_bars(ax_ref, x_values, volumes, width):
        if not p["volume_enabled"]:
            return
        if not x_values or not volumes:
            return
        if len(x_values) != len(volumes):
            return
        ax_vol = ax_ref.twinx()
        ax_vol.bar(x_values, volumes, width=width, color="gray", alpha=0.3, edgecolor="none")
        ax_vol.set_ylabel("Volume [km³]", color="gray")
        ax_vol.tick_params(axis="y", labelcolor="gray")
        ax_vol.set_ylim(bottom=0)
        ax_vol.get_xaxis().set_visible(False)

    fig, axes = plt.subplots(4, 2, figsize=(10, 10))
    ax = axes.flatten()

    ax_a, ax_b, ax_c, ax_e, ax_g, ax_h, ax_d, ax_f = ax

    panel_labels = ["(a)", "(b)", "(c)", "(e)", "(g)", "(h)", "(d)", "(f)"]
    font_scale = 0.75
    marker_size = int(4 * font_scale)
    temp_ticks = sorted(set(p["temp_midpoints"])) if p["temp_midpoints"] else []
    temp_tick_positions = []
    if temp_ticks:
        start_tick = 50 * math.floor(temp_ticks[0] / 50.0)
        end_tick = 50 * math.ceil(temp_ticks[-1] / 50.0)
        temp_tick_positions = list(np.arange(start_tick, end_tick + 1, 50))

    for a, label in zip([ax_a, ax_b, ax_c, ax_e, ax_g, ax_h, ax_d, ax_f], panel_labels):
        a.text(0.02, 0.95, label, transform=a.transAxes, fontsize=int(10 * font_scale))
        a.grid(True, alpha=0.3)
        a.tick_params(axis="both", labelsize=int(10 * font_scale))

    ax_a.plot(p["temp_midpoints_sorted"], p["depths"], marker="o", markersize=marker_size, color="black")
    ax_a.invert_yaxis()
    ax_a.set_title("Temperature vs Depth", fontsize=int(12 * font_scale))
    ax_a.set_xlabel("Temperature [°C]", fontsize=int(11 * font_scale))
    ax_a.set_ylabel("Depth [m]", fontsize=int(11 * font_scale))

    add_volume_bars(ax_a, p["temp_midpoints_sorted"], p["volume_sequence"], p["bar_width_temp"])

    prod_x = [t for t, v in p["prod_data"]]
    prod_y = [v for t, v in p["prod_data"]]

    ax_b.plot(prod_x, prod_y, marker="o", markersize=marker_size, color="green")
    ax_b.set_title("Volumetric H₂ Production: Lherzolite", fontsize=int(12 * font_scale))
    ax_b.set_xlabel("Temperature [°C]", fontsize=int(11 * font_scale))
    ax_b.set_ylabel("Production Rate [mol/kg rock]", fontsize=int(11 * font_scale))

    add_volume_bars(ax_b, prod_x, p["volume_sequence"], p["bar_width_temp"])

    if p["h2o_clean"]:
        xs = [xx for xx, yy in p["h2o_clean"]]
        ys = [yy for xx, yy in p["h2o_clean"]]
        ax_c.plot(xs, ys, marker="o", markersize=marker_size, color="blue")
        ax_c.set_yscale("log")
        ax_c.relim()
        ax_c.autoscale()

    ax_c.set_title("H₂O Incorporated", fontsize=int(12 * font_scale))
    ax_c.set_xlabel("Temperature [°C]", fontsize=int(11 * font_scale))
    ax_c.set_ylabel("H₂O [kg/day]", fontsize=int(11 * font_scale))
    if temp_ticks:
        ax_c.set_xlim(temp_ticks[0], temp_ticks[-1])
        if temp_tick_positions:
            ax_c.set_xticks(temp_tick_positions)
            ax_c.set_xticklabels([int(t) for t in temp_tick_positions])

    temp_positions_full = p["temp_midpoints"]

    add_volume_bars(ax_c, temp_positions_full, p["volume_sequence"], p["bar_width_temp"])
    ax_d.plot(p["corr_percentages"], p["corr_values"], marker="o", markersize=marker_size, color="green")
    ax_d.set_title("Correction vs Serpentinization", fontsize=int(12 * font_scale))
    ax_d.set_xlabel("Serpentinization [%]", fontsize=int(11 * font_scale))
    ax_d.set_ylabel("Correction [%]", fontsize=int(11 * font_scale))

    vx = [x for x, y in p["vel_data"]]
    vy = [y for x, y in p["vel_data"]]

    ax_e.plot(vx, vy, marker="o", markersize=marker_size, color="purple")
    ax_e.set_yscale("log")
    ax_e.text(
        0.95,
        0.98,
        r"$\times 10^{5}$",
        transform=ax_e.transAxes,
        ha="right",
        va="top",
        fontsize=int(11 * font_scale),
    )
    ax_e.set_title("Serpentinization Front Velocities", fontsize=int(12 * font_scale))
    ax_e.set_xlabel("Temperature [°C]", fontsize=int(11 * font_scale))
    ax_e.set_ylabel("Velocity [cm/day]", fontsize=int(11 * font_scale))

    add_volume_bars(ax_e, vx, p["volume_sequence"], p["bar_width_temp"])

    sol_x = [p["pressure_position_lookup"].get(x, x) for x, y in p["solubility_series"]]
    sol_y = [y for x, y in p["solubility_series"]]

    ax_f.plot(sol_x, sol_y, marker="o", markersize=marker_size, color="red")
    ax_f.set_title("H₂ Solubility vs Pressure", fontsize=int(12 * font_scale))
    ax_f.set_xlabel("Pressure [MPa]", fontsize=int(11 * font_scale))
    ax_f.set_ylabel("Solubility [mol/kg H₂O]", fontsize=int(11 * font_scale))
    ax_f.set_xlim(p["p_axis_min"], p["p_axis_max"])
    ax_f.set_xticks(p["pressure_tick_positions"])
    ax_f.set_xticklabels(p["pressure_tick_labels"], rotation=45)

    if p["inflow_diff"]:
        xs = [xx for xx, yy in p["inflow_diff"]]
        ys = [yy for xx, yy in p["inflow_diff"]]
        ax_g.plot(xs, ys, marker="o", color="magenta", markersize=marker_size)
        ax_g.set_yscale("log")
        ax_g.relim()
        ax_g.autoscale()

    ax_g.set_title("Water inflow by Diffusion", fontsize=int(12 * font_scale))
    ax_g.set_xlabel("Temperature [°C]", fontsize=int(11 * font_scale))
    ax_g.set_ylabel("H₂O inflow [kg/day]", fontsize=int(11 * font_scale))
    if temp_ticks:
        ax_g.set_xlim(temp_ticks[0], temp_ticks[-1])
        if temp_tick_positions:
            ax_g.set_xticks(temp_tick_positions)
            ax_g.set_xticklabels([int(t) for t in temp_tick_positions])

    add_volume_bars(ax_g, temp_positions_full, p["volume_sequence"], p["bar_width_temp"])

    if p["inflow_frac"]:
        xs = [xx for xx, yy in p["inflow_frac"]]
        ys = [yy for xx, yy in p["inflow_frac"]]
        ax_h.plot(xs, ys, marker="o", color="brown", markersize=marker_size)
        ax_h.set_yscale("log")
        ax_h.relim()
        ax_h.autoscale()

    ax_h.set_title("Water inflow by Fractures", fontsize=int(12 * font_scale))
    ax_h.set_xlabel("Temperature [°C]", fontsize=int(11 * font_scale))
    ax_h.set_ylabel("H₂O inflow [kg/day]", fontsize=int(11 * font_scale))
    if temp_ticks:
        ax_h.set_xlim(temp_ticks[0], temp_ticks[-1])
        if temp_tick_positions:
            ax_h.set_xticks(temp_tick_positions)
            ax_h.set_xticklabels([int(t) for t in temp_tick_positions])

    add_volume_bars(ax_h, temp_positions_full, p["volume_sequence"], p["bar_width_temp"])
    plt.tight_layout()
    if results_path:
        plt.savefig(os.path.join(results_path, "lab_data.png"))
        plt.savefig(os.path.join(results_path, "lab_data.svg"))
    plt.show()
    plt.close()


def plot_h2_production_summary(
    df_out,
    volume_at_temperature,
    years=1,
    results_path=None,
    *,
    stats: Optional[pd.DataFrame] = None,
    production_rate_volumetric=None,
):
    """Plot H₂ production (top) and serpentinite volume (bottom) across temperature ranges."""
    temp_bins = sorted(df_out["Temperature Range"].unique(), key=lambda tr: int(tr.split("_")[0]))
    mid_temps = [int(tr.split("_")[0]) + (int(tr.split("_")[1]) - int(tr.split("_")[0])) / 2 for tr in temp_bins]
    if len(mid_temps) > 1:
        width = float(mid_temps[1] - mid_temps[0]) / 4
    else:
        width = 5.0

    cmap = cm.get_cmap("coolwarm")
    norm = matplotlib.colors.Normalize(min(mid_temps), max(mid_temps))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    values = []
    std_values: List[float] = []
    for tr in temp_bins:
        tr_row = df_out[df_out["Temperature Range"] == tr]
        prod_val = tr_row["H2 total [tons]"].iloc[0] if len(tr_row) > 0 else 0.0
        values.append(prod_val)

        if "H2 std [tons]" in tr_row.columns:
            std_val = tr_row["H2 std [tons]"].iloc[0] if len(tr_row) > 0 else 0.0
        else:
            std_val = None
        std_values.append(std_val if std_val is not None else 0.0)
    bar_values = [float(v) for v in values]

    if all(std == 0.0 for std in std_values):
        std_values = [0.0 for _ in temp_bins]
        if stats is not None and not stats.empty:
            try:
                if isinstance(stats.columns, pd.MultiIndex) and ("H2 total [tons]", "std") in stats.columns:
                    std_values = [float(stats.loc[tr][("H2 total [tons]", "std")]) if tr in stats.index else 0.0 for tr in temp_bins]
                else:
                    df_all = stats.attrs.get("df_all") if hasattr(stats, "attrs") else None
                    if df_all is not None and not df_all.empty and "H2 total [tons]" in df_all.columns:
                        std_map = df_all.groupby("Temperature Range")["H2 total [tons]"].std(ddof=0).to_dict()
                        std_values = [float(std_map.get(tr, 0.0)) for tr in temp_bins]
            except Exception:
                std_values = [0.0 for _ in temp_bins]

    ax1.errorbar(
        mid_temps,
        bar_values,
        yerr=std_values,
        fmt="o",
        color="royalblue",
        ecolor="black",
        elinewidth=1.2,
        capsize=5,
        markersize=8,
        label="H₂ Production",
    )
    for x, v, err in zip(mid_temps, bar_values, std_values):
        if v > 0:
            offset = (max(bar_values) * 0.01) if bar_values else 0.0
            ax1.text(x, v + offset + err, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    ax1.set_ylabel("H₂ Production (tons/yr)")
    ax1.set_title(f"H₂ Production over {years} Year(s) with Saturation")
    ax1.grid(True, axis="y", linestyle=":", linewidth=0.5)
    point_handle = Line2D([0], [0], color="royalblue", marker="o", linestyle="None", markersize=8, label="H₂ Production")
    handles = [point_handle]
    try:
        if any((sv is not None and float(sv) > 0.0) for sv in std_values):
            std_handle = Line2D([0], [0], color="black", marker="_", linestyle="None", markersize=10, label="H₂ std")
            handles.append(std_handle)
    except Exception:
        pass

    if production_rate_volumetric:
        used_wr_key = list(production_rate_volumetric.keys())[0]
        prod_data = [
            (
                int(tr.split("_")[0]) + (int(tr.split("_")[1]) - int(tr.split("_")[0])) / 2,
                rate,
            )
            for tr, rate in production_rate_volumetric[used_wr_key]
        ]
        prod_data.sort()
        prod_x = [x for x, _ in prod_data]
        prod_y = [y for _, y in prod_data]

        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("axes", 1.0))

        ax3.plot(prod_x, prod_y, "-", color="gray", alpha=0.4, linewidth=1.2, label="Volumetric H₂")
        ax3.plot(prod_x, prod_y, "o", color="gray", alpha=0.7, markersize=4, label="_nolegend_")
        ax3.set_ylabel("Volumetric H₂ (mol/kg rock)", color="gray")
        ax3.tick_params(axis="y", labelcolor="gray")
        formatter_vol = ScalarFormatter(useMathText=True)
        formatter_vol.set_scientific(True)
        formatter_vol.set_powerlimits((0, 0))
        ax3.yaxis.set_major_formatter(formatter_vol)
        curve_handle = Line2D([0], [0], color="gray", linestyle="-", linewidth=1.2, alpha=0.4, label="Volumetric H₂")
        handles.append(curve_handle)
    ax1.legend(handles=handles, title="Legend", loc="upper left")

    vols = [float(volume_at_temperature.get(tr, 0)) for tr in temp_bins]
    bar_colors = [cmap(norm(t_val)) for t_val in mid_temps]
    ax2.bar(mid_temps, vols, width=width * 3, color=bar_colors, edgecolor="black", linewidth=0.8, alpha=0.7)
    for x, v in zip(mid_temps, vols):
        if vols:
            ax2.text(x, v + max(vols) * 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax2.set_ylabel("Volume (km³)")
    ax2.set_xlabel("Temperature Range (°C)")
    ax2.grid(True, axis="y", linestyle=":", linewidth=0.5)

    ax2.set_xticks(mid_temps)
    ax2.set_xticklabels([tr.replace("_", "-") for tr in temp_bins], rotation=45)

    plt.tight_layout()
    if results_path:
        file_name_png = os.path.join(results_path, "combined_h2_and_volume.png")
        file_name_svg = os.path.join(results_path, "combined_h2_and_volume.svg")
        plt.savefig(file_name_png)
        plt.savefig(file_name_svg)
    plt.show()


def plot_fracture_mc_histogram(results):
    """Plot permeability distributions from the fracture Monte Carlo run."""
    k = results["k"]
    success = results["success"]
    fig, ax = plt.subplots(figsize=(10, 5))
    log_k_all = np.log10(k)
    log_k_s = np.log10(k[success])
    ax.hist(log_k_all, bins=50, color="lightgray", edgecolor="black")
    ax.hist(log_k_s, bins=50, color="green", alpha=0.6)
    plt.close(fig)
    return fig


def build_series(
    column_name,
    *,
    stats_mc_saturation,
    temp_ranges_plot,
    canonical_ranges,
    temp_mid_lookup,
    fill_missing=False,
):
    """Build a '(temperature, value)' series for plotting."""
    values = extract_column(
        column_name,
        stats_mc_saturation=stats_mc_saturation,
        temp_ranges_plot=temp_ranges_plot,
    )
    value_map = {tr: val for tr, val in zip(temp_ranges_plot, values)}
    series = []
    for tr in canonical_ranges:
        val = value_map.get(tr)
        temp_val = temp_mid_lookup.get(tr)
        if temp_val is None:
            continue
        if val is None or not np.isfinite(val):
            if fill_missing:
                series.append((temp_val, 0.0))
            continue
        series.append((temp_val, val))
    if fill_missing and not series:
        series = [(temp_mid_lookup.get(tr), 0.0) for tr in canonical_ranges if temp_mid_lookup.get(tr) is not None]
    return series
