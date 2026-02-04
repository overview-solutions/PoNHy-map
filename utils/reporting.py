from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from utils.general import build_thermo_lookup_by_range

REPORT_WIDTH = 150
REPORT_SEPARATOR = "-" * REPORT_WIDTH


def print_no_saturation_report(df):
    """Print formatted report for 'compute_saturation_from_volumetric' results."""
    if df is None or df.empty:
        print("No saturation results available.")
        return
    col_desc = df.attrs.get("column_descriptions", [])

    def fmt0(val, fmt, width=12):
        if val is None:
            return "-".rjust(width)
        try:
            v = float(val)
        except Exception:
            return "-".rjust(width)
        if not np.isfinite(v):
            return "-".rjust(width)
        return fmt.format(v).rjust(width)

    print("\n" + "H₂ Generation Daily Parameters NOT considering water saturation" + "\n")
    print(REPORT_SEPARATOR)
    print(
        f"{'Temp':<10}{'Pressure':>13}{'Solubility':>13}{'H2O front':>13}"
        f"{'H2O abs':>13}{'H₂ Diss':>13}{'H₂ Gas':>13}{'Daily H₂':>13}"
        f"{'Daily H₂':>13}{'Annual H₂':>13}{'Std clip':>13}"
        f"{'Sat ratio':>13}"
    )
    print(
        f"{'[°C]':<10}{'[MPa]':>13}{'[mol/kg]':>13}{'[kg/day]':>13}"
        f"{'[kg/day]':>13}{'[mol/day]':>13}{'[mol/day]':>13}{'[mol/day]':>13}"
        f"{'[kg/day]':>13}{'[tons/yr]':>13}{'[tons/yr]':>13}"
        f"{'[-]':>13}"
    )
    print(REPORT_SEPARATOR)

    for _, r in df.iterrows():
        print(
            f"{r['Temperature Range']:<10}"
            f"{fmt0(r['Pressure [MPa]'], '{:.2f}'):>13}"
            f"{fmt0(r['Solubility [mol/kg]'], '{:.2e}'):>13}"
            f"{fmt0(r['Water [kg/day]'], '{:.2e}'):>13}"
            f"{fmt0(r['Absorbed H₂O [kg/day]'], '{:.2e}'):>13}"
            f"{fmt0(r['Dissolvable [mol/day]'], '{:.2e}'):>13}"
            f"{fmt0(r['H2 gaseous [mol/day]'], '{:.2e}'):>13}"
            f"{fmt0(r['Actual [mol/day]'], '{:.2e}'):>13}"
            f"{fmt0(r['Actual [kg/day]'], '{:.2e}'):>13}"
            f"{fmt0(r['H2 [tons/yr]'], '{:.2f}'):>13}"
            f"{fmt0(r.get('Std clip [tons/yr]', 0.0), '{:.2f}'):>13}"
            f"{fmt0(r['Sat ratio'], '{:.2f}'):>13}"
        )

    print("\nDescriptions:")
    for desc in col_desc:
        if isinstance(desc, str) and ("Std Annual H₂" in desc or "p05" in desc or "p95" in desc or "p95-p05" in desc):
            continue
        print(desc)


def print_no_saturation_summary_report(df_no_saturation, total_tons_no_sat, total_kg_rocks, years, no_sat_csv_path):
    """Print summary for the no-saturation workflow."""
    print("\n" + f"Hydrogen generation summary (no saturation) saved to: {no_sat_csv_path}" + "\n")
    print("\n" + "Results for serpentinized rock mass without considering saturation of H2:")
    print(REPORT_SEPARATOR)
    print(f"{'Temperature Range [°C]':<50}{'Rocks [kg/day]':<50}")
    print(REPORT_SEPARATOR)

    rock_str = f"{total_kg_rocks:.2e}" if not np.isnan(total_kg_rocks) else "NaN"
    print(f"{'100_500':<50}{rock_str:<50}" + "\n")
    print("\n" + REPORT_SEPARATOR)
    print(f"{'ESTIMATED H₂ GENERATION SUMMARY NO SATURATION':^{REPORT_WIDTH}}")
    print(REPORT_SEPARATOR)
    print(f"{'Simulation duration:':<40} {years:>10,.0f} years")
    print(f"{'Total H₂ generated:':<40} {total_tons_no_sat:>10,.1f} tons")

    attrs = getattr(df_no_saturation, "attrs", {}) if df_no_saturation is not None else {}
    attr_iter_totals = attrs.get("iter_totals")
    attr_std_clip = attrs.get("std_total_clip", 0.0)

    std_total_mc = 0.0
    try:
        if attr_iter_totals is not None and len(attr_iter_totals) > 1:
            std_total_mc = float(np.std(np.asarray(attr_iter_totals, dtype=float), ddof=0))
        elif attr_std_clip:
            std_total_mc = float(attr_std_clip)
    except Exception:
        std_total_mc = 0.0

    if std_total_mc > 0:
        print(f"{'Std total (MC)  :':<40} {std_total_mc:>10,.1f} tons")

    print(REPORT_SEPARATOR)
    if std_total_mc > 0:
        print()
        print(
            "Std total (MC)  : Standard deviation of the total H₂ computed over every Monte Carlo iteration (captures cross-range covariance)."
        )
    print()


def print_saturation_monte_carlo_report(stats, years, flow_target):
    """Print formatted Monte Carlo saturation results."""
    if stats is None or stats.empty:
        print("No Monte Carlo results available.")
        return

    sat_column_descriptions = [
        " Temp        : Temperature range",
        " Flow Max    : Maximum clean water allocated per day (scaled flow target)",
        " Diff H2O    : Diffusive clean-water inflow",
        " Frac H2O    : Fracture-driven clean-water inflow",
        " H2O abs     : Clean water absorbed by the rock (not saturated)",
        " Solub       : Hydrogen solubility in water",
        " Saturation  : Dissolved H₂ per kilogram of delivered water",
        " Sat time    : Days required for pore water to reach ≥99% saturation",
        " Eff         : Efficiency relative to the no-saturation case",
        " H2 Diss     : Dissolved hydrogen",
        " H2 Gas      : Gaseous hydrogen",
        " H2 Total    : Total hydrogen produced",
        " Std H₂      : Standard deviation of total hydrogen per range",
        " Rocks       : Serpentinite mass participating in reactions",
        " W/R ratio   : Water-to-rock ratio",
    ]

    def fmt_compact(val, fmt_str):
        if isinstance(val, str):
            return val
        if val is None or not np.isfinite(val):
            return "-"
        return fmt_str.format(val)

    print(
        f"\n H₂ Production for Simulation Period ({years} years CONSIDERING water saturation and Target flow {flow_target} L/day)"
    )
    print("\n Production is limited to water saturation\n")
    header_fmt = (
        "{temp:<9} {flow:>11} {diff:>11} {frac:>11} {h2oabs:>11} "
        "{sol:>9} {sat:>9} {sat_time:>11} {eff:>7} {h2d:>11} {h2g:>11} {h2t:>11} "
        "{std:>11} {rocks:>13} {wr:>8}"
    )
    header_line = header_fmt.format(
        temp="Temp",
        flow="Flow",
        diff="Diff",
        frac="Frac",
        h2oabs="H2Oabs",
        sol="Sol",
        sat="Sat",
        sat_time="SatTime",
        eff="Eff%",
        h2d="H2Diss",
        h2g="H2Gas",
        h2t="H2Tot",
        std="Std",
        rocks="Rocks",
        wr="W/R",
    )
    units_line = header_fmt.format(
        temp="[°C]",
        flow="[kg/d]",
        diff="[kg/d]",
        frac="[kg/d]",
        h2oabs="[kg/d]",
        sol="[mol/kg]",
        sat="[mol/kg]",
        sat_time="[days]",
        eff="[%]",
        h2d="[tons/yr]",
        h2g="[tons/yr]",
        h2t="[tons/yr]",
        std="[tons/yr]",
        rocks="[kg/d]",
        wr="[-]",
    )
    dash_len = max(len(header_line), len(units_line))
    print("-" * dash_len)
    print(header_line)
    print(units_line)
    print("-" * dash_len)

    for temp_range in stats.index:
        r = stats.loc[temp_range]

        row_str = header_fmt.format(
            temp=str(temp_range)[:9],
            flow=fmt_compact(r["Flow target [kg/day]"]["mean"], "{:.2e}"),
            diff=fmt_compact(r["daily_diffused_H2O [kg/day]"]["mean"], "{:.2e}"),
            frac=fmt_compact(r["daily_fractured_H2O [kg/day]"]["mean"], "{:.2e}"),
            h2oabs=fmt_compact(r["H2O absorbed [kg/day]"]["mean"], "{:.2e}"),
            sol=fmt_compact(r["Solubility [mol/kg]"]["mean"], "{:.3f}"),
            sat=fmt_compact(r["Saturation [mol/kg]"]["mean"], "{:.3f}"),
            sat_time=fmt_compact(r.get(("Pore saturation time [days]", "mean")), "{:.1f}"),
            eff=fmt_compact(r["Efficiency [%]"]["mean"], "{:.2f}"),
            h2d=fmt_compact(r["H2 dissolved [tons]"]["mean"], "{:.2f}"),
            h2g=fmt_compact(r["H2 gaseous [tons]"]["mean"], "{:.2f}"),
            h2t=fmt_compact(r["H2 total [tons]"]["mean"], "{:.2f}"),
            std=fmt_compact(r["H2 total [tons]"]["std"], "{:.2f}"),
            rocks=fmt_compact(r["Rocks [kg/day]"]["mean"], "{:.2e}"),
            wr=fmt_compact(r["W/R ratio [-]"]["mean"], "{:.3f}"),
        )
        print(row_str)

    print("\n" + "Column meanings:")
    for desc in sat_column_descriptions:
        print(desc)


def print_inversion_config(topo_filename, dx, dy, dz, dist_x, dist_y, dist_z):
    """Print input file locations and core mesh parameters (units in meters)."""
    print("\n" + " Topography and Data Files ".center(REPORT_WIDTH, "-"))
    print(f"{'Topography file':<45}: {topo_filename}")
    print(f"{'Core cell size (dx, dy, dz) [m]':<45}: ({dx:.2f}, {dy:.2f}, {dz:.2f})")
    print("\n" + " Fracturing of the Serpentinite Rocks ".center(REPORT_WIDTH, "-"))
    print(f"{'Fracture spacing in X [m]':<45}: {dist_x:.2f}")
    print(f"{'Fracture spacing in Y [m]':<45}: {dist_y:.2f}")
    print(f"{'Fracture spacing in Z [m]':<45}: {dist_z:.2f}")


def print_inversion_config_summary(
    *,
    topo_filename,
    grav_filename,
    mag_filename,
    receiver_grav,
    receiver_mag,
    strength,
    inclination,
    declination,
    nx,
    ny,
    nz,
    total_distance_x,
    total_distance_y,
    total_distance_z,
    dx,
    dy,
    dz,
    use_initial_model,
    initial_model_filename,
    background_dens,
    background_susc,
    uncer_grav,
    uncer_mag,
    uncertainties_grav,
    uncertainties_mag,
    unit_labels,
    unit_dens_adj,
    unit_magsus,
    unit_dens_disp,
    unit_magsus_disp,
    alpha0_ratio,
    beta,
    beta_it,
    targets,
    mref_in_smooth,
    update_smallness,
    scaling_init,
    separator="=",
    section_separator="-",
):
    """Print a formatted summary of the current inversion configuration values."""
    width = len(section_separator) if section_separator else REPORT_WIDTH
    dash = section_separator[0] if section_separator else "-"
    print("\n" + " Topography and Data Files ".center(width, dash))
    print(f"{'Topography file':<45}: {topo_filename}")
    print(f"{'Gravity data file':<45}: {grav_filename}")
    print(f"{'Magnetic data file':<45}: {mag_filename}")

    print("\n" + " Survey Details ".center(width, dash))
    print(f"{'Gravity receiver components':<45}: {receiver_grav.components}")
    print(f"{'Magnetic receiver components':<45}: {receiver_mag.components}")
    print(f"{'Magnetic field strength [nT]':<45}: {strength}")
    print(f"{'Inclination / Declination [°]':<45}: {inclination}, {declination}")

    print("\n" + " Mesh Parameters ".center(width, dash))
    print(f"{'Mesh dimensions (nx, ny, nz)':<45}: ({nx}, {ny}, {nz})")
    print(
        f"{'Total extent (X, Y, Z) [m]':<45}: ({total_distance_x:.2f}, {total_distance_y:.2f}, {total_distance_z:.2f})"
    )
    print(f"{'Core cell size (dx, dy, dz) [m]':<45}: ({dx:.2f}, {dy:.2f}, {dz:.2f})")

    print("\n" + " Model Parameters ".center(width, dash))
    if use_initial_model:
        print(f"{'Initial model':<45}: Used")
        print(f"{'Initial model file':<45}: {initial_model_filename}")
    else:
        print(f"{'Initial model':<45}: Not used (background values)")
        print(f"{'Background density [g/cc]':<45}: {background_dens:.2e}")
        print(f"{'Background susceptibility [SI]':<45}: {background_susc:.2e}")

    print("\n" + " Uncertainties ".center(width, dash))
    print(f"{'Gravity uncertainty [%] / abs [mGal]':<45}: {uncer_grav:.2f}% = {uncertainties_grav[0]:.2f}")
    print(f"{'Magnetic uncertainty [%] / abs [nT]':<45}: {uncer_mag:.2f}% = {uncertainties_mag[0]:.5f}")

    print("\n" + " Gaussian Mixture Model (GMM) Parameters ".center(width, dash))
    print(" Physical property means per unit ".center(width, dash))
    for label, dens, susc in zip(unit_labels, unit_dens_adj, unit_magsus):
        label_fmt = f"  · {label}"
        print(f"{label_fmt:<25}-> Density: {dens:.2f} g/cc, Susceptibility: {susc:.5f} SI")

    print("\n" + " Dispersion per unit ".center(width, dash))
    for label, dens_disp, susc_disp in zip(unit_labels, unit_dens_disp, unit_magsus_disp):
        label_fmt = f"  · {label}"
        print(f"{label_fmt:<25}-> σ_density: {dens_disp:.2f}, σ_susc: {susc_disp:.5f}")

    print("\n" + " PGI Regularization Parameters ".center(width, dash))
    print(f"{'alpha_pgi':<45}: {1.0:.2f}")
    print(f"{'alpha_x/y/z':<45}: ({1.0:.2f}, {1.0:.2f}, {1.0:.2f})")
    print(f"{'alpha_xx/yy/zz':<45}: ({0.0:.2f}, {0.0:.2f}, {0.0:.2f})")

    print("\n" + " Directive Parameters ".center(width, dash))
    print(f"{'Alpha0 ratio':<45}: {alpha0_ratio}")
    print(f"{'Beta estimate ratio':<45}: {beta.beta0_ratio:.2e}")
    print(f"{'Beta cooling factor':<45}: {beta_it.coolingFactor:.2f}")
    print(f"{'Beta tolerance':<45}: {beta_it.tolerance:.2f}")
    print(f"{'Target misfit tolerance':<45}: {targets.verbose}")
    print(f"{'MrefInSmooth wait till stable':<45}: {mref_in_smooth.wait_till_stable}")
    print(f"{'Update GMM in smallness':<45}: {update_smallness.update_gmm}")
    print(f"{'Scaling init chi0_ratio':<45}: {scaling_init.chi0_ratio}")


def print_saturation_summary(total_kg_rocks, years, total_tons_sat, std_total_mc: float, mean_efficiency):
    """Print the final saturation summary for H₂ generation."""
    print("\n" + "Serpentinized rock mass:")
    print(REPORT_SEPARATOR)
    print(f"{'Temperature Range [°C]':<50}{'Rocks [kg/day]':<50}")
    print(REPORT_SEPARATOR)
    print(f"{'100_500':<50}{total_kg_rocks:<50.2e}")
    print(REPORT_SEPARATOR)

    print("\n" + REPORT_SEPARATOR)
    print(f"{'FINAL ESTIMATED H₂ GENERATION SUMMARY':^{REPORT_WIDTH}}")
    print(REPORT_SEPARATOR)
    print(f"{'Simulation duration:':<40} {years:>10,.1f} years")

    print(f"{'Total H₂ generated:':<40} {total_tons_sat:>10,.1f} tons")
    if std_total_mc and std_total_mc > 0:
        print(f"{'Std total (MC)  :':<40} {std_total_mc:>10,.1f} tons")

    print(f"{'Production efficiency:':<40} {mean_efficiency:>10.2f}%")
    print(REPORT_SEPARATOR)
    if std_total_mc and std_total_mc > 0:
        print()
        print(
            "Std total (MC)  : Standard deviation of the total H₂ computed over every Monte Carlo iteration (captures cross-range covariance)."
        )
    print()


def print_summary_serpentinization_section(
    volume_density_magsus,
    volume_at_temperature_total_100_500,
    mode_density,
    mode_magsus,
    volume_at_temperature,
    depths_for_temp_extremes,
    serpentinization_degree,
    serp_corr_percentage,
    temperature_ranges,
):
    """Print the high-level serpentinization summary."""
    print(" Fracturing of the Serpentinite Rocks ".center(REPORT_WIDTH, "-"))
    print(f"  Estimated volume of serpentinite rocks: {volume_density_magsus:.2f} km³")
    print(
        "  Estimated volume at correct temperature window (100 to 500°C):"
        f"  {volume_at_temperature_total_100_500:.2f} km³"
    )
    print(f"  Avg density: {mode_density:.2f} g/cm³")
    print(f"  Avg susceptibility: {mode_magsus:.6f} SI \n")

    serpentinization_corrections, total_volume = print_partial_volumes_and_serpentinization(
        volume_at_temperature,
        depths_for_temp_extremes,
        serpentinization_degree,
        serp_corr_percentage,
        temperature_ranges,
    )

    print("\n" + "Average serpentinite rock data ".center(REPORT_WIDTH, "-") + "\n")

    if total_volume > 0:
        print("Estimated reduction of the blocks 'serpentinization degree':")
        print(f"  From inversion: Avg: {serpentinization_degree:.2f}%\n")
    else:
        print("No valid temperature ranges to calculate average serpentinization\n")

    return serpentinization_corrections, total_volume


def print_parameters_water_flow_serpentinization(
    flow_target,
    int_fracture_spacing,
    permeability_fractures,
    porosity_front,
    density_serpentinite,
    waterrockratio,
    flow_target_fracture_config,
    density_litho,
    gravity,
    years,
    lithology_code,
    molar_mass_h2,
    molar_mass_h2o,
):
    """Print section '5. PARAMETERS FOR WATER FLOW & SERPENTINIZATION'."""
    print()
    print(f"Target flow to sustain reaction:         {flow_target:,.2e} L/day")
    print(f"Internal fracture spacing:               {int_fracture_spacing:.2f} m")
    print(f"Fracture permeability:                   {permeability_fractures:.1e} m²")
    print(f"Porosity at serpentinization front:      {porosity_front:.1f} %")
    print(f"Serpentinite density:                    {density_serpentinite:,.0f} kg/m³")
    print(f"Water-rock ratio (W/R):                  {waterrockratio:.2f}")
    print()
    print(f"Fault segment length (L_fault):          {flow_target_fracture_config['L_fault']:,} m")
    print(f"Lithostatic reference density:           {density_litho} kg/m³")
    print(f"Gravity acceleration:                    {gravity:.2f} m/s²")
    print(f"Simulation duration:                     {years} year(s)")
    print(f"Lithology code used:                     {lithology_code}")
    print()
    print(f"Molar mass of H₂:                        {molar_mass_h2:.8f} kg/mol")
    print(f"Molar mass of H₂O:                       {molar_mass_h2o:.8f} kg/mol")


def print_production_rate_volumetric_report(
    thermo_data,
    lithology_code,
    waterrockratio,
    temperature_ranges,
    mean_pressure_ranges,
    lithologies_dict=None,
    thermo_lookup=None,
):
    """Print a formatted table of hydrogen production stats for a lithology and W/R ratio."""
    if thermo_lookup is None:
        thermo_lookup = build_thermo_lookup_by_range(
            thermo_data,
            lithology_code,
            waterrockratio,
            temperature_ranges,
            mean_pressure_ranges,
        )

    if lithologies_dict and lithology_code in lithologies_dict:
        comp = lithologies_dict[lithology_code]
        comp_list = [f"{ox}={val:.2f}" for ox, val in comp.items()]
        comp_str = "  ".join(comp_list)
        print("Composition in %:")
        while comp_str:
            print(comp_str[:REPORT_WIDTH])
            comp_str = comp_str[REPORT_WIDTH:]
    else:
        print("Composition: No composition data available.")

    print("\n" + "Reacting fluid: Pure water" + "\n")
    print("-" * 150)
    print(f"{'Temp Range':<15}{'Pressure':>10}{'mol/kg H₂':>15}{'n_H₂':>12}{'V_H₂':>12}"
          f"{'n_H₂O_sol':>15}{'ρ H₂O':>12}")
    print(f"{'[°C]':<15}{'[bar]':>10}{'[mol/kg]':>15}{'[mol]':>12}{'[m³]':>12}"
          f"{'[mol]':>15}{'[kg/m³]':>12}")
    print("-" * 150)
    for tr in temperature_ranges:
        stats = thermo_lookup.get(tr) if thermo_lookup else None
        if stats is None:
            mol_kg_h2 = n_h2 = v_h2 = n_h2o_solids = rho_h2o = 0.0
            p_bar = mean_pressure_ranges.get(tr, float("nan"))
        else:
            mol_kg_h2 = stats["mol_kg_h2"]
            n_h2 = stats["n_h2"]
            v_h2 = stats["v_h2"]
            n_h2o_solids = stats["n_h2o_solids"]
            rho_h2o = stats["rho_h2o"]
            p_bar = stats["pressure_bar"]
        print(
            f"{tr:<15}{p_bar:>10.1f}{mol_kg_h2:>15.2e}{n_h2:>12.2e}"
            f"{v_h2:>12.2e}{n_h2o_solids:>15.2e}{rho_h2o:>12.1f}"
        )

    print("\n" + "Descriptions:")
    print(" mol/kg rock  : Hydrogen generated per unit mass of rock")
    print(" n_H₂         : Total hydrogen amount in moles")
    print(" V_H₂         : Total hydrogen volume in cubic meters")
    print(" n_H₂O_sol    : Water moles incorporated into the solid phase (serpentinization)")
    print(" ρ H₂O        : Water density at given pressure and temperature [kg/m³]" + "\n")


def print_partial_volumes_and_serpentinization(
    volume_at_temperature,
    depths_for_temp_extremes,
    serpentinization_degree,
    serp_corr_percentage,
    temperature_ranges,
):
    """Print partial serpentinite volumes and serpentinization corrections."""
    print("\n" + "Partial volumes and serpentinization degree")
    print("-" * 150)
    print(f"{'Temp Range':<16}{'Volume':<13}{'Depth':<13}{'Serp.':^10}{'Correction':^11}")
    print(f"{'[°C]':<16}{'[km³]':<13}{'[m]':<13}{'[%]':^10}{'[-]':^11}")
    print("-" * 150)

    total_volume = 0
    serpentinization_corrections = {}

    for temp_range in temperature_ranges:
        volume = volume_at_temperature.get(temp_range, 0)
        total_volume += volume
        min_depth, max_depth = depths_for_temp_extremes[temp_range]
        depth_range = f"{round(min_depth):.0f}-{round(max_depth):.0f}"
        percentage_affected = serpentinization_degree
        correction = 1 - np.interp(
            percentage_affected,
            serp_corr_percentage["percentage"],
            serp_corr_percentage["correction"],
        ) / 100

        serpentinization_corrections[temp_range] = {"avg": correction}

        print(f"{temp_range:<16}{volume:<13.2f}{depth_range:<13}" f"{int(percentage_affected):^10}{correction:^11.2f}")

    print("\n" + "Descriptions:")
    print(" Temp Range      : Temperature interval associated with each block")
    print(" Volume          : Volume of serpentinite rock at that temperature range")
    print(" Depth           : Minimum and maximum depths corresponding to the temperature range")
    print(" Serp. [%]       : Estimated percentage of rock affected by serpentinization")
    print(" Correction      : Reduction factor to account for incomplete serpentinization [0-1]" + "\n")
    return serpentinization_corrections, total_volume


def save_mc_convergence_sweep_report(sweep_df, results_dir, save_plot=True):
    """Save artifacts and plot results for the MC convergence sweep."""
    try:
        os.makedirs(results_dir, exist_ok=True)
        csv_path = os.path.join(results_dir, "mc_convergence_sweep.csv")
        sweep_df.to_csv(csv_path, index=False)
        print(f"Convergence sweep summary saved to: {csv_path}")
        if save_plot:
            fig, ax = plt.subplots(figsize=(8, 4))
            n_array = np.asarray(sweep_df["n_iter"].astype(float).to_numpy(), dtype=float)
            mean_array = np.asarray(sweep_df["mean_total_tons"].astype(float).to_numpy(), dtype=float)
            std_array = np.asarray(sweep_df["std_total_tons"].astype(float).to_numpy(), dtype=float)
            ax.plot(n_array, mean_array, marker="o", label="Mean")
            ax.errorbar(
                n_array,
                mean_array,
                yerr=std_array,
                fmt="none",
                color="skyblue",
                ecolor="skyblue",
                elinewidth=2,
                capsize=4,
                label="Mean ± Std",
            )
            ax.set_xlabel("n_iter")
            ax.set_ylabel("Total H₂ [tons]")
            ax.set_title("Monte Carlo convergence (saturation)")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            plot_path = os.path.join(results_dir, "mc_convergence_sweep.png")
            fig.savefig(plot_path, dpi=300)
            plt.close(fig)
            print(f"Convergence sweep plot saved to: {plot_path}")
    except Exception as err:
        print(f"[MC Convergence] Warning: failed to save sweep artifacts: {err}")


def print_fracture_monte_carlo_report(results):
    """Print a concise summary report for fracture Monte Carlo results."""
    print(" FRACTURE MONTE CARLO — RESULTS SUMMARY")
    print("-" * 150)
    print(f"Flow target: {results['flow_target']:,.2e} L/day")
    print(f"Success rate: {results['success_rate']:.2f} %")


def print_inversion_results_summary(
    *,
    separator: str,
    section_separator: str,
    results_path: str,
    formatted_time: str,
    convergence_status: Optional[str] = None,
    converged: Optional[bool] = None,
    opt_iter: Optional[int] = None,
    opt_max_iter: Optional[int] = None,
) -> None:
    """Print the inversion results summary block."""
    status_label = "Process completed"
    if converged is not None:
        status_label = f"Process completed — Converged: {'YES' if converged else 'NO'}"
    print("\n" + section_separator)
    print(f"{status_label:^{REPORT_WIDTH}}")
    print(section_separator)
    print("\nAll main results have been saved:")
    print(f"  • Density model:         {os.path.join(results_path, 'Density_complete_model.csv')}")
    print(f"  • Magnetic susceptibility: {os.path.join(results_path, 'Magsus_complete_model.csv')}")
    print(f"  • Geology model:         {os.path.join(results_path, 'Geology_complete_model.csv')}")
    print(f"  • Serpentinite density:  {os.path.join(results_path, 'Density_Serpentinite.csv')}")
    print(f"  • Serpentinite magsus:   {os.path.join(results_path, 'Magsus_Serpentinite.csv')}")
    print("\n" + f"Execution time: {formatted_time}")
    if convergence_status:
        print(f"Convergence status: {convergence_status}")
    if converged is not None:
        print(f"Converged: {'YES' if converged else 'NO'}")
    if opt_iter is not None and opt_max_iter is not None:
        print(f"Iterations: {opt_iter}/{opt_max_iter}")
