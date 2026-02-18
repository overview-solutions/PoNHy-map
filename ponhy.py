import csv
import math
import os
import random
import sys
import time
import warnings
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    cast,
)
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
import discretize
import discretize.utils
import simpeg.potential_fields as pf
from discretize.utils import (
    active_from_xyz,
)
from scipy.interpolate import griddata
from simpeg import (
    data,
    data_misfit,
    directives,
    inversion,
    inverse_problem,
    maps,
    optimization,
    regularization,
    utils,
)

from utils.geometry import (
    build_mesh_from_topography,
    compute_rock_volumes_and_masks,
    compute_temp_range_depths,
    interpolate_nearest_neighbor,
)
from utils.helpers import (
    _as_scalar_value,
    _clamp_worker_count,
    _format_value_for_filename,
    _get_progress_bar,
    _is_missing,
    _moving_average,
    _normalize_unit_parameters,
    _print_header,
    _print_header_once,
    _resolve_base_dir_path,
    _report_missing_params,
    _sample_unit_hypercube,
    _scale_samples_to_ranges,
    _select_stable_window,
    _trim_trailing_dropoff,
)
from utils.no_saturation import run_no_saturation_workflow
from utils.plotting import (
    _save_plot_pair,
    build_series,
    plot_fracture_mc_histogram,
    plot_gravity_observed_predicted_residual,
    plot_lab_data_panels,
    plot_learned_gmm,
    plot_misfit_evolution,
    plot_magnetic_observed_predicted_residual,
    plot_residual_histograms,
    prepare_lab_plot_inputs,
    set_plot_save_svg,
)
from utils.logging import CustomPrint, log_info
from utils.reporting import (
    print_fracture_monte_carlo_report,
    print_inversion_config_summary,
    print_inversion_config,
    print_inversion_results_summary,
    print_no_saturation_report,
    print_no_saturation_summary_report,
    print_parameters_water_flow_serpentinization,
    print_production_rate_volumetric_report,
    print_saturation_monte_carlo_report,
    print_saturation_summary,
    print_summary_serpentinization_section,
    save_mc_convergence_sweep_report,
)
from utils.saturation import run_saturation_workflow
from utils.uncertainties import (
    analyze_limiting_factors_by_flow_target,
    run_fracture_monte_carlo_simulation,
    run_mc_convergence_sweep,
    run_no_saturation_univariate_sweep,
    run_saturation_univariate_sweep,
)
from utils.general import (
    build_production_rate_volumetric_dict,
    build_thermo_lookup_by_range,
    compute_h2_solubility_kk_pr,
    compute_mean_lithostatic_pressure_by_range,
    compute_serpentinization_degree,
    compute_serpentinization_front_velocities,
    convert_h2_mol_to_kg,
    extract_column,
    load_h2_production_database,
    plot_serpentinization_heatmap,
    parse_temperature_range,
    save_saturation_csv,
    sort_temperature_ranges,
    weighted_average_rates,
)
from utils.config import (
    build_flow_target_limiting_factors_params,
    build_inversion_config_summary,
    build_inversion_mesh_config,
    build_inversion_results_summary,
    build_mc_common_kwargs,
    build_mc_no_sat_kwargs,
    build_mean_lithostatic_pressure_params,
    build_no_saturation_workflow_params,
    build_production_rate_report_params,
    build_production_rate_volumetric_params,
    build_saturation_workflow_params,
    build_serpentinization_front_velocity_params,
    build_serpentinization_summary_params,
    build_thermo_lookup_params,
    Config,
    InversionConfigSummary,
    InversionMeshConfig,
    InversionResultsSummary,
    MeanLithostaticPressureParams,
    NoSaturationSummaryReportParams,
    NoSaturationWorkflowParams,
    ProductionRateReportParams,
    ProductionRateVolumetricParams,
    SaturationMonteCarloReportParams,
    SaturationSummaryParams,
    SaturationWorkflowParams,
    SerpentinizationFrontVelocityParams,
    SerpentinizationSummaryParams,
    ThermoLookupParams,
    WaterFlowSerpentinizationParams,
    load_config,
)

def _select_config_yaml(default_name: str = "ponhy_config_pyrenees.yaml") -> str:
    """Select a YAML config from the current working directory.

    When multiple YAML files exist, prompt the user to choose one. If stdin is not
    interactive or the input is invalid, fall back to the first candidate.
    """
    cwd = os.getcwd()
    candidates = [
        os.path.join(cwd, name)
        for name in sorted(os.listdir(cwd))
        if name.lower().endswith((".yaml", ".yml"))
        and name.lower() != "environment.yml"
    ]

    if not candidates:
        raise FileNotFoundError(
            "No YAML config files found in the current working directory. "
            "Please add a ponhy_config_*.yaml file and try again."
        )

    if len(candidates) == 1:
        return candidates[0]

    print("\n[PoNHy] Config YAMLs found:")
    for idx, path in enumerate(candidates, start=1):
        print(f"  {idx}) {os.path.basename(path)}")

    if not sys.stdin.isatty():
        print("[PoNHy] Non-interactive session detected. Using the first YAML.")
        return candidates[0]

    while True:
        choice = input("Select YAML to use (number -> Enter): ").strip()
        if not choice:
            return candidates[0]
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(candidates):
                return candidates[idx - 1]
        print("Invalid selection. Try again.")


def _discover_data_dirs(root: str) -> List[str]:
    if not root or not os.path.isdir(root):
        return []
    return [
        os.path.join(root, name)
        for name in sorted(os.listdir(root))
        if name.lower().startswith("data") and os.path.isdir(os.path.join(root, name))
    ]


def _select_data_dir(root: str) -> Optional[str]:
    candidates = _discover_data_dirs(root)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    print("\n[PoNHy] Data folders found:")
    for idx, path in enumerate(candidates, start=1):
        print(f"  {idx}) {path}")

    if not sys.stdin.isatty():
        raise RuntimeError("Multiple Data folders found but no interactive terminal to select one.")

    while True:
        choice = input("Select Data folder (number -> Enter): ").strip()
        if not choice:
            return candidates[0]
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(candidates):
                return candidates[idx - 1]
        print("Invalid selection. Try again.")


def _rewrite_data_path(base_dir: str, data_dir: str, path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    if not isinstance(path, str):
        path = str(path)
    cleaned = path.strip()
    if not cleaned:
        return None

    if os.path.isabs(cleaned):
        return cleaned

    has_sep = any(sep in cleaned for sep in (os.sep, "/", "\\"))
    if has_sep:
        return os.path.join(base_dir, cleaned)

    return os.path.join(data_dir, cleaned)


CONFIG_YAML_FILE = _select_config_yaml()

# ========================================================  Dictionaries and data for calculations ========================================================
# Compositions of the rocks estimated in thermodynamic databases. This is just for info and printing (not used in calculations)
lithologies_dict = {
    'HZ1': {
        'Al2O3': 3.13, 'CaO': 1.25, 'CoO': 0.0, 'Cr2O3': 0.0, 'FeO': 8.69, 'Fe2O3': 0.0,
        'H2O': 0.0, 'K2O': 0.0, 'MnO': 0.0, 'MgO': 43.53, 'Na2O': 0.08, 'NiO': 0.0,
        'P2O5': 0.0, 'SiO2': 43.33, 'TiO2': 0.0
    },
    'LH1': {
        'Al2O3': 4.46, 'CaO': 2.65, 'CoO': 0.0, 'Cr2O3': 0.0, 'FeO': 7.86, 'Fe2O3': 0.0,
        'H2O': 0.0, 'K2O': 0.0, 'MnO': 0.0, 'MgO': 40.13, 'Na2O': 0.16, 'NiO': 0.0,
        'P2O5': 0.0, 'SiO2': 44.75, 'TiO2': 0.0
    }
}

# Data for volumetric serpentinization degree interpolation (x: density (g/cm3), y: magnetic susceptibility (SI), value: serpentinization degree (%))
# This represents how much of the block is 100% serpentinized (e.g., 50% means half of the block is fully serpentinized, rest is unaltered)
serpentinization_data = {
    'x_points': np.array([2.62  , 2.6355, 2.651 , 2.6665, 2.682 , 2.6975, 2.713 , 2.7285,
                          2.744 , 2.7595, 2.775 , 2.7905, 2.806 , 2.8215, 2.837 , 2.8525,
                          2.868 , 2.8835, 2.899 , 2.9145, 2.93  , 2.9455, 2.961 , 2.9765,
                          2.992 , 3.0075, 3.023 , 3.0385, 3.054 , 3.0695, 3.085 , 3.1005,
                          3.116 , 3.1315, 3.147 , 3.1625, 3.178 , 3.1935, 3.209 , 3.2245,
                          3.24  ]),
       
    'y_points': np.array([0.07     , 0.0682775, 0.066555 , 0.0648325, 0.06311  , 0.0613875,
                          0.059665 , 0.0579425, 0.05622  , 0.0544975, 0.052775 , 0.0510525,
                          0.04933  , 0.0476075, 0.045885 , 0.0441625, 0.04244  , 0.0407175,
                          0.038995 , 0.0372725, 0.03555  , 0.0338275, 0.032105 , 0.0303825,
                          0.02866  , 0.0269375, 0.025215 , 0.0234925, 0.02177  , 0.0200475,
                          0.018325 , 0.0166025, 0.01488  , 0.0131575, 0.011435 , 0.0097125,
                          0.00799  , 0.0062675, 0.004545 , 0.0028225, 0.0011 ]),
       
    'values': np.array([100. ,  97.5,  95. ,  92.5,  90. ,  87.5,  85. ,  82.5,  80. ,
                         77.5,  75. ,  72.5,  70. ,  67.5,  65. ,  62.5,  60. ,  57.5,
                         55. ,  52.5,  50. ,  47.5,  45. ,  42.5,  40. ,  37.5,  35. ,
                         32.5,  30. ,  27.5,  25. ,  22.5,  20. ,  17.5,  15. ,  12.5,
                         10. ,   7.5,   5. ,   2.5,   0. ])
}

# Values of corrections for hydrogen production according to the volumetric serpentinization degree (content of remaining olivine in the block). 
serp_corr_percentage = {
    'percentage': np.array([0,       5,   10,    15,    20,    25,    30,    35,    40,    45,    50,    55,    60,    65,    70,    75,    80,    85,    90,   95,   100]),
    'correction': np.array([0.00, 3.29, 6.65, 10.06, 13.55, 17.10, 20.74, 24.46, 28.28, 32.21, 36.26, 40.45, 44.80, 49.33, 54.08, 59.11, 64.48, 70.33, 76.89, 84.70, 98.00])
}

cfg: Config = load_config(CONFIG_YAML_FILE)

# Auto-detect base_dir unless a manual path is provided in YAML.
base_dir_candidate = (cfg.base_dir or "").strip()
if not base_dir_candidate or base_dir_candidate.lower() in {"auto", "none"}:
    base_dir_candidate = os.path.abspath(os.getcwd())
cfg.base_dir = base_dir_candidate

data_keys = (
    "topo_file",
    "grav_file",
    "mag_file",
    "initial_model",
    "db_dir",
    "temperature_file",
    "density_file",
    "magsus_file",
)
data_dir_selected = _select_data_dir(cfg.base_dir)
if data_dir_selected is None:
    raise FileNotFoundError(
        "No Data* folders found under base_dir. "
        "Add a Data_... folder or update the config paths."
    )

for key in data_keys:
    value = getattr(cfg, key, None)
    setattr(cfg, key, _rewrite_data_path(cfg.base_dir, data_dir_selected, value))

# Bind config values into module-level names.
run_inversion = cfg.run_inversion
run_h2_quantification = cfg.run_h2_quantification
base_dir = cfg.base_dir
topo_file = cfg.topo_file
grav_file = cfg.grav_file
mag_file = cfg.mag_file
initial_model = cfg.initial_model
use_initial_model = cfg.use_initial_model
background_dens = cfg.background_dens
background_susc = cfg.background_susc
db_dir = cfg.db_dir
temperature_file = cfg.temperature_file
density_file = cfg.density_file
magsus_file = cfg.magsus_file
dx = cfg.dx
dy = cfg.dy
dz = cfg.dz
depth_core = cfg.depth_core
expansion_percentage = cfg.expansion_percentage
expansion_type = cfg.expansion_type
dominant_side = cfg.dominant_side
original_y = cfg.original_y
original_x = cfg.original_x
uncer_grav = cfg.uncer_grav
uncer_mag = cfg.uncer_mag
inclination = cfg.inclination
declination = cfg.declination
strength = cfg.strength
unit_labels = cfg.unit_labels
unit_dens_adj_list = cfg.unit_dens_adj_list
unit_magsus_list = cfg.unit_magsus_list
unit_dens_disp_list = cfg.unit_dens_disp_list
unit_magsus_disp_list = cfg.unit_magsus_disp_list
vol_unit_list = cfg.vol_unit_list
serpentinite_label = cfg.serpentinite_label
low_dens_adj = cfg.low_dens_adj
up_dens_adj = cfg.up_dens_adj
lower_susceptibility = cfg.lower_susceptibility
upper_susceptibility = cfg.upper_susceptibility
max_iter = cfg.max_iter
max_iter_ls = cfg.max_iter_ls
max_iter_cg = cfg.max_iter_cg
tol_cg = cfg.tol_cg
save_pred_residual_npy = cfg.save_pred_residual_npy
save_svg = cfg.save_svg
alpha_pgi = cfg.alpha_pgi
alpha_x = cfg.alpha_x
alpha_y = cfg.alpha_y
alpha_z = cfg.alpha_z
alpha_xx = cfg.alpha_xx
alpha_yy = cfg.alpha_yy
alpha_zz = cfg.alpha_zz
alpha0_ratio_dens = cfg.alpha0_ratio_dens
alpha0_ratio_susc = cfg.alpha0_ratio_susc
beta0_ratio = cfg.beta0_ratio
cooling_factor = cfg.cooling_factor
tolerance = cfg.tolerance
progress = cfg.progress
use_chi_small = cfg.use_chi_small
chi_small = cfg.chi_small
wait_till_stable = cfg.wait_till_stable
update_gmm = cfg.update_gmm
kappa = cfg.kappa
chi0_ratio = cfg.chi0_ratio
seed = cfg.seed
use_global_seed = cfg.use_global_seed
dist_x = cfg.dist_x
dist_y = cfg.dist_y
dist_z = cfg.dist_z
density_serpentinite = cfg.density_serpentinite
porosity_front = cfg.porosity_front
int_fracture_spacing = cfg.int_fracture_spacing
permeability_fractures = cfg.permeability_fractures
waterrockratio = cfg.waterrockratio
flow_target = cfg.flow_target
density_litho = cfg.density_litho
gravity = cfg.gravity
lithology_code = cfg.lithology_code
years = cfg.years
molar_mass_h2 = cfg.molar_mass_h2
molar_mass_h2o = cfg.molar_mass_h2o
v_ref_synthetic = cfg.v_ref_synthetic
t_ref_range = cfg.t_ref_range
n_cores = cfg.n_cores
run_montecarlo_fault = cfg.run_montecarlo_fault
fault_mc_n_iter = cfg.fault_mc_n_iter
flow_target_fracture_config = asdict(cfg.flow_target_fracture_config)
mc_no_saturation_config = asdict(cfg.mc_no_saturation_config)
mc_saturation_config = asdict(cfg.mc_saturation_config)
run_univariate_analysis_no_sat = cfg.run_univariate_analysis_no_sat
run_univariate_analysis_sat = cfg.run_univariate_analysis_sat
univariate_analysis_config = asdict(cfg.univariate_analysis_config)
run_mc_convergence_sweep_flag = cfg.run_mc_convergence_sweep
mc_convergence_sweep_config = asdict(cfg.mc_convergence_sweep_config)
run_analyze_limiting_factors = cfg.run_analyze_limiting_factors
flow_target_log_min = cfg.flow_target_log_min
flow_target_log_max = cfg.flow_target_log_max
flow_target_n_samples = cfg.flow_target_n_samples
mc_flow_target_config = asdict(cfg.mc_flow_target_config)

# Runtime-only values that should not live in the config object.
runtime_state: Dict[str, Any] = {"surface_area_per_km3": None}


def _compute_surface_area_per_km3(dist_x: float, dist_y: float, dist_z: float) -> float:
    original_volume_m3 = 1000 ** 3  # 1 km³
    voxel_volume_m3 = dist_x * dist_y * dist_z
    total_voxels = original_volume_m3 / voxel_volume_m3
    voxel_surface_area = 2 * (dist_x * dist_y + dist_x * dist_z + dist_y * dist_z)
    return voxel_surface_area * total_voxels

# Configure global plotting settings (e.g., optional SVG output).
set_plot_save_svg(save_svg)


if __name__ != "__main__":
    run_inversion = False
    run_h2_quantification = False

######################################################################################################################################################
################################################################# Inversion ##########################################################################
######################################################################################################################################################

INVERSION_PARAM_NAMES = [
    "base_dir", "topo_file", "grav_file", "mag_file", "initial_model", "use_initial_model",
    "background_dens", "background_susc", "dx", "dy", "dz", "depth_core",
    "expansion_percentage", "expansion_type", "dominant_side", "original_y", "original_x",
    "uncer_grav", "uncer_mag", "inclination", "declination", "strength", "unit_labels",
    "unit_dens_adj_list", "unit_magsus_list", "unit_dens_disp_list", "unit_magsus_disp_list",
    "vol_unit_list", "serpentinite_label", "low_dens_adj", "up_dens_adj", "lower_susceptibility",
    "upper_susceptibility", "max_iter", "max_iter_ls", "max_iter_cg", "tol_cg",
    "save_pred_residual_npy", "save_svg", "alpha_pgi", "alpha_x", "alpha_y", "alpha_z", "alpha_xx",
    "alpha_yy", "alpha_zz",
    "alpha0_ratio_dens", "alpha0_ratio_susc", "beta0_ratio", "cooling_factor", "tolerance",
    "progress", "use_chi_small", "chi_small", "wait_till_stable", "update_gmm", "kappa",
    "chi0_ratio",
]


if run_inversion:
    # Print the report header once for the full run (stdout is redirected later).
    _print_header_once()

    print("\nPlease wait, it can take a few minutes.\n")

    # Validate required inversion parameters are present and non-empty.
    inversion_inputs = {name: getattr(cfg, name, None) for name in INVERSION_PARAM_NAMES}
    missing_inversion = [name for name in INVERSION_PARAM_NAMES if _is_missing(inversion_inputs.get(name))]
    if missing_inversion:
        # Report missing parameters with context-specific messaging.
        _report_missing_params(missing_inversion, "inversion")
        raise ValueError("Missing required inversion parameters.")

    start_time = time.time()
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    dir_path = os.path.join(base_dir, "")

    folder_name = f"Results_Inversion_{current_time}"
    results_path = os.path.join(dir_path, folder_name)

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    output_file_path = os.path.join(results_path, "output.txt")

    # Resolve input data paths relative to base_dir.
    topo_filename = _resolve_base_dir_path(base_dir, topo_file)
    grav_data_filename = _resolve_base_dir_path(base_dir, grav_file)
    mag_data_filename = _resolve_base_dir_path(base_dir, mag_file)

    assert topo_filename is not None, "Resolved topo_filename is required"
    assert grav_data_filename is not None, "Resolved grav_data_filename is required"
    assert mag_data_filename is not None, "Resolved mag_data_filename is required"

     # Resolve and load the initial model CSV path for the inversion only when enabled.
    if use_initial_model:
        initial_model_filename = _resolve_base_dir_path(base_dir, initial_model)
        assert initial_model_filename is not None, "Resolved initial_model_filename is required"
        if os.path.isdir(initial_model_filename):
            raise IsADirectoryError(
                f"initial_model points to a directory: {initial_model_filename}. "
                "Set initial_model to a CSV file or disable use_initial_model."
            )
        if not os.path.exists(initial_model_filename):
            raise FileNotFoundError(
                f"initial_model file not found: {initial_model_filename}. "
                "Update initial_model to a valid CSV path or disable use_initial_model."
            )
        initial_model_data = np.loadtxt(initial_model_filename, delimiter=",", skiprows=1)
    else:
        initial_model_filename = None
        initial_model_data = np.array([])

    

    # Redirect stdout to a file-based logger for the inversion output.
    custom_printer = CustomPrint(output_file_path)
    sys.stdout = custom_printer

    separator = "=" * 150
    section_separator = "-" * 150

    # Runtime globals 
    receiver_grav: Any = None
    receiver_mag: Any = None
    nx = ny = nz = 0
    total_distance_x = total_distance_y = total_distance_z = 0.0
    uncertainties_grav: np.ndarray = np.array([])
    uncertainties_mag: np.ndarray = np.array([])
    alpha0_ratio: Any = np.array([])
    beta: Any = None
    beta_it: Any = None
    targets: Any = None
    mref_in_smooth: Any = None
    update_smallness: Any = None
    save_values_directive: Any = None
    scaling_init: Any = None

    # ====================================================== Density adjustments / constants ===================================================== #
    (
        unit_labels_resolved,
        unit_dens_adj_resolved,
        unit_magsus_resolved,
        unit_dens_disp_resolved,
        unit_magsus_disp_resolved,
        vol_unit_resolved,
    # Normalize and validate unit parameter lists (labels, means, dispersions, weights).
    ) = _normalize_unit_parameters(
        labels=unit_labels,
        dens_adj=unit_dens_adj_list,
        magsus=unit_magsus_list,
        dens_disp=unit_dens_disp_list,
        magsus_disp=unit_magsus_disp_list,
        weights=vol_unit_list,
    )

    unit_dens = 2.67 - unit_dens_adj_resolved

    lower_density = 2.67 - low_dens_adj
    upper_density = 2.67 - up_dens_adj

    # ===================================================== Data loading (topo / grav / mag) ===================================================== #
    def _load_txt(path: str) -> np.ndarray:
        return np.loadtxt(str(path), dtype=np.float32)

    with ThreadPoolExecutor(max_workers=3) as ex:
        fut_topo = ex.submit(_load_txt, topo_filename)
        fut_grav = ex.submit(_load_txt, grav_data_filename)
        fut_mag = ex.submit(_load_txt, mag_data_filename)
        topo_xyz = fut_topo.result()
        dobs_grav = fut_grav.result()
        dobs_mag = fut_mag.result()

    receiver_locations = topo_xyz[:, 0:3]

    dobs_grav = dobs_grav[:, -1]
    dobs_mag = dobs_mag[:, -1]

    # ==================================================== Surveys setup (gravity / magnetic) ==================================================== #
    maximum_anomaly_grav = np.max(np.abs(dobs_grav))
    uncertainties_grav = (uncer_grav / 100.0) * maximum_anomaly_grav
    uncertainties_grav = np.full_like(dobs_grav, uncertainties_grav)
    maximum_anomaly_mag = np.max(np.abs(dobs_mag))
    uncertainties_mag = (uncer_mag / 100.0) * maximum_anomaly_mag
    uncertainties_mag = np.full_like(dobs_mag, uncertainties_mag)

    receiver_grav = pf.gravity.receivers.Point(receiver_locations, components="gz")
    source_field_grav = pf.gravity.sources.SourceField(receiver_list=[receiver_grav])
    survey_grav = pf.gravity.survey.Survey(source_field_grav)

    receiver_mag = pf.magnetics.receivers.Point(receiver_locations, components="tmi")
    source_field_mag = pf.magnetics.sources.UniformBackgroundField(
        receiver_list=[receiver_mag],
        amplitude=strength,
        inclination=inclination,
        declination=declination,
    )
    survey_mag = pf.magnetics.survey.Survey(source_field_mag)

    data_object_grav = data.Data(survey_grav, dobs=dobs_grav, standard_deviation=uncertainties_grav)
    data_object_mag = data.Data(survey_mag, dobs=dobs_mag, standard_deviation=uncertainties_mag)

    # =================================================== Mesh, active cells, and initial model ================================================== #
    # Build the 3D inversion mesh from the topography grid.
    mesh, nx, ny, nz = build_mesh_from_topography(topo_xyz, dx, dy, dz, depth_core)
    actv = active_from_xyz(mesh, topo_xyz, "CC")

    assert getattr(mesh, 'cell_centers', None) is not None, "mesh.cell_centers is required"
    assert getattr(mesh, 'cell_volumes', None) is not None, "mesh.cell_volumes is required"
    cell_centers = cast(np.ndarray, mesh.cell_centers)
    cell_volumes = cast(np.ndarray, mesh.cell_volumes)
    assert actv is not None, "active cell mask (actv) should not be None"

    ndv = np.nan 
    actvMap = maps.InjectActiveCells(mesh, actv, ndv)
    nactv = int(actv.sum()) 

    total_distance_x = nx * dx
    total_distance_y = ny * dy
    total_distance_z = nz * dz

    xyz_active = mesh.gridCC[actv, :] 

    if use_initial_model:
        x_csv, y_csv, z_csv = initial_model_data[:, 0], initial_model_data[:, 1], initial_model_data[:, 2]
        initial_density, initial_susceptibility = initial_model_data[:, 3], initial_model_data[:, 4]
        xyz_csv = np.vstack((x_csv, y_csv, z_csv)).T
        # Interpolate initial model properties onto active mesh cells.
        density_mesh = interpolate_nearest_neighbor(xyz_csv, initial_density, xyz_active)
        susceptibility_mesh = interpolate_nearest_neighbor(xyz_csv, initial_susceptibility, xyz_active)
        m0 = np.r_[density_mesh, susceptibility_mesh]
    else:
        m0 = np.r_[background_dens * np.ones(actvMap.nP), background_susc * np.ones(actvMap.nP)]

    # ==================================================== Physics, misfits, and reference GMM =================================================== #
    nC = nactv
    wires: Any = maps.Wires(("den", nC), ("sus", nC)) 

    idenMap = maps.IdentityMap(nP=nactv)
    gravmap = actvMap * wires.den  
    magmap = actvMap * wires.sus 

    simulation_grav = pf.gravity.simulation.Simulation3DIntegral(
        survey=survey_grav,
        mesh=mesh,
        rhoMap=wires.den,  
        ind_active=actv,
    )

    simulation_mag = pf.magnetics.simulation.Simulation3DIntegral(
        survey=survey_mag,
        mesh=mesh,
        chiMap=wires.sus,  
        ind_active=actv,
    )

    dmis_grav = data_misfit.L2DataMisfit(data=data_object_grav, simulation=simulation_grav) 
    dmis_mag = data_misfit.L2DataMisfit(data=data_object_mag, simulation=simulation_mag)

    dmis = 0.5 * dmis_grav + 0.5 * dmis_mag

    gmmref = utils.WeightedGaussianMixture(
        n_components=len(unit_dens),
        mesh=mesh,
        actv=actv,
        covariance_type="diag",
    )

    gmmref.fit(np.random.randn(nactv, 2))
    gmmref.means_ = np.column_stack((unit_dens, unit_magsus_resolved))

    gmmref.covariances_ = np.column_stack((
        unit_dens_disp_resolved ** 2,
        unit_magsus_disp_resolved ** 2,
    ))

    gmmref.compute_clusters_precisions()
    gmmref.weights_ = vol_unit_resolved

    ax = gmmref.plot_pdf(flag2d=True, plotting_precision=100, padding=2)
    ax[0].set_xlabel("Density contrast [g/cc]")
    ax[2].set_ylabel("magnetic Susceptibility [SI]")
    fig_init = ax[0].figure if hasattr(ax[0], "figure") else plt.gcf()
    file_base = os.path.join(results_path, 'initial_GMM')

    # Save the initial GMM plot in PNG/SVG formats.
    _save_plot_pair(file_base, fig_init)
    plt.close(fig_init)

    # =============================================== Weights, PGI regularization, and alphas/betas ============================================== #
    x_min, x_max = cell_centers[actv][:, 0].min(), cell_centers[actv][:, 0].max()
    y_min, y_max = cell_centers[actv][:, 1].min(), cell_centers[actv][:, 1].max()
    border_distance = 0 * max(dx, dy)

    is_edge_cell = (
        (cell_centers[actv][:, 0] < x_min + border_distance) |  
        (cell_centers[actv][:, 0] > x_max - border_distance) |  
        (cell_centers[actv][:, 1] < y_min + border_distance) |  
        (cell_centers[actv][:, 1] > y_max - border_distance)    
    )

    if simulation_grav.G is not None:
        wr_grav = np.sum(simulation_grav.G**2., axis=0)**0.5 / (cell_volumes[actv]) 
        wr_grav = wr_grav / np.max(wr_grav)
        wr_grav[is_edge_cell] *= 0.5  
    else:
        wr_grav = np.ones(nactv)

    if simulation_mag.G is not None:
        wr_mag = np.sum(simulation_mag.G**2., axis=0)**0.5 / (cell_volumes[actv])  
        wr_mag = wr_mag / np.max(wr_mag)
        wr_mag[is_edge_cell] *= 0.5 
    else:
        wr_mag = np.ones(nactv)

    weights_dict = {
        'cell_x': cell_centers[actv][:, 0],
        'cell_y': cell_centers[actv][:, 1],
        'cell_z': cell_centers[actv][:, 2],
        'wr_grav': wr_grav,
        'wr_mag': wr_mag
    }

    df_weights = pd.DataFrame(weights_dict)
    weights_file_path = os.path.join(results_path, 'weights_gravity_magnetism.csv')
    df_weights.to_csv(weights_file_path, index=False)
                    
    reg = regularization.PGI(
        gmmref=gmmref,
        mesh=mesh,
        wiresmap=wires,
        maplist=[idenMap, idenMap],
        active_cells=actv,
        alpha_pgi=alpha_pgi,
        alpha_x=alpha_x,
        alpha_y=alpha_y,
        alpha_z=alpha_z,
        alpha_xx=alpha_xx,
        alpha_yy=alpha_yy,
        alpha_zz=alpha_zz,
        reference_model=utils.mkvc(
            cast(np.ndarray, gmmref.means_)[
                cast(np.ndarray, gmmref.predict(m0.reshape(actvMap.nP, -1)))
            ]  
        ),
        weights_list=[wr_grav, wr_mag],  
    )

    reg_objfcts = getattr(reg, "objfcts", None)

    def _count_smooth_terms(seq: Sequence[Any], idx: int) -> int:
        if 0 <= idx < len(seq):
            child = seq[idx]
            sub = getattr(child, "objfcts", None)
            if isinstance(sub, Sequence):
                return max(1, len(sub[1:]))
        return 1

    if isinstance(reg_objfcts, Sequence):
        reg_obj_seq = cast(Sequence[Any], reg_objfcts)
        dens_terms = _count_smooth_terms(reg_obj_seq, 1)
        susc_terms = _count_smooth_terms(reg_obj_seq, 2)
    else:
        dens_terms = 1
        susc_terms = 1

    alpha0_ratio = np.r_[
        alpha0_ratio_dens * np.ones(dens_terms),
        alpha0_ratio_susc * np.ones(susc_terms),
    ]

    alphas = directives.AlphasSmoothEstimate_ByEig(alpha0_ratio=alpha0_ratio, verbose=True)

    beta = directives.BetaEstimate_ByEig(beta0_ratio=beta0_ratio)

    beta_it = directives.PGI_BetaAlphaSchedule(
        verbose=True,
        coolingFactor=cooling_factor,
        tolerance=tolerance,
        progress=progress,
    )

    if use_chi_small:
        targets = directives.MultiTargetMisfits(
            verbose=True,
            chiSmall=chi_small,
        )
    else:
        targets = directives.MultiTargetMisfits(
            verbose=True,
        )

    mref_in_smooth = directives.PGI_AddMrefInSmooth(
        wait_till_stable=wait_till_stable,
        verbose=True,
    )

    update_smallness = directives.PGI_UpdateParameters(
        update_gmm=update_gmm,
        kappa=kappa,
    )

    update_Jacobi = directives.UpdatePreconditioner()

    scaling_init = directives.ScalingMultipleDataMisfits_ByEig(chi0_ratio=chi0_ratio)
    scale_schedule = directives.JointScalingSchedule(verbose=True)

    # =============================================== Optimization, inverse problem, and directives ============================================== #
    # Optimization setup
    lowerbound = np.r_[upper_density * np.ones(actvMap.nP), lower_susceptibility * np.ones(actvMap.nP)]
    upperbound = np.r_[lower_density * np.ones(actvMap.nP), upper_susceptibility * np.ones(actvMap.nP)]

    if lowerbound is not None and upperbound is not None:
        opt = optimization.ProjectedGNCG(
            maxIter=max_iter,
            lower=lowerbound,
            upper=upperbound,
            maxIterLS=max_iter_ls,
            maxIterCG=max_iter_cg,
            tolCG=tol_cg,
        )
    else:
        opt = optimization.ProjectedGNCG(
            maxIter=max_iter,
            maxIterLS=max_iter_ls,
            maxIterCG=max_iter_cg,
            tolCG=tol_cg,
        )

    invProb = inverse_problem.BaseInvProblem(dmis, reg, opt)

    directiveList = [
        alphas,
        scaling_init,
        beta,
        update_smallness,
        targets,
        scale_schedule,
        beta_it,
        mref_in_smooth,
        update_Jacobi,
    ]

    class SaveValuesDirective(directives.InversionDirective):
        def initialize(self):
            self.phi_d_list = []
            self.phi_m_list = []
            self.beta_list = []
            
            self.phi_d_gravity_list = []
            self.phi_d_magnetic_list = []
            self.phi_m_smallness_list = []
            
        def endIter(self):
            self.phi_d_list.append(self.invProb.phi_d)
            self.phi_m_list.append(self.invProb.phi_m)
            self.beta_list.append(self.invProb.beta)
            
            if hasattr(self.invProb, 'dmisfit') and hasattr(self.invProb.dmisfit, 'objfcts'):
                if self.invProb.dmisfit.objfcts is not None and len(self.invProb.dmisfit.objfcts) > 1:
                    phi_d_gravity = self.invProb.dmisfit.objfcts[0](self.invProb.model)  
                    phi_d_magnetic = self.invProb.dmisfit.objfcts[1](self.invProb.model)  
                else:
                    phi_d_gravity = 0.0
                    phi_d_magnetic = 0.0
            else:
                phi_d_gravity = 0.0
                phi_d_magnetic = 0.0
            
            phi_m_smallness = self.invProb.reg.objfcts[0](self.invProb.model)  
            
            self.phi_d_gravity_list.append(phi_d_gravity)
            self.phi_d_magnetic_list.append(phi_d_magnetic)
            self.phi_m_smallness_list.append(phi_m_smallness)

    save_values_directive = SaveValuesDirective()
    directiveList.append(save_values_directive)

    # =============================================================== Run inversion ============================================================== #
    # ==================================================================================================================================================== 
    print("\n" + separator)
    print("1. INVERSION CONFIGURATION SUMMARY".center(150))
    print(separator)
    # ==================================================================================================================================================== 
    
    # Print a full inversion configuration summary for the report.
    inversion_summary = build_inversion_config_summary(locals())
    print_inversion_config_summary(inversion_summary)

    inv = inversion.BaseInversion(
        invProb,
        directiveList=directiveList
    )
    pgi_model_no_info = inv.run(m0)

    # Capture optimizer iteration counts for reporting.
    opt_iter = getattr(opt, "iter", None)
    if opt_iter is None:
        opt_iter = getattr(opt, "iterNum", None)
    opt_max_iter = getattr(opt, "maxIter", None)

    # Plot the evolution of data/regularization misfits during inversion.
    plot_misfit_evolution(save_values_directive, results_path=results_path)

    density_model_no_info = gravmap * pgi_model_no_info
    magsus_model_no_info = magmap * pgi_model_no_info

    learned_gmm = None
    quasi_geology_model_no_info = np.zeros(mesh.nC)
    reg_objfcts_post = getattr(reg, "objfcts", None)
    if isinstance(reg_objfcts_post, Sequence) and len(reg_objfcts_post) > 0:
        reg_first = reg_objfcts_post[0]
        gmm_candidate = getattr(reg_first, "gmm", None)
        if gmm_candidate is not None:
            learned_gmm = gmm_candidate
        compute_quasi = getattr(reg_first, "compute_quasi_geology_model", None)
        if callable(compute_quasi):
            quasi_geology_model_no_info = actvMap * compute_quasi() 

    # Plot the learned GMM against the recovered model.
    plot_learned_gmm(
        learned_gmm=learned_gmm,
        density_model_no_info=density_model_no_info,
        magsus_model_no_info=magsus_model_no_info,
        ind_active=actv,
        results_path=results_path,
        true_means=gmmref.means_,
        model_colors=quasi_geology_model_no_info,
    )

    density_model_no_info = density_model_no_info.astype(float)
    magsus_model_no_info = magsus_model_no_info.astype(float)
    quasi_geology_model_no_info = np.asarray(quasi_geology_model_no_info)

    density_model_no_info = density_model_no_info[actv]
    magsus_model_no_info = magsus_model_no_info[actv]
    quasi_geology_model_no_info = quasi_geology_model_no_info[actv]

    # ======================================================= Central crop and model export ====================================================== #
    x_min, x_max = xyz_active[:, 0].min(), xyz_active[:, 0].max()
    y_min, y_max = xyz_active[:, 1].min(), xyz_active[:, 1].max()

    expansion_x = 0.0
    expansion_y = 0.0

    if expansion_type == "rectangular":
        original_distance_x = total_distance_x / (1 + expansion_percentage / 100)
        original_distance_y = total_distance_y / (1 + expansion_percentage / 100)
        expansion_x = (total_distance_x - original_distance_x) / 2
        expansion_y = (total_distance_y - original_distance_y) / 2

    elif expansion_type == "square":
        l_new = total_distance_x

        if dominant_side == "x":
            original_x = l_new / (1 + expansion_percentage / 100)
            expansion_x = (l_new - original_x) / 2
            expansion_y = (l_new - original_y) / 2

        elif dominant_side == "y":
            original_y = l_new / (1 + expansion_percentage / 100)
            expansion_y = (l_new - original_y) / 2
            expansion_x = (l_new - original_x) / 2

    x_min_central = x_min + expansion_x
    x_max_central = x_max - expansion_x
    y_min_central = y_min + expansion_y
    y_max_central = y_max - expansion_y

    central_mask = (
        (xyz_active[:, 0] >= x_min_central) & (xyz_active[:, 0] <= x_max_central + dx) &
        (xyz_active[:, 1] >= y_min_central) & (xyz_active[:, 1] <= y_max_central + dy)
    )

    xyz_active = xyz_active[central_mask]
    density_model_no_info = density_model_no_info[central_mask]
    magsus_model_no_info = magsus_model_no_info[central_mask]
    quasi_geology_model_no_info = quasi_geology_model_no_info[central_mask]

    export_data_density_complete = np.hstack((xyz_active, density_model_no_info.reshape(-1, 1)))
    export_data_magsus_complete = np.hstack((xyz_active, magsus_model_no_info.reshape(-1, 1)))
    export_geology_complete = np.hstack((xyz_active, quasi_geology_model_no_info.reshape(-1, 1)))

    density_filepath = os.path.join(results_path, "Density_complete_model.csv")
    np.savetxt(density_filepath, export_data_density_complete, delimiter=',', header="X,Y,Z,Density", fmt="%.6f")

    magsus_filepath = os.path.join(results_path, "Magsus_complete_model.csv")
    np.savetxt(magsus_filepath, export_data_magsus_complete, delimiter=',', header="X,Y,Z,Magsus", fmt="%.6f")

    geology_filepath = os.path.join(results_path, "Geology_complete_model.csv")
    np.savetxt(geology_filepath, export_geology_complete, delimiter=',', header="X,Y,Z,Geology", fmt="%.6f")

    mask_geology = (quasi_geology_model_no_info == serpentinite_label)
    density_model_no_info[~mask_geology] = np.nan
    magsus_model_no_info[~mask_geology] = np.nan

    density_export_cells = density_model_no_info
    magsus_export_cells = magsus_model_no_info

    export_data_density = np.hstack((xyz_active, density_export_cells.reshape(-1, 1)))
    export_data_magsus = np.hstack((xyz_active, magsus_export_cells.reshape(-1, 1)))
    density_filepath = os.path.join(results_path, "Density_Serpentinite.csv")
    np.savetxt(density_filepath, export_data_density, delimiter=',', header="X,Y,Z,Density", fmt="%.6f")
    magsus_filepath = os.path.join(results_path, "Magsus_Serpentinite.csv")
    np.savetxt(magsus_filepath, export_data_magsus, delimiter=',', header="X,Y,Z,Magsus", fmt="%.6f")

    if run_h2_quantification:
        density_file = density_filepath
        magsus_file = magsus_filepath

    # ==================================================== Fields, residuals, and final plots ==================================================== #
    fields_grav = simulation_grav.fields(pgi_model_no_info)
    d_pred_grav = simulation_grav.dpred(pgi_model_no_info, f=fields_grav)
    fields_mag = simulation_mag.fields(pgi_model_no_info)
    d_pred_mag = simulation_mag.dpred(pgi_model_no_info, f=fields_mag)

    if save_pred_residual_npy:
        np.save(os.path.join(results_path, "pred_grav"), d_pred_grav)
        np.save(os.path.join(results_path, "pred_mag"), d_pred_mag)

    residual_grav = dobs_grav - d_pred_grav
    residual_mag = dobs_mag - d_pred_mag

    if save_pred_residual_npy:
        np.save(os.path.join(results_path, "residual_grav"), residual_grav)
        np.save(os.path.join(results_path, "residual_mag"), residual_mag)

    # Plot observed vs predicted gravity and residuals.
    plot_gravity_observed_predicted_residual(
        receiver_locations,
        dobs_grav,
        d_pred_grav,
        residual_grav,
        results_path=results_path,
    )
    # Plot observed vs predicted magnetic data and residuals.
    plot_magnetic_observed_predicted_residual(
        receiver_locations,
        dobs_mag,
        d_pred_mag,
        residual_mag,
        results_path=results_path,
    )
    # Plot residual histograms for gravity and magnetic data.
    plot_residual_histograms(residual_grav, residual_mag, results_path=results_path)

    # ============================================================= Summary and close ============================================================ #
    end_time = time.time()
    execution_time = end_time - start_time
    hours = execution_time // 3600
    minutes = (execution_time % 3600) // 60
    seconds = execution_time % 60
    formatted_time = f"{int(hours)}:{int(minutes)}:{int(seconds)}"

    # Print the inversion results summary (timing, outputs, iterations).
    results_summary = build_inversion_results_summary(locals())
    print_inversion_results_summary(results_summary)

    custom_printer.close()
else:
    print("Skipping Routine 1 (inversion).")


######################################################################################################################################################
################################################################# Quantification #####################################################################
######################################################################################################################################################

QUANT_PARAM_NAMES = [
    "base_dir", "topo_file", "db_dir", "temperature_file", "density_file", "magsus_file",
    "seed", "use_global_seed", "dist_x", "dist_y", "dist_z", "density_serpentinite",
    "porosity_front", "int_fracture_spacing", "permeability_fractures", "waterrockratio",
    "flow_target", "density_litho", "gravity", "lithology_code", "years", "molar_mass_h2",
    "molar_mass_h2o", "v_ref_synthetic", "t_ref_range", "save_svg", "run_montecarlo_fault", "fault_mc_n_iter",
    "flow_target_fracture_config", "mc_no_saturation_config", "mc_saturation_config",
    "run_univariate_analysis_no_sat", "run_univariate_analysis_sat", "univariate_analysis_config",
    "run_mc_convergence_sweep_flag", "mc_convergence_sweep_config", "run_analyze_limiting_factors", "flow_target_log_min",
    "flow_target_log_max", "flow_target_n_samples", "mc_flow_target_config", "lithologies_dict",
    "serpentinization_data", "serp_corr_percentage",
]


# Shared context for Monte Carlo worker processes (avoids re-pickling heavy arguments per task)
_MC_WORKER_CONTEXT: Optional[Dict[str, Any]] = None


if run_h2_quantification:
    # Print the report header once for the quantification run.
    _print_header_once()

    # Validate required quantification parameters are present and non-empty.
    quant_inputs = {name: getattr(cfg, name, None) for name in QUANT_PARAM_NAMES}
    quant_inputs.update(
        {
            "run_mc_convergence_sweep_flag": run_mc_convergence_sweep_flag,
            "mc_convergence_sweep_config": mc_convergence_sweep_config,
            "run_analyze_limiting_factors": run_analyze_limiting_factors,
            "flow_target_log_min": flow_target_log_min,
            "flow_target_log_max": flow_target_log_max,
            "flow_target_n_samples": flow_target_n_samples,
            "mc_flow_target_config": mc_flow_target_config,
            "lithologies_dict": lithologies_dict,
            "serpentinization_data": serpentinization_data,
            "serp_corr_percentage": serp_corr_percentage,
        }
    )
    missing_quant = [name for name in QUANT_PARAM_NAMES if _is_missing(quant_inputs.get(name))]

    if not run_montecarlo_fault:
        for name in ["fault_mc_n_iter", "flow_target_fracture_config"]:
            if name in missing_quant:
                missing_quant.remove(name)

    if not run_univariate_analysis_no_sat and not run_univariate_analysis_sat:
        if "univariate_analysis_config" in missing_quant:
            missing_quant.remove("univariate_analysis_config")

    if not run_mc_convergence_sweep_flag and "mc_convergence_sweep_config" in missing_quant:
        missing_quant.remove("mc_convergence_sweep_config")

    if not run_analyze_limiting_factors:
        for name in ["flow_target_log_min", "flow_target_log_max", "flow_target_n_samples", "mc_flow_target_config"]:
            if name in missing_quant:
                missing_quant.remove(name)

    if missing_quant:

        # Report missing parameters with context-specific messaging.
        _report_missing_params(missing_quant, "H2 quantification")
        raise ValueError("Missing required H2 quantification parameters.")
    # ============================================================== Begin of time =============================================================== #
    start_time = time.time()
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    dir_path = base_dir + os.path.sep

    # Define the folder name with date and time
    folder_name = f"Results_Generation{current_time}"
    results_path = os.path.join(dir_path, folder_name)

    # Create the folder if it doesn't exist
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    separator = "=" * 150
    section_separator = "-" * 150

    # ============================================================== Paths to data =============================================================== #

    # Resolve input data paths relative to the working directory.
    topo_filename = _resolve_base_dir_path(dir_path, topo_file)
    assert topo_filename is not None, "Resolved topo_filename is required"

    # Resolve the H2 database path.
    dsn_db = _resolve_base_dir_path(dir_path, db_dir)
    assert dsn_db is not None, "Resolved dsn_db is required"

    # Resolve the temperature model path.
    temperature_filename = _resolve_base_dir_path(dir_path, temperature_file)
    assert temperature_filename is not None, "Resolved temperature_filename is required"

    # Resolve density and magnetic susceptibility model paths.
    density_path = _resolve_base_dir_path(dir_path, density_file)
    magsus_path = _resolve_base_dir_path(dir_path, magsus_file)
    assert density_path is not None, "Resolved density_path is required"
    assert magsus_path is not None, "Resolved magsus_path is required"

    # Load pre-filtered CSV data (pre-calculated using the inversion code)
    export_data_density = np.loadtxt(
        density_path,
        delimiter=',', skiprows=1
    )

    export_data_magsus = np.loadtxt(
        magsus_path,
        delimiter=',', skiprows=1
    )

    # Define the path for the output file within the results folder
    output_file_path = os.path.join(results_path, "output.txt")

    # Redirect all prints to the custom output
    custom_printer = CustomPrint(output_file_path)
    sys.stdout = custom_printer

    if use_global_seed:
        print(f"\n[INFO] Random generators seeded with seed={seed}.")
        np.random.seed(seed)
        random.seed(seed)
    else:
        print("[INFO] Random generators running without a fixed seed.")

    # Ensure the configured number of cores never exceeds the machine capacity.
    n_cores = _clamp_worker_count(n_cores)

    # Common 16-bin temperature ranges used across reporting and simulations (°C)
    TEMP_BINS: Tuple[str, ...] = (
        "100_125", "125_150", "150_175", "175_200",
        "200_225", "225_250", "250_275", "275_300",
        "300_325", "325_350", "350_375", "375_400",
        "400_425", "425_450", "450_475", "475_500",
    )

    # ====================================================================================================================================================
    print("\n" + separator)
    print("2. MONTECARLO FOR FAULT FLOW".center(150))
    print(separator)
    # ====================================================================================================================================================

    # - Execute the fracture Monte Carlo to estimate permeability and cumulative water delivery.
    if run_montecarlo_fault:
        # Run Monte Carlo sampling to estimate fracture permeability and flow success.
        k_samples, success = run_fracture_monte_carlo_simulation(
            flow_target_fracture_config,
            flow_target=flow_target,
            n_samples=fault_mc_n_iter,
            make_plot=True
        )
        # Call print and plot helpers consecutively
        results = {
            "k": k_samples,
            "success": success,
            "success_rate": np.mean(success) * 100 if len(success) > 0 else 0,
            "flow_target": flow_target,
            "Q_m3s": None  # required by helpers
        }
        # Print the fault-flow Monte Carlo summary.
        print_fracture_monte_carlo_report(results)
        # Plot the histogram of sampled fracture permeability values.
        plot_fracture_mc_histogram(results)
    else:
        print("Skipped fault-flow Monte Carlo (run_montecarlo_fault=False).")

    # ====================================================================================================================================================
    print("\n" + separator)
    print("3. CONFIGURATION SUMMARY".center(150))
    print(separator)
    # ====================================================================================================================================================

    porosity_front = porosity_front / 100.0

    # - Load gridded topography and define the surface receiver locations used downstream.
    topo_xyz = np.loadtxt(str(topo_filename))
    receiver_locations = topo_xyz[:, 0:3]
    x = receiver_locations[:, 0]
    y = receiver_locations[:, 1]

    # - Build the 3D mesh from topography and gather basic mesh stats.
    # Build the 3D mesh for quantification from topography.
    mesh, nx, ny, nz = build_mesh_from_topography(topo_xyz, dx, dy, dz, depth_core)

    # Active cell coordinates and filtered property models
    xyz_active = export_data_density[:, :3]
    density_model_no_info = export_data_density[:, 3]
    magsus_model_no_info = export_data_magsus[:, 3]
    density_export_cells = density_model_no_info
    magsus_export_cells = magsus_model_no_info

    # Load_ external temperature CSV, interpolate onto active cells.
    temperature_csv = np.loadtxt(temperature_filename, delimiter=',', skiprows=1)
    x_csv_temperature = temperature_csv[:, 0]
    y_csv_temperature = temperature_csv[:, 1]
    z_csv_temperature = temperature_csv[:, 2]
    temperature_csv = temperature_csv[:, 3]
    xyz_csv_temperature = np.vstack((x_csv_temperature, y_csv_temperature, z_csv_temperature)).T
    temperature_mesh = griddata(xyz_csv_temperature, temperature_csv, xyz_active, method='nearest')
    temperature_ranges = {tr: None for tr in TEMP_BINS}

        
    # Estimate depths associated with temperature range extremes.
    depths_for_temp_extremes = compute_temp_range_depths(temperature_mesh, xyz_active, temperature_ranges)
        
    # Create a mask to identify non-NaN values in the density data (i.e., valid serpentinite cells)
    mask_not_nan = ~np.isnan(density_export_cells)

    # Compute rock volumes and temperature masks for serpentinite cells.
    volume_density_magsus, volume_at_temperature, volume_at_temperature_total_100_500, combined_masks, combined_mask_total_100_500 = compute_rock_volumes_and_masks(
        temperature_mesh,
        dx,
        dy,
        dz,
        mask_not_nan,
    )

    #  Persist binary masks for all serpentinite cells and for the 100-500 °C subset.
    cell_indicator_gm = np.where(mask_not_nan, 1, 0)
    export_data_indicator_gm = np.hstack((xyz_active, cell_indicator_gm.reshape(-1, 1)))
    indicator_filepath_gm = os.path.join(results_path, "Indicator_serpentinite.csv")
    np.savetxt(indicator_filepath_gm, export_data_indicator_gm, delimiter=',', header="X (m),Y (m),Z (m),Indicator", fmt="%d")

    # Create an indicator array for serpentinite rocks within the 100-500°C temperature range and export to CSV
    cell_indicator_um_100_500 = np.where(combined_mask_total_100_500, 1, 0)
    export_data_indicator_um_100_500 = np.hstack((xyz_active, cell_indicator_um_100_500.reshape(-1, 1)))
    indicator_filepath_um_100_500 = os.path.join(results_path, "Indicator_at_100_500_serpentinite.csv")
    np.savetxt(indicator_filepath_um_100_500, export_data_indicator_um_100_500, delimiter=',', header="X (m),Y (m),Z (m),Indicator", fmt="%d")

    #  Interpolates lab-derived curves to observed density/magnetic data and plots diagnostic heat maps.
    x_points = serpentinization_data['x_points']
    y_points = serpentinization_data['y_points']
    values = serpentinization_data['values']

    # Compute serpentinization degree from observed density/magnetic susceptibility and lab curves.
    mode_density, mode_magsus, serpentinization_degree = compute_serpentinization_degree(density_export_cells, 
                                                                                    magsus_export_cells, 
                                                                                    mask_not_nan, 
                                                                                    x_points, y_points, values) 
                                                                                                      
    # Plot a heat map showing the degree of serpentinization
    plot_serpentinization_heatmap(
        density_export_cells,
        magsus_export_cells,
        mask_not_nan,
        mode_density,
        mode_magsus,
        x_points,
        y_points,
        values,
        results_path=results_path,
        save_svg=save_svg,
    )

    # Print a compact summary of the inversion/mesh configuration used for quantification.
    inversion_mesh_config = build_inversion_mesh_config(locals())
    print_inversion_config(inversion_mesh_config)

    # Load the thermodynamic H2 production database for the target lithologies.
    thermo_data = load_h2_production_database(dsn_db, rock_codes=["HZ1", "LH1"])

    # ====================================================================================================================================================
    print("\n" + separator)
    print("4. SUMMARY OF VOLUMES AND DEGREE OF SERPENTINIZATION".center(150))
    print(separator + "\n")
    # ====================================================================================================================================================

    # Print the serpentinization volume/degree summary tables.
    serpentinization_corrections, total_volume = print_summary_serpentinization_section(
        build_serpentinization_summary_params(locals())
    )
    # Compute mean lithostatic pressures and depth stats per temperature range.
    xyz_mesh_temperature = xyz_active
    mean_pressure_ranges, mean_depths = compute_mean_lithostatic_pressure_by_range(
        build_mean_lithostatic_pressure_params(locals())
    )

    # Build a lookup of H2 production rates by temperature/pressure range.
    thermo_lookup = build_thermo_lookup_by_range(build_thermo_lookup_params(locals()))

    # ====================================================================================================================================================
    print("\n" + separator)
    print(f"5. ROCK DATABASE: Lithology: {lithology_code} | W/R Ratio = {waterrockratio}".center(150))
    print(separator)
    # ====================================================================================================================================================

    # Print production-rate tables by temperature range.
    print_production_rate_volumetric_report(build_production_rate_report_params(locals()))
    # Build a dict of production rates for downstream calculations.
    production_rate_volumetric = build_production_rate_volumetric_dict(
        build_production_rate_volumetric_params(locals())
    )


    # Derive serpentinization-front velocities per temperature range.
    serpentinization_front_velocities = compute_serpentinization_front_velocities(
        build_serpentinization_front_velocity_params(locals())
    )

    # ====================================================================================================================================================
    print("\n" + separator)
    print(f"{'6. PARAMETERS FOR WATER FLOW & SERPENTINIZATION':^150}")
    print(separator + "\n")
    # ====================================================================================================================================================
             
    # Print the water-flow and serpentinization parameter summary.
    water_flow_params = WaterFlowSerpentinizationParams.from_cfg(
        cfg,
        porosity_front=porosity_front,
    )
    print_parameters_water_flow_serpentinization(water_flow_params)

    # ==================================================================================================================================================== 
    print()
    print("\n" + separator)
    print(f"{'7. H2 ESTIMATION NO SATURATION ':^130}")
    print(separator + "\n")
    # ====================================================================================================================================================
    # Full no-saturation volumetric pipeline (surface area -> MC -> summary/CSV)

    temp_bins = TEMP_BINS
    run_prints = True
    surface_area_per_km3 = runtime_state.get("surface_area_per_km3")
    if surface_area_per_km3 is None:
        surface_area_per_km3 = _compute_surface_area_per_km3(dist_x, dist_y, dist_z)
        runtime_state["surface_area_per_km3"] = surface_area_per_km3

    no_sat_params = build_no_saturation_workflow_params(cfg, locals())
    (   results_temp_volumetric, kg_rocks_dict, std_results_temp_volumetric, df_no_saturation,
        total_tons_no_sat, no_sat_csv_path, total_kg_rocks,
    # Run the no-saturation Monte Carlo workflow and export results.
    ) = run_no_saturation_workflow(no_sat_params)

    # ====================================================================================================================================================
    print(separator)
    print(f"{'8. H2 ESTIMATION WITH SATURATION ':^130}")
    print(separator)
    # ====================================================================================================================================================
    # Full saturation volumetric pipeline (surface area -> MC -> summary/CSV)

    df_saturation_table = df_no_saturation
    # Run the saturation-limited Monte Carlo workflow and export results.
    sat_params = build_saturation_workflow_params(cfg, locals())
    stats_mc_saturation, total_tons_sat, std_total_mc, mean_efficiency = run_saturation_workflow(
        sat_params
    )


    temp_ranges_plot = list(stats_mc_saturation.index) if stats_mc_saturation is not None and not stats_mc_saturation.empty else list(mean_pressure_ranges.keys()) if mean_pressure_ranges else []
    canonical_ranges = sorted(mean_pressure_ranges.keys(), key=lambda x: float(x.split('_')[0])) if mean_pressure_ranges else []
    temp_mid_lookup = {}
    for tr in canonical_ranges:
        try:
            lo, hi = tr.split("_")
            temp_mid_lookup[tr] = (float(lo) + float(hi)) / 2.0
        except Exception:
            temp_mid_lookup[tr] = None

    # Build timeseries for H2O absorbed and inflow components.
    h2o_incorporated = build_series(
        "H2O absorbed [kg/day]",
        stats_mc_saturation=stats_mc_saturation,
        temp_ranges_plot=temp_ranges_plot,
        canonical_ranges=canonical_ranges,
        temp_mid_lookup=temp_mid_lookup,
        fill_missing=False,
    )
    inflow_diff = build_series(
        "daily_diffused_H2O [kg/day]",
        stats_mc_saturation=stats_mc_saturation,
        temp_ranges_plot=temp_ranges_plot,
        canonical_ranges=canonical_ranges,
        temp_mid_lookup=temp_mid_lookup,
        fill_missing=True,
    )
    inflow_frac = build_series(
        "daily_fractured_H2O [kg/day]",
        stats_mc_saturation=stats_mc_saturation,
        temp_ranges_plot=temp_ranges_plot,
        canonical_ranges=canonical_ranges,
        temp_mid_lookup=temp_mid_lookup,
        fill_missing=True,
    )

    # Prepare inputs for the multi-panel lab comparison plots.
    prepped = prepare_lab_plot_inputs(
        production_rate_volumetric=production_rate_volumetric,
        serpentinization_front_velocities=serpentinization_front_velocities,
        serp_corr_percentage=serp_corr_percentage,
        mean_pressure_ranges=mean_pressure_ranges,
        mean_depths=mean_depths,
        saturation_df=df_no_saturation,
        h2o_incorporated=h2o_incorporated,
        inflow_diff=inflow_diff,
        inflow_frac=inflow_frac,
        volume_at_temperature=volume_at_temperature,
    )

    # Plot the lab/field comparison panels.
    plot_lab_data_panels(prepped, results_path=results_path)

    # Save saturation summary tables to CSV.
    save_saturation_csv(stats_mc_saturation, mean_pressure_ranges, results_path)


    # ====================================================================================================================================================
    # POST PROCCESSING AND SENSITIVITY ANALYSIS
    # ==================================================================================================================================================== 
    mc_common_kwargs = build_mc_common_kwargs(locals(), cfg=cfg)

    # Ensure we pass a valid surface area per km³ (small fractures) to the no-sat MC.
    # If the global was not set (e.g., running univariate directly), recompute it here.
    _surface_area_no_sat = runtime_state.get("surface_area_per_km3")
    if _surface_area_no_sat is None:
        _surface_area_no_sat = _compute_surface_area_per_km3(dist_x, dist_y, dist_z)
        runtime_state["surface_area_per_km3"] = _surface_area_no_sat

    mc_no_sat_kwargs = build_mc_no_sat_kwargs(
        locals(),
        surface_area_per_km3=_surface_area_no_sat,
        cfg=cfg,
    )

    if run_univariate_analysis_sat:
        # Run univariate sensitivity sweeps for the saturation MC.
        run_saturation_univariate_sweep(
            mc_kwargs=mc_common_kwargs,
            base_config=mc_saturation_config,
            uni_config=univariate_analysis_config,
            results_dir=results_path,
            n_cores=n_cores,
            seed=seed,
            mc_saturation_config=mc_saturation_config,
        )
    else:
        print("Skipped univariate sensitivity (run_univariate_analysis_sat=False)")


    if run_univariate_analysis_no_sat:
        # Run univariate sensitivity sweeps for the no-saturation MC.
        run_no_saturation_univariate_sweep(
            mc_kwargs=mc_no_sat_kwargs,
            base_config=mc_no_saturation_config,
            uni_config=univariate_analysis_config,
            results_dir=results_path,
            n_cores=n_cores,
            seed=seed,
        )
    else:
        print("Skipped univariate sensitivity (no-saturation) (run_univariate_analysis_no_sat=False)")


    if run_mc_convergence_sweep_flag:
        # ====================================================================================================================================================
        print("\n" + separator)
        print(f"{'CONVERGENCE SWEEPS':^150}")
        print(separator)
        # ====================================================================================================================================================

    if run_mc_convergence_sweep_flag:
        base_df_mc = stats_mc_saturation.attrs.get("df_all") if stats_mc_saturation is not None else None
        # Evaluate convergence vs iteration count for the saturation MC.
        run_mc_convergence_sweep(
            mc_convergence_sweep_config,
            base_df=base_df_mc,
            mc_kwargs=mc_common_kwargs,
            results_dir=results_path
        )
    else:
        print("")
    print("Skipped MC convergence sweep (run_mc_convergence_sweep_flag=False).")



    if run_analyze_limiting_factors:
        # ====================================================================================================================================================
        print("\n" + separator)
        print(f"{'FLOW TARGETS':^150}")
        print(separator)
        # ====================================================================================================================================================

        # Analyze limiting factors across a range of flow-target values.
        flow_target_params = build_flow_target_limiting_factors_params(locals(), cfg=cfg)
        analyze_limiting_factors_by_flow_target(flow_target_params)
    else:
        print("Skipped flow-target Monte Carlo analysis (run_analyze_limiting_factors=False)")

    # ==================================================================================================================================================== 
    print(separator)
    print(f"{'9. EXECUTION TIME':^130}")
    print(separator + "\n")
    # ==================================================================================================================================================== 

    # Execution Time Summary and Warnings
    end_time = time.time()
    execution_time = end_time - start_time

    hours = execution_time // 3600
    minutes = (execution_time % 3600) // 60
    seconds = execution_time % 60
    formatted_time = f"{int(hours)}:{int(minutes)}:{int(seconds)}"
    print(f"The required time for execution was: {formatted_time}\n")
    print(section_separator + "\n")

    custom_printer.close()
else:
    print("Skipping Routine 2 (H2 quantification).")
