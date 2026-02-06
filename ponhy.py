import csv
import math
import os
import random
import sys
import time
import warnings
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
    _compute_mc_chunk_size,
    _coerce_config_value,
    _apply_text_config,
    _format_value_for_filename,
    _get_progress_bar,
    _is_missing,
    _load_text_config,
    _moving_average,
    _normalize_unit_parameters,
    _parse_value,
    _print_header,
    _print_header_once,
    _resolve_base_dir_path,
    _remove_inline_comment,
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
    run_dt_convergence_sweep,
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

RUN_INVERSION = True  # Set True to run the inversion routine.
RUN_H2_QUANTIFICATION = True  # Set True to run the H2 quantification routine.

# ================================================== Paths (IQ) ================================================== #
BASE_DIR = "/home/christiansen/Projects/Python_Git/PoNHy"  # Change this to your base directory [IQ]

# Topography, gravity and magnetic data (gravity, magnetic and topography points must be the same)
# Shared data paths (used by both routines)
TOPO_FILE = "/Data/Ext_Topo.txt"  # Path to topography file [IQ]

# Inversion-only paths (used when RUN_INVERSION=True)
GRAV_FILE = "/Data/Ext_Grav.txt"  # Path to gravity data file [I]
MAG_FILE = "/Data/Ext_Mag_down.txt"  # Path to magnetic data file [I]
INITIAL_MODEL = "/Data/Updated_Geology_Pyrenees.csv"  # Path to initial model file [I]

# Initial model. If not set it to: USE_INITIAL_MODEL = False
USE_INITIAL_MODEL = True  # [I]

# Define density contrast values for each unit in g/cc and SI.
BACKGROUND_DENS, BACKGROUND_SUSC = -1e-4, 1e-4  # Used when USE_INITIAL_MODEL = False [I]

# H2 quantification paths (used when RUN_H2_QUANTIFICATION=True)
# Database for H2 production (pre-calculated in Theriak-Domino and converted to zarr format)
DB_DIR = "/Data/DB0"  # Path to H2 database directory [Q]
TEMPERATURE_FILE = "/Data/Temperature_model_new.csv"  # Path to temperature file [Q]
DENSITY_FILE = "/Data/Density_Serpentinite.csv"  # Path to density of serpentinites file [Q]
MAGSUS_FILE = "/Data/Magsus_Serpentinite.csv"  # Path to magnetic susceptibility of serpentinites file [Q]

SAVE_SVG = False  # Save SVG copies of plots alongside PNG outputs [IQ]

# ================================================== Mesh parameters (IQ) ================================================== #
# Mesh parameters. 
DX = 750  # Units of m [IQ]
DY = 750  # Units of m [IQ]
DZ = 500  # Units of m [IQ]
DEPTH_CORE = 20000  # Units of m [IQ]

# There are two expanding methods one can use: rectangular or square. 
EXPANSION_PERCENTAGE = 10  # e.g. The grid was expanded by 10% [I]
EXPANSION_TYPE = "rectangular"  # "rectangular" or "square" [I]

# If expansion was square, dominant side of the original grid, if not ignore
DOMINANT_SIDE = "x"  # or "y", as appropriate [I]
# Ignore the other side
ORIGINAL_Y = 0  # Units of m [I]
ORIGINAL_X = 0  # Units of m [I]

# ================================================== Geophysical parameters (I) ================================================== #
# Uncertainties in gravity and mag data in % of the maximum anomaly
UNCER_GRAV = 2.0  # Units of % [I]
UNCER_MAG = 2.0  # Units of % [I]

# Define the inducing field H0 = (intensity [nT], inclination [deg], declination [deg])
INCLINATION = 90  # Units of deg [I]
DECLINATION = 0  # Units of deg [I]
STRENGTH = 45000  # Units of nT # IGRF calculator year 2014 [I]

# ================================================== Petrophysical parameters (I) ================================================== #
# Petrophysical parameters
# Define all units using lists (add/remove units here)
UNIT_LABELS = ["Crust", "Serpentinite", "Mantle"]  # Names used in reporting [I]
UNIT_DENS_ADJ_LIST = [2.60, 2.85, 3.05]  # Units of g/cm³  [I]
UNIT_MAGSUS_LIST = [0.001, 0.030, 0.0001]  # Units of SI  [I]

# Standar deviation of the petrophysical parameters (for the covariance matrix)
UNIT_DENS_DISP_LIST = [0.07, 0.04, 0.04]  # Units of g/cm³ [I]
UNIT_MAGSUS_DISP_LIST = [0.003, 0.005, 0.003]  # Units of SI [I]

# Define the approximate volumes of the units (must sum 1)
VOL_UNIT_LIST = [0.70, 0.10, 0.20]  # Must sum 1 [I]

# Label used to identify serpentinite cells in the quasi-geology model
SERPENTINITE_LABEL = 2  # Default PGI cluster index for Unit 2 (Serpentinite) [I]

# Define the approximate density and susceptibility ranges for the inversion 
LOW_DENS_ADJ = 2.2  # Units of g/cm³ [I]
UP_DENS_ADJ = 3.5  # Units of g/cm³ [I]
LOWER_SUSCEPTIBILITY = 0  # Units of SI [I]
UPPER_SUSCEPTIBILITY = 0.1  # Units of SI [I]

# ================================================== Inversion parameters (I) ================================================== #
# Define the parameters for the inversion
MAX_ITER = 25  # Max non-linear (outer) iterations [I]
MAX_ITER_LS = 20  # Max line-search iterations per outer iteration [I]
MAX_ITER_CG = 100  # Max conjugate-gradient iterations per outer iteration [I]
TOL_CG = 1e-4  # CG tolerance (stopping criterion) [I]
SAVE_PRED_RESIDUAL_NPY = False  # Save predicted/residual arrays as .npy files [I]


# Regularization: alpha_x/y/z/(xx/yy/zz) -> smoothness; alpha_pgi -> PGI (petrophysical clustering).
ALPHA_PGI = 1.0  # [I]
ALPHA_X = 1.0  # [I]
ALPHA_Y = 1.0  # [I]
ALPHA_Z = 1.0  # [I]
ALPHA_XX = 0.0  # [I]
ALPHA_YY = 0.0  # [I]
ALPHA_ZZ = 0.0  # [I]

# AlphasSmoothEstimate_ByEig: alpha0_ratio scales smoothness alphas vs smallness (eig-based).
ALPHA0_RATIO_DENS = 5e-3  # [I]
ALPHA0_RATIO_SUSC = 5e-1  # [I]

# BetaEstimate_ByEig: beta0_ratio scales beta_0 (bigger -> more regularization; smaller -> more data fit).
BETA0_RATIO = 1e-3  # [I]

# PGI_BetaAlphaSchedule: beta /= coolingFactor; tolerance/progress control cooling decisions vs targets.
COOLING_FACTOR = 1.25  # [I]
TOLERANCE = 0.5  # [I]
PROGRESS = 1.0  # [I]

# MultiTargetMisfits: chiSmall scales the target petrophysical/smallness misfit (not geophysical).
USE_CHI_SMALL = False  # When True, apply CHI_SMALL in MultiTargetMisfits [I]
CHI_SMALL = 1.0  # [I]

# add learned mref in smooth once stable
WAIT_TILL_STABLE = True  # [I]

# Update the parameters in smallness (L2-approx of PGI)
UPDATE_GMM = False  # if True, update the GMM (MAP estimate) during the inversion [I]
KAPPA = np.c_[
    [1e10, 1e10],  # Crust
    [1e10, 1e10],  # Serpentinite
    [0, 0],  # Mantle
].T

# ScalingMultipleDataMisfits_ByEig: chi0_ratio sets initial relative misfit weights (then eig-balance + normalize).
CHI0_RATIO = [5e-4, 1e2]  # [I]


# ======================================================== Seeds for reproducibility (Q) ========================================================= #
# If you want fully random runs (non-reproducible), set USE_GLOBAL_SEED=False 
SEED = 12345  # [Q]
USE_GLOBAL_SEED = True  # [Q]
DEFAULT_SAMPLING_SEED = SEED if USE_GLOBAL_SEED else None  # [Q]

# ========================================================= Field and lab parameters (Q) ========================================================= #
# Distance between fractures in meters (size of the serpentinite blocks)
DIST_X = 1      # Units of m. [Q]
DIST_Y = 1      # Units of m. [Q]
DIST_Z = 1      # Units of m. [Q]

# Parameters for serpentinite
DENSITY_SERPENTINITE = 2910     # Density in kg/m³ (can be obtained from the inversion code) [Q]
POROSITY_FRONT = 8              # Units of %. Porosity of the serpentinites at the serpentinization front (e.g., Chogani and Plümper, 2023) [Q]
INT_FRACTURE_SPACING = 0.05     # m Internal fracture spacing  (within each serpentinite block to allow fluid flow) [Q]
PERMEABILITY_FRACTURES = 1e-20  # m² Permeability of the internal fractures [Q]
WATERROCKRATIO = 0.16           # No units. W/R values: [0.0, 0.02, 0.04, 0.06, 0.08, 0.1,  0.12, 0.14, 0.16, 0.18, 0.2] [Q]
FLOW_TARGET = 5e5               # [L/day] volumetric flow we want to deliver to mantle rocks (the saturation case will be calculated using this value) [Q]

# Other parameters
DENSITY_LITHO = 2700         # kg/m³ (for lithostatic pressure) [Q]
GRAVITY = 9.81               # m/s² [Q]
LITHOLOGY_CODE = 'LH1'       # Harzburgite HZ1, Lherzolite LH1 [Q]
YEARS = 1                    # duration of simulation in years (present time and conditions) [Q]
MOLAR_MASS_H2 = 0.00201588   # kg/mol [Q]
MOLAR_MASS_H2O = 0.01801528  # kg/mol [Q]

# ==================================================== Serpentinization front velocities (Q) ===================================================== #
# Serpentinization front velocities (cm/day). You must provide one reference point with T)
V_REF_SYNTHETIC = 1.272e-5     # cm/day. Reference velocity values from Malvoisin & Brunet (2014) [Q]
T_REF_RANGE = "300_325"        # Temperature range in °C for the reference velocity from Malvoisin & Brunet (2014) [Q]

# =========================================================== PARALLEL PROCESSING (Q) ============================================================ #
# Number of cores to use in parallel processing. When left as None the helper will fall back to the available CPUs.
N_CORES = None  # [Q]

# ===================================================== WATER FLOW SIMULATION IN FAULTS (Q) ====================================================== #
# Geometry/hydraulics to estimate if the target flow can be reached.
RUN_MONTECARLO_FAULT = True  # Run the Monte Carlo flow simulation [Q]
FAULT_MC_N_ITER = 5000  # number of Monte Carlo realizations (iterations) for the fault-flow simulation [Q]
FLOW_TARGET_FRACTURE_CONFIG = {
    "L_fault": 50000,                 # [m] along-strike length of the fault segment considered
    "width_min": 100,                 # [m] minimum width of the fault zone
    "width_max": 1500,                # [m] maximum width of the fault zone
    "fractured_length_min": 0.10,     # [-] min fraction of the total fault length that is fractured/active
    "fractured_length_max": 0.70,     # [-] max fraction of the total fault length that is fractured/active
    "D_min": 1000,                    # [m] minimum sampled depth / flow-path length (depth)
    "D_max": 12000,                   # [m] maximum sampled depth / flow-path length (depth)
    "fracture_density_min": 1/100,    # [fractures/m] min density
    "fracture_density_max": 1/5,      # [fractures/m] max density
    "deltaP_min": 60,                 # [MPa] minimum pressure
    "deltaP_max": 100,                # [MPa] maximum pressure
    "mu_min": 1e-3,                   # [Pa s] minimum viscosity
    "mu_max": 1e-5,                   # [Pa s] maximum viscosity
    "connection_fraction_min": 0.25,  # [-] min hydraulically connected fraction
    "connection_fraction_max": 0.50,  # [-] max hydraulically connected fraction
}

# ============================================================== NO SATURATION (Q) =============================================================== #
# Controls on the computation of H2 production with no solubility or saturation limits (purely theoretical output).
MC_NO_SATURATION_CONFIG = {
    "n_iter": 5000,                      # Number of Monte Carlo iterations
    "verbose": True,                     # Show detailed results
    "sampling": "sobol",                 # Sampling mode: "uniform", "lhs", or "sobol"
    "sampling_seed": DEFAULT_SAMPLING_SEED,  # DEFAULT_SAMPLING_SEED or none. Seed for reproducibility (None for random)
    "v_ref_range": (0.4, 1.6),           # Range factor for serpentinization front velocity
    "prod_rate_range": (0.5, 1.5),       # Range factor for production rate
    "volume_range": (0.8, 1.2),          # Range factor for serpentinite volume per temp range
    "serp_correction_range": (0.8, 1.2), # Range factor for serpentinization correction
    "surface_area_range": (0.6, 1.4),    # Range factor for fracture surface area per km³
    "show_progress": True,               # When True, display a tqdm bar during the volumetric Monte Carlo
}


# ================================================================ SATURATION (Q) ================================================================ #
# Physical and geochemical uncertainties for the calculation with saturation limits
MC_SATURATION_CONFIG = {
    "n_iter": 5000,                          # Number of Monte Carlo iterations
    "sampling": "sobol",                      # Sampling: 'lhs' (Latin Hypercube) or 'sobol' (quasi-random) for lower variance with the same n_iter
    "sampling_seed": DEFAULT_SAMPLING_SEED,   # DEFAULT_SAMPLING_SEED or none. Seed for reproducibility of sampling (LHS/Sobol)
    "dt_day": 1,                              # Time step for the within-day integration loop [day] (not yet implemented correctly)
    "max_chunk_size": 200,                    # Maximum number of iterations submitted per executor batch (smaller ⇒ faster feedback)
    "vol_range": (0.8, 1.2),                  # Factor for total volume per temperature range
    "mean_press_range": (0.7, 1.3),           # Factor for mean pressure per range
    "serp_deg_range": (0.8, 1.2),             # Factor for serpentinization degree
    "spacing_range": (0.5, 1.5),              # Factor for internal fracture spacing
    "perm_range": (0.8, 1.2),                 # Factor for fracture permeability
    "prod_rate_range": (0.8, 1.2),            # Factor for volumetric H₂ production
    "d_range": (0.5, 1.5),                    # Unified factor for dist_x, dist_y, dist_z
    "kg_rocks_range": (0.4, 1.6),             # Factor for reactive rock mass per range
    "solubility_scaling_range": (0.7, 1.3),   # Factor for H₂ solubility 
}

# Univariate sensitivity analysis for parameters (all other factors fixed at 1.0).
RUN_UNIVARIATE_ANALYSIS_NO_SAT = True  # enable univariate sweep for the no-saturation MC [Q]
RUN_UNIVARIATE_ANALYSIS_SAT = True # enable univariate sweep for the saturation MC [Q]
UNIVARIATE_ANALYSIS_CONFIG = {
    "n_points": 50,                          # number of factor values between (min,max)
    "n_rep": 100,                            # replicates per point
    "baseline_n_iter": 5000,                 # iterations for the baseline run (all factors = 1.0)
    "sampling": "sobol",                     # Sampling: 'lhs' (Latin Hypercube) or 'sobol' (quasi-random) for lower variance with the same n_iter
    "sampling_seed": DEFAULT_SAMPLING_SEED,  # reproducible seed for the sweep (None when USE_GLOBAL_SEED=False)
    "show_progress": True,                   # show per-parameter progress bar
    "make_plot": True,                       # Save parameter-vs-H2 curve plots for the sweep (PNG/SVG)
    "worker_count": N_CORES,                 # Use >1 to parallelize the inner MC per factor value; default to all available cores
}

# ============================================================ CONVERGENCE SWEEPS ============================================================= #
# Convergence sweep for saturation analysis (How many iterations are needed for convergence?). 
RUN_MC_CONVERGENCE_SWEEP = False
MC_CONVERGENCE_SWEEP_CONFIG = {
    "iter_values": [10, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000],  # Monte Carlo iterations to evaluate
    "reuse_from_single_run": False,       # If True, reuse a single large run instead of running per n_iter
    "result_column": "H2 total [tons]",   # Column used to evaluate convergence 
    "tolerance_mean_pct": 10.0,           # Allowed change (%) in mean when doubling n_iter
    "tolerance_std_pct": 5.0,             # Allowed change (%) in std when doubling n_iter
    "save_plot": True,                    # Save convergence plot alongside CSV (when run)
    "silence_runs": True                  # Suppress textual output during sweep runs (keep progress bar)
}

# Time steps are automatic. If you want to test your own time steps, set RUN_DT_CONVERGENCE_SWEEP = True
RUN_DT_CONVERGENCE_SWEEP = False
DT_CONVERGENCE_SWEEP_CONFIG = {
    "dt_values": [10, 7, 5, 3, 1, 1/100, 1/1000, 1/5000, 1/10000],   # Time-step values (in days) to evaluate
    "n_iter": 5000,                           # MC iterations per dt value (fallback to MC_SATURATION_CONFIG if None)
    "result_column": "H2 total [tons]",     # Column used to evaluate convergence
    "silence_runs": True,                   # Suppress textual output during sweep runs (keep progress bar)
    "save_plot": True,                      # Save plot of mean ± std vs dt_day
}
# ========================================================  FLOW TARGETS ========================================================
# Used to run tests across flow-target values to evaluate outputs and which factors are limiting.
RUN_ANALYZE_LIMITING_FACTORS = True  # Boolean to control whether analyze_limiting_factors_by_flow_target is executed
FLOW_TARGET_LOG_MIN = 1e1            # [L/day] minimum
FLOW_TARGET_LOG_MAX = 1e10           # [L/day] maximum
FLOW_TARGET_N_SAMPLES = 30           # [-] number of flow targets (the x axis is divided in this number)

MC_FLOW_TARGET_CONFIG = {
    "n_iter": 5000,                  # Number of Monte Carlo iterations per flow target
    "verbose": False,                 # Show detailed results
    "show_progress": False,           # Show progress bar
    "save_timeseries_plots": True,   # Save debug plots for timeseries totals per flow
}

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

CONFIG_PARAM_NAMES = [
    "RUN_INVERSION", "RUN_H2_QUANTIFICATION", "BASE_DIR", "TOPO_FILE", "GRAV_FILE", "MAG_FILE",
    "INITIAL_MODEL", "USE_INITIAL_MODEL", "BACKGROUND_DENS", "BACKGROUND_SUSC", "DB_DIR",
    "TEMPERATURE_FILE", "DENSITY_FILE", "MAGSUS_FILE", "DX", "DY", "DZ", "DEPTH_CORE",
    "EXPANSION_PERCENTAGE", "EXPANSION_TYPE", "DOMINANT_SIDE", "ORIGINAL_Y", "ORIGINAL_X",
    "UNCER_GRAV", "UNCER_MAG", "INCLINATION", "DECLINATION", "STRENGTH", "UNIT_LABELS",
    "UNIT_DENS_ADJ_LIST", "UNIT_MAGSUS_LIST", "UNIT_DENS_DISP_LIST", "UNIT_MAGSUS_DISP_LIST",
    "VOL_UNIT_LIST", "SERPENTINITE_LABEL", "LOW_DENS_ADJ", "UP_DENS_ADJ", "LOWER_SUSCEPTIBILITY",
    "UPPER_SUSCEPTIBILITY", "MAX_ITER", "MAX_ITER_LS", "MAX_ITER_CG", "TOL_CG",
    "SAVE_PRED_RESIDUAL_NPY", "SAVE_SVG", "ALPHA_PGI", "ALPHA_X", "ALPHA_Y", "ALPHA_Z", "ALPHA_XX",
    "ALPHA_YY", "ALPHA_ZZ", "ALPHA0_RATIO_DENS", "ALPHA0_RATIO_SUSC", "BETA0_RATIO",
    "COOLING_FACTOR", "TOLERANCE", "PROGRESS", "USE_CHI_SMALL", "CHI_SMALL", "WAIT_TILL_STABLE",
    "UPDATE_GMM", "KAPPA", "CHI0_RATIO", "SEED", "USE_GLOBAL_SEED", "DIST_X", "DIST_Y",
    "DIST_Z", "DENSITY_SERPENTINITE", "POROSITY_FRONT", "INT_FRACTURE_SPACING",
    "PERMEABILITY_FRACTURES", "WATERROCKRATIO", "FLOW_TARGET", "DENSITY_LITHO", "GRAVITY",
    "LITHOLOGY_CODE", "YEARS", "MOLAR_MASS_H2", "MOLAR_MASS_H2O", "V_REF_SYNTHETIC",
    "T_REF_RANGE", "N_CORES", "RUN_MONTECARLO_FAULT", "FAULT_MC_N_ITER",
    "FLOW_TARGET_FRACTURE_CONFIG", "MC_NO_SATURATION_CONFIG", "MC_SATURATION_CONFIG",
    "RUN_UNIVARIATE_ANALYSIS_NO_SAT", "RUN_UNIVARIATE_ANALYSIS_SAT", "UNIVARIATE_ANALYSIS_CONFIG",
    "RUN_MC_CONVERGENCE_SWEEP", "MC_CONVERGENCE_SWEEP_CONFIG", "RUN_DT_CONVERGENCE_SWEEP",
    "DT_CONVERGENCE_SWEEP_CONFIG", "RUN_ANALYZE_LIMITING_FACTORS", "FLOW_TARGET_LOG_MIN",
    "FLOW_TARGET_LOG_MAX", "FLOW_TARGET_N_SAMPLES", "MC_FLOW_TARGET_CONFIG",
]


CONFIG_TEXT_FILE = os.path.join(os.path.dirname(__file__), "ponhy_config.txt")
# Load optional text-based overrides for config parameters.
USE_TEXT_CONFIG = True  # Set to False to ignore the text config file and use only the hardcoded defaults in this module.

if __name__ == "__main__":
    try:
        response = input("Use config file 'ponhy_config.txt'? [Y/n]: ").strip().lower()
        if response in {"n", "no", "0", "false"}:
            USE_TEXT_CONFIG = False
    except (EOFError, KeyboardInterrupt):
        USE_TEXT_CONFIG = True

text_config = _load_text_config(CONFIG_TEXT_FILE, CONFIG_PARAM_NAMES) if USE_TEXT_CONFIG else None
if text_config is not None:
    # Apply config overrides to module globals with type-aware coercion.
    _apply_text_config(text_config, CONFIG_PARAM_NAMES, globals())

# Configure global plotting settings (e.g., optional SVG output).
set_plot_save_svg(SAVE_SVG)


if __name__ != "__main__":
    RUN_INVERSION = False
    RUN_H2_QUANTIFICATION = False

######################################################################################################################################################
################################################################# Inversion ##########################################################################
######################################################################################################################################################

INVERSION_PARAM_NAMES = [
    "BASE_DIR", "TOPO_FILE", "GRAV_FILE", "MAG_FILE", "INITIAL_MODEL", "USE_INITIAL_MODEL",
    "BACKGROUND_DENS", "BACKGROUND_SUSC", "DX", "DY", "DZ", "DEPTH_CORE",
    "EXPANSION_PERCENTAGE", "EXPANSION_TYPE", "DOMINANT_SIDE", "ORIGINAL_Y", "ORIGINAL_X",
    "UNCER_GRAV", "UNCER_MAG", "INCLINATION", "DECLINATION", "STRENGTH", "UNIT_LABELS",
    "UNIT_DENS_ADJ_LIST", "UNIT_MAGSUS_LIST", "UNIT_DENS_DISP_LIST", "UNIT_MAGSUS_DISP_LIST",
    "VOL_UNIT_LIST", "SERPENTINITE_LABEL", "LOW_DENS_ADJ", "UP_DENS_ADJ", "LOWER_SUSCEPTIBILITY",
    "UPPER_SUSCEPTIBILITY", "MAX_ITER", "MAX_ITER_LS", "MAX_ITER_CG", "TOL_CG",
    "SAVE_PRED_RESIDUAL_NPY", "SAVE_SVG", "ALPHA_PGI", "ALPHA_X", "ALPHA_Y", "ALPHA_Z", "ALPHA_XX",
    "ALPHA_YY", "ALPHA_ZZ",
    "ALPHA0_RATIO_DENS", "ALPHA0_RATIO_SUSC", "BETA0_RATIO", "COOLING_FACTOR", "TOLERANCE",
    "PROGRESS", "USE_CHI_SMALL", "CHI_SMALL", "WAIT_TILL_STABLE", "UPDATE_GMM", "KAPPA",
    "CHI0_RATIO",
]


if RUN_INVERSION:
    # Print the report header once for the full run (stdout is redirected later).
    _print_header_once()
    # Validate required inversion parameters are present and non-empty.
    missing_inversion = [name for name in INVERSION_PARAM_NAMES if _is_missing(globals().get(name))]
    if missing_inversion:
        # Report missing parameters with context-specific messaging.
        _report_missing_params(missing_inversion, "inversion")
        raise ValueError("Missing required inversion parameters.")

    START_TIME = time.time()
    CURRENT_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")

    DIR_PATH = os.path.join(BASE_DIR, "")

    FOLDER_NAME = f"Results_Inversion_{CURRENT_TIME}"
    RESULTS_PATH = os.path.join(DIR_PATH, FOLDER_NAME)

    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)

    OUTPUT_FILE_PATH = os.path.join(RESULTS_PATH, "output.txt")

    # Resolve input data paths relative to BASE_DIR.
    TOPO_FILENAME = _resolve_base_dir_path(BASE_DIR, TOPO_FILE)
    GRAV_DATA_FILENAME = _resolve_base_dir_path(BASE_DIR, GRAV_FILE)
    MAG_DATA_FILENAME = _resolve_base_dir_path(BASE_DIR, MAG_FILE)

    assert TOPO_FILENAME is not None, "Resolved TOPO_FILENAME is required"
    assert GRAV_DATA_FILENAME is not None, "Resolved GRAV_DATA_FILENAME is required"
    assert MAG_DATA_FILENAME is not None, "Resolved MAG_DATA_FILENAME is required"

     # Resolve and load the initial model CSV path for the inversion only when enabled.
    if USE_INITIAL_MODEL:
        INITIAL_MODEL_FILENAME = _resolve_base_dir_path(BASE_DIR, INITIAL_MODEL)
        assert INITIAL_MODEL_FILENAME is not None, "Resolved INITIAL_MODEL_FILENAME is required"
        if os.path.isdir(INITIAL_MODEL_FILENAME):
            raise IsADirectoryError(
                f"INITIAL_MODEL points to a directory: {INITIAL_MODEL_FILENAME}. "
                "Set INITIAL_MODEL to a CSV file or disable USE_INITIAL_MODEL."
            )
        if not os.path.exists(INITIAL_MODEL_FILENAME):
            raise FileNotFoundError(
                f"INITIAL_MODEL file not found: {INITIAL_MODEL_FILENAME}. "
                "Update INITIAL_MODEL to a valid CSV path or disable USE_INITIAL_MODEL."
            )
        INITIAL_MODEL_DATA = np.loadtxt(INITIAL_MODEL_FILENAME, delimiter=",", skiprows=1)
    else:
        INITIAL_MODEL_FILENAME = None
        INITIAL_MODEL_DATA = np.array([])

    

    # Redirect stdout to a file-based logger for the inversion output.
    CUSTOM_PRINTER = CustomPrint(OUTPUT_FILE_PATH)
    sys.stdout = CUSTOM_PRINTER

    SEPARATOR = "=" * 150
    SECTION_SEPARATOR = "-" * 150

    # Runtime globals 
    RECEIVER_GRAV: Any = None
    RECEIVER_MAG: Any = None
    NX = NY = NZ = 0
    TOTAL_DISTANCE_X = TOTAL_DISTANCE_Y = TOTAL_DISTANCE_Z = 0.0
    UNCERTAINTIES_GRAV: np.ndarray = np.array([])
    UNCERTAINTIES_MAG: np.ndarray = np.array([])
    ALPHA0_RATIO: Any = np.array([])
    BETA: Any = None
    BETA_IT: Any = None
    TARGETS: Any = None
    MREF_IN_SMOOTH: Any = None
    UPDATE_SMALLNESS: Any = None
    SAVE_VALUES_DIRECTIVE: Any = None
    SCALING_INIT: Any = None

    # ====================================================== Density adjustments / constants ===================================================== #
    (
        UNIT_LABELS_RESOLVED,
        UNIT_DENS_ADJ_RESOLVED,
        UNIT_MAGSUS_RESOLVED,
        UNIT_DENS_DISP_RESOLVED,
        UNIT_MAGSUS_DISP_RESOLVED,
        VOL_UNIT_RESOLVED,
    # Normalize and validate unit parameter lists (labels, means, dispersions, weights).
    ) = _normalize_unit_parameters(
        labels=UNIT_LABELS,
        dens_adj=UNIT_DENS_ADJ_LIST,
        magsus=UNIT_MAGSUS_LIST,
        dens_disp=UNIT_DENS_DISP_LIST,
        magsus_disp=UNIT_MAGSUS_DISP_LIST,
        weights=VOL_UNIT_LIST,
    )

    UNIT_DENS = 2.67 - UNIT_DENS_ADJ_RESOLVED

    LOWER_DENSITY = 2.67 - LOW_DENS_ADJ
    UPPER_DENSITY = 2.67 - UP_DENS_ADJ

    # ===================================================== Data loading (topo / grav / mag) ===================================================== #
    def _load_txt(path: str) -> np.ndarray:
        return np.loadtxt(str(path), dtype=np.float32)

    with ThreadPoolExecutor(max_workers=3) as ex:
        fut_topo = ex.submit(_load_txt, TOPO_FILENAME)
        fut_grav = ex.submit(_load_txt, GRAV_DATA_FILENAME)
        fut_mag = ex.submit(_load_txt, MAG_DATA_FILENAME)
        topo_xyz = fut_topo.result()
        dobs_grav = fut_grav.result()
        dobs_mag = fut_mag.result()

    receiver_locations = topo_xyz[:, 0:3]

    dobs_grav = dobs_grav[:, -1]
    dobs_mag = dobs_mag[:, -1]

    # ==================================================== Surveys setup (gravity / magnetic) ==================================================== #
    maximum_anomaly_grav = np.max(np.abs(dobs_grav))
    UNCERTAINTIES_GRAV = (UNCER_GRAV / 100.0) * maximum_anomaly_grav
    UNCERTAINTIES_GRAV = np.full_like(dobs_grav, UNCERTAINTIES_GRAV)
    maximum_anomaly_mag = np.max(np.abs(dobs_mag))
    UNCERTAINTIES_MAG = (UNCER_MAG / 100.0) * maximum_anomaly_mag
    UNCERTAINTIES_MAG = np.full_like(dobs_mag, UNCERTAINTIES_MAG)

    RECEIVER_GRAV = pf.gravity.receivers.Point(receiver_locations, components="gz")
    source_field_grav = pf.gravity.sources.SourceField(receiver_list=[RECEIVER_GRAV])
    survey_grav = pf.gravity.survey.Survey(source_field_grav)

    RECEIVER_MAG = pf.magnetics.receivers.Point(receiver_locations, components="tmi")
    source_field_mag = pf.magnetics.sources.UniformBackgroundField(
        receiver_list=[RECEIVER_MAG],
        amplitude=STRENGTH,
        inclination=INCLINATION,
        declination=DECLINATION,
    )
    survey_mag = pf.magnetics.survey.Survey(source_field_mag)

    data_object_grav = data.Data(survey_grav, dobs=dobs_grav, standard_deviation=UNCERTAINTIES_GRAV)
    data_object_mag = data.Data(survey_mag, dobs=dobs_mag, standard_deviation=UNCERTAINTIES_MAG)

    # =================================================== Mesh, active cells, and initial model ================================================== #
    # Build the 3D inversion mesh from the topography grid.
    mesh, NX, NY, NZ = build_mesh_from_topography(topo_xyz, DX, DY, DZ, DEPTH_CORE)
    actv = active_from_xyz(mesh, topo_xyz, "CC")

    assert getattr(mesh, 'cell_centers', None) is not None, "mesh.cell_centers is required"
    assert getattr(mesh, 'cell_volumes', None) is not None, "mesh.cell_volumes is required"
    cell_centers = cast(np.ndarray, mesh.cell_centers)
    cell_volumes = cast(np.ndarray, mesh.cell_volumes)
    assert actv is not None, "active cell mask (actv) should not be None"

    ndv = np.nan 
    actvMap = maps.InjectActiveCells(mesh, actv, ndv)
    nactv = int(actv.sum()) 

    TOTAL_DISTANCE_X = NX * DX
    TOTAL_DISTANCE_Y = NY * DY
    TOTAL_DISTANCE_Z = NZ * DZ

    xyz_active = mesh.gridCC[actv, :] 

    if USE_INITIAL_MODEL:
        x_csv, y_csv, z_csv = INITIAL_MODEL_DATA[:, 0], INITIAL_MODEL_DATA[:, 1], INITIAL_MODEL_DATA[:, 2]
        initial_density, initial_susceptibility = INITIAL_MODEL_DATA[:, 3], INITIAL_MODEL_DATA[:, 4]
        xyz_csv = np.vstack((x_csv, y_csv, z_csv)).T
        # Interpolate initial model properties onto active mesh cells.
        density_mesh = interpolate_nearest_neighbor(xyz_csv, initial_density, xyz_active)
        susceptibility_mesh = interpolate_nearest_neighbor(xyz_csv, initial_susceptibility, xyz_active)
        m0 = np.r_[density_mesh, susceptibility_mesh]
    else:
        m0 = np.r_[BACKGROUND_DENS * np.ones(actvMap.nP), BACKGROUND_SUSC * np.ones(actvMap.nP)]

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
        n_components=len(UNIT_DENS),  
        mesh=mesh,  
        actv=actv,  
        covariance_type="diag",  
    )

    gmmref.fit(np.random.randn(nactv, 2))
    gmmref.means_ = np.column_stack((UNIT_DENS, UNIT_MAGSUS_RESOLVED))

    gmmref.covariances_ = np.column_stack((
        UNIT_DENS_DISP_RESOLVED ** 2,
        UNIT_MAGSUS_DISP_RESOLVED ** 2,
    ))

    gmmref.compute_clusters_precisions()
    gmmref.weights_ = VOL_UNIT_RESOLVED

    ax = gmmref.plot_pdf(flag2d=True, plotting_precision=100, padding=2)
    ax[0].set_xlabel("Density contrast [g/cc]")
    ax[2].set_ylabel("magnetic Susceptibility [SI]")
    fig_init = ax[0].figure if hasattr(ax[0], "figure") else plt.gcf()
    file_base = os.path.join(RESULTS_PATH, 'initial_GMM')

    # Save the initial GMM plot in PNG/SVG formats.
    _save_plot_pair(file_base, fig_init)
    plt.close(fig_init)

    # =============================================== Weights, PGI regularization, and alphas/betas ============================================== #
    x_min, x_max = cell_centers[actv][:, 0].min(), cell_centers[actv][:, 0].max()
    y_min, y_max = cell_centers[actv][:, 1].min(), cell_centers[actv][:, 1].max()
    border_distance = 0 * max(DX, DY)

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
    weights_file_path = os.path.join(RESULTS_PATH, 'weights_gravity_magnetism.csv')
    df_weights.to_csv(weights_file_path, index=False)
                    
    reg = regularization.PGI(
        gmmref=gmmref,
        mesh=mesh,
        wiresmap=wires,
        maplist=[idenMap, idenMap],
        active_cells=actv,
        alpha_pgi=ALPHA_PGI,
        alpha_x=ALPHA_X,
        alpha_y=ALPHA_Y,
        alpha_z=ALPHA_Z,
        alpha_xx=ALPHA_XX,
        alpha_yy=ALPHA_YY,
        alpha_zz=ALPHA_ZZ,
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

    ALPHA0_RATIO = np.r_[
        ALPHA0_RATIO_DENS * np.ones(dens_terms),
        ALPHA0_RATIO_SUSC * np.ones(susc_terms),
    ]

    Alphas = directives.AlphasSmoothEstimate_ByEig(alpha0_ratio=ALPHA0_RATIO, verbose=True)

    BETA = directives.BetaEstimate_ByEig(beta0_ratio=BETA0_RATIO)

    BETA_IT = directives.PGI_BetaAlphaSchedule(
        verbose=True,
        coolingFactor=COOLING_FACTOR,
        tolerance=TOLERANCE,
        progress=PROGRESS,
    )

    if USE_CHI_SMALL:
        TARGETS = directives.MultiTargetMisfits(
            verbose=True,
            chiSmall=CHI_SMALL,
        )
    else:
        TARGETS = directives.MultiTargetMisfits(
            verbose=True,
        )

    MREF_IN_SMOOTH = directives.PGI_AddMrefInSmooth(
        wait_till_stable=WAIT_TILL_STABLE,
        verbose=True,
    )

    UPDATE_SMALLNESS = directives.PGI_UpdateParameters(
        update_gmm=UPDATE_GMM,
        kappa=KAPPA,
    )

    update_Jacobi = directives.UpdatePreconditioner()

    SCALING_INIT = directives.ScalingMultipleDataMisfits_ByEig(chi0_ratio=CHI0_RATIO)
    scale_schedule = directives.JointScalingSchedule(verbose=True)

    # =============================================== Optimization, inverse problem, and directives ============================================== #
    # Optimization setup
    lowerbound = np.r_[UPPER_DENSITY * np.ones(actvMap.nP), LOWER_SUSCEPTIBILITY * np.ones(actvMap.nP)]
    upperbound = np.r_[LOWER_DENSITY * np.ones(actvMap.nP), UPPER_SUSCEPTIBILITY * np.ones(actvMap.nP)]

    if lowerbound is not None and upperbound is not None:
        opt = optimization.ProjectedGNCG(
            maxIter=MAX_ITER,
            lower=lowerbound,
            upper=upperbound,
            maxIterLS=MAX_ITER_LS,
            maxIterCG=MAX_ITER_CG,
            tolCG=TOL_CG,
        )
    else:
        opt = optimization.ProjectedGNCG(
            maxIter=MAX_ITER,
            maxIterLS=MAX_ITER_LS,
            maxIterCG=MAX_ITER_CG,
            tolCG=TOL_CG,
        )

    invProb = inverse_problem.BaseInvProblem(dmis, reg, opt)

    directiveList = [
        Alphas,
        SCALING_INIT,
        BETA,
        UPDATE_SMALLNESS,
        TARGETS,
        scale_schedule,
        BETA_IT,
        MREF_IN_SMOOTH,
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

    SAVE_VALUES_DIRECTIVE = SaveValuesDirective()
    directiveList.append(SAVE_VALUES_DIRECTIVE)

    # =============================================================== Run inversion ============================================================== #
    # ==================================================================================================================================================== 
    print("\n" + SEPARATOR)
    print("1. INVERSION CONFIGURATION SUMMARY".center(150))
    print(SEPARATOR)
    # ==================================================================================================================================================== 
    
    # Print a full inversion configuration summary for the report.
    print_inversion_config_summary(
        topo_filename=TOPO_FILENAME,
        grav_filename=GRAV_DATA_FILENAME,
        mag_filename=MAG_DATA_FILENAME,
        receiver_grav=RECEIVER_GRAV,
        receiver_mag=RECEIVER_MAG,
        strength=STRENGTH,
        inclination=INCLINATION,
        declination=DECLINATION,
        nx=NX,
        ny=NY,
        nz=NZ,
        total_distance_x=TOTAL_DISTANCE_X,
        total_distance_y=TOTAL_DISTANCE_Y,
        total_distance_z=TOTAL_DISTANCE_Z,
        dx=DX,
        dy=DY,
        dz=DZ,
        use_initial_model=USE_INITIAL_MODEL,
        initial_model_filename=INITIAL_MODEL_FILENAME,
        background_dens=BACKGROUND_DENS,
        background_susc=BACKGROUND_SUSC,
        uncer_grav=UNCER_GRAV,
        uncer_mag=UNCER_MAG,
        uncertainties_grav=UNCERTAINTIES_GRAV,
        uncertainties_mag=UNCERTAINTIES_MAG,
    unit_labels=UNIT_LABELS_RESOLVED,
    unit_dens_adj=UNIT_DENS_ADJ_RESOLVED,
    unit_magsus=UNIT_MAGSUS_RESOLVED,
    unit_dens_disp=UNIT_DENS_DISP_RESOLVED,
    unit_magsus_disp=UNIT_MAGSUS_DISP_RESOLVED,
        alpha0_ratio=ALPHA0_RATIO,
        beta=BETA,
        beta_it=BETA_IT,
        targets=TARGETS,
        mref_in_smooth=MREF_IN_SMOOTH,
        update_smallness=UPDATE_SMALLNESS,
        scaling_init=SCALING_INIT,
        section_separator=SECTION_SEPARATOR,
    )

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
    plot_misfit_evolution(SAVE_VALUES_DIRECTIVE, results_path=RESULTS_PATH)

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
        results_path=RESULTS_PATH,
        true_means=gmmref.means_,
        model_colors=quasi_geology_model_no_info  
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

    if EXPANSION_TYPE == "rectangular":
        original_distance_x = TOTAL_DISTANCE_X / (1 + EXPANSION_PERCENTAGE / 100)
        original_distance_y = TOTAL_DISTANCE_Y / (1 + EXPANSION_PERCENTAGE / 100)
        expansion_x = (TOTAL_DISTANCE_X - original_distance_x) / 2
        expansion_y = (TOTAL_DISTANCE_Y - original_distance_y) / 2

    elif EXPANSION_TYPE == "square":
        L_new = TOTAL_DISTANCE_X

        if DOMINANT_SIDE == "x":
            original_x = L_new / (1 + EXPANSION_PERCENTAGE / 100)
            expansion_x = (L_new - original_x) / 2
            expansion_y = (L_new - ORIGINAL_Y) / 2

        elif DOMINANT_SIDE == "y":
            original_y = L_new / (1 + EXPANSION_PERCENTAGE / 100)
            expansion_y = (L_new - original_y) / 2
            expansion_x = (L_new - ORIGINAL_X) / 2

    x_min_central = x_min + expansion_x
    x_max_central = x_max - expansion_x
    y_min_central = y_min + expansion_y
    y_max_central = y_max - expansion_y

    central_mask = (
        (xyz_active[:, 0] >= x_min_central) & (xyz_active[:, 0] <= x_max_central + DX) &
        (xyz_active[:, 1] >= y_min_central) & (xyz_active[:, 1] <= y_max_central + DY)
    )

    xyz_active = xyz_active[central_mask]
    density_model_no_info = density_model_no_info[central_mask]
    magsus_model_no_info = magsus_model_no_info[central_mask]
    quasi_geology_model_no_info = quasi_geology_model_no_info[central_mask]

    export_data_density_complete = np.hstack((xyz_active, density_model_no_info.reshape(-1, 1)))
    export_data_magsus_complete = np.hstack((xyz_active, magsus_model_no_info.reshape(-1, 1)))
    export_geology_complete = np.hstack((xyz_active, quasi_geology_model_no_info.reshape(-1, 1)))

    density_filepath = os.path.join(RESULTS_PATH, "Density_complete_model.csv")
    np.savetxt(density_filepath, export_data_density_complete, delimiter=',', header="X,Y,Z,Density", fmt="%.6f")

    magsus_filepath = os.path.join(RESULTS_PATH, "Magsus_complete_model.csv")
    np.savetxt(magsus_filepath, export_data_magsus_complete, delimiter=',', header="X,Y,Z,Magsus", fmt="%.6f")

    geology_filepath = os.path.join(RESULTS_PATH, "Geology_complete_model.csv")
    np.savetxt(geology_filepath, export_geology_complete, delimiter=',', header="X,Y,Z,Geology", fmt="%.6f")

    mask_geology = (quasi_geology_model_no_info == SERPENTINITE_LABEL)
    density_model_no_info[~mask_geology] = np.nan
    magsus_model_no_info[~mask_geology] = np.nan

    density_export_cells = density_model_no_info
    magsus_export_cells = magsus_model_no_info

    export_data_density = np.hstack((xyz_active, density_export_cells.reshape(-1, 1)))
    export_data_magsus = np.hstack((xyz_active, magsus_export_cells.reshape(-1, 1)))
    density_filepath = os.path.join(RESULTS_PATH, "Density_Serpentinite.csv")
    np.savetxt(density_filepath, export_data_density, delimiter=',', header="X,Y,Z,Density", fmt="%.6f")
    magsus_filepath = os.path.join(RESULTS_PATH, "Magsus_Serpentinite.csv")
    np.savetxt(magsus_filepath, export_data_magsus, delimiter=',', header="X,Y,Z,Magsus", fmt="%.6f")

    if RUN_H2_QUANTIFICATION:
        DENSITY_FILE = density_filepath
        MAGSUS_FILE = magsus_filepath

    # ==================================================== Fields, residuals, and final plots ==================================================== #
    fields_grav = simulation_grav.fields(pgi_model_no_info)
    d_pred_grav = simulation_grav.dpred(pgi_model_no_info, f=fields_grav)
    fields_mag = simulation_mag.fields(pgi_model_no_info)
    d_pred_mag = simulation_mag.dpred(pgi_model_no_info, f=fields_mag)

    if SAVE_PRED_RESIDUAL_NPY:
        np.save(os.path.join(RESULTS_PATH, "pred_grav"), d_pred_grav)
        np.save(os.path.join(RESULTS_PATH, "pred_mag"), d_pred_mag)

    residual_grav = dobs_grav - d_pred_grav
    residual_mag = dobs_mag - d_pred_mag

    if SAVE_PRED_RESIDUAL_NPY:
        np.save(os.path.join(RESULTS_PATH, "residual_grav"), residual_grav)
        np.save(os.path.join(RESULTS_PATH, "residual_mag"), residual_mag)

    # Plot observed vs predicted gravity and residuals.
    plot_gravity_observed_predicted_residual(
        receiver_locations,
        dobs_grav,
        d_pred_grav,
        residual_grav,
        results_path=RESULTS_PATH,
    )
    # Plot observed vs predicted magnetic data and residuals.
    plot_magnetic_observed_predicted_residual(
        receiver_locations,
        dobs_mag,
        d_pred_mag,
        residual_mag,
        results_path=RESULTS_PATH,
    )
    # Plot residual histograms for gravity and magnetic data.
    plot_residual_histograms(residual_grav, residual_mag, results_path=RESULTS_PATH)

    # ============================================================= Summary and close ============================================================ #
    END_TIME = time.time()
    EXECUTION_TIME = END_TIME - START_TIME
    HOURS = EXECUTION_TIME // 3600
    MINUTES = (EXECUTION_TIME % 3600) // 60
    SECONDS = EXECUTION_TIME % 60
    FORMATTED_TIME = f"{int(HOURS)}:{int(MINUTES)}:{int(SECONDS)}"

    # Print the inversion results summary (timing, outputs, iterations).
    print_inversion_results_summary(
        separator=SEPARATOR,
        section_separator=SECTION_SEPARATOR,
        results_path=RESULTS_PATH,
        formatted_time=FORMATTED_TIME,
        opt_iter=opt_iter,
        opt_max_iter=opt_max_iter,
    )

    CUSTOM_PRINTER.close()
else:
    print("Skipping Routine 1 (inversion).")


######################################################################################################################################################
################################################################# Quantification #####################################################################
######################################################################################################################################################

QUANT_PARAM_NAMES = [
    "BASE_DIR", "TOPO_FILE", "DB_DIR", "TEMPERATURE_FILE", "DENSITY_FILE", "MAGSUS_FILE",
    "SEED", "USE_GLOBAL_SEED", "DIST_X", "DIST_Y", "DIST_Z", "DENSITY_SERPENTINITE",
    "POROSITY_FRONT", "INT_FRACTURE_SPACING", "PERMEABILITY_FRACTURES", "WATERROCKRATIO",
    "FLOW_TARGET", "DENSITY_LITHO", "GRAVITY", "LITHOLOGY_CODE", "YEARS", "MOLAR_MASS_H2",
    "MOLAR_MASS_H2O", "V_REF_SYNTHETIC", "T_REF_RANGE", "SAVE_SVG", "RUN_MONTECARLO_FAULT", "FAULT_MC_N_ITER",
    "FLOW_TARGET_FRACTURE_CONFIG", "MC_NO_SATURATION_CONFIG", "MC_SATURATION_CONFIG",
    "RUN_UNIVARIATE_ANALYSIS_NO_SAT", "RUN_UNIVARIATE_ANALYSIS_SAT", "UNIVARIATE_ANALYSIS_CONFIG",
    "RUN_MC_CONVERGENCE_SWEEP", "MC_CONVERGENCE_SWEEP_CONFIG", "RUN_DT_CONVERGENCE_SWEEP",
    "DT_CONVERGENCE_SWEEP_CONFIG", "RUN_ANALYZE_LIMITING_FACTORS", "FLOW_TARGET_LOG_MIN",
    "FLOW_TARGET_LOG_MAX", "FLOW_TARGET_N_SAMPLES", "MC_FLOW_TARGET_CONFIG", "lithologies_dict",
    "serpentinization_data", "serp_corr_percentage",
]


# Shared context for Monte Carlo worker processes (avoids re-pickling heavy arguments per task)
_MC_WORKER_CONTEXT: Optional[Dict[str, Any]] = None


if RUN_H2_QUANTIFICATION:
    # Print the report header once for the quantification run.
    _print_header_once()

    # Validate required quantification parameters are present and non-empty.
    missing_quant = [name for name in QUANT_PARAM_NAMES if _is_missing(globals().get(name))]

    if not RUN_MONTECARLO_FAULT:
        for name in ["FAULT_MC_N_ITER", "FLOW_TARGET_FRACTURE_CONFIG"]:
            if name in missing_quant:
                missing_quant.remove(name)

    if not RUN_UNIVARIATE_ANALYSIS_NO_SAT and not RUN_UNIVARIATE_ANALYSIS_SAT:
        if "UNIVARIATE_ANALYSIS_CONFIG" in missing_quant:
            missing_quant.remove("UNIVARIATE_ANALYSIS_CONFIG")

    if not RUN_MC_CONVERGENCE_SWEEP and "MC_CONVERGENCE_SWEEP_CONFIG" in missing_quant:
        missing_quant.remove("MC_CONVERGENCE_SWEEP_CONFIG")

    if not RUN_DT_CONVERGENCE_SWEEP and "DT_CONVERGENCE_SWEEP_CONFIG" in missing_quant:
        missing_quant.remove("DT_CONVERGENCE_SWEEP_CONFIG")

    if not RUN_ANALYZE_LIMITING_FACTORS:
        for name in ["FLOW_TARGET_LOG_MIN", "FLOW_TARGET_LOG_MAX", "FLOW_TARGET_N_SAMPLES", "MC_FLOW_TARGET_CONFIG"]:
            if name in missing_quant:
                missing_quant.remove(name)

    if missing_quant:

        # Report missing parameters with context-specific messaging.
        _report_missing_params(missing_quant, "H2 quantification")
        raise ValueError("Missing required H2 quantification parameters.")
    # ============================================================== Begin of time =============================================================== #
    start_time = time.time()
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    dir_path = BASE_DIR + os.path.sep

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
    topo_filename = _resolve_base_dir_path(dir_path, TOPO_FILE)
    assert topo_filename is not None, "Resolved topo_filename is required"

    # Resolve the H2 database path.
    dsn_db = _resolve_base_dir_path(dir_path, DB_DIR)
    assert dsn_db is not None, "Resolved dsn_db is required"

    # Resolve the temperature model path.
    temperature_filename = _resolve_base_dir_path(dir_path, TEMPERATURE_FILE)
    assert temperature_filename is not None, "Resolved temperature_filename is required"

    # Resolve density and magnetic susceptibility model paths.
    density_path = _resolve_base_dir_path(dir_path, DENSITY_FILE)
    magsus_path = _resolve_base_dir_path(dir_path, MAGSUS_FILE)
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

    if USE_GLOBAL_SEED:
        print(f"\n[INFO] Random generators seeded with SEED={SEED}.")
        np.random.seed(SEED)
        random.seed(SEED)
    else:
        print("[INFO] Random generators running without a fixed seed.")

    # Ensure the configured number of cores never exceeds the machine capacity.
    N_CORES = _clamp_worker_count(N_CORES)

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
    if RUN_MONTECARLO_FAULT:
        # Run Monte Carlo sampling to estimate fracture permeability and flow success.
        k_samples, success = run_fracture_monte_carlo_simulation(
            FLOW_TARGET_FRACTURE_CONFIG,
            flow_target=FLOW_TARGET,
            n_samples=FAULT_MC_N_ITER,
            make_plot=True
        )
        # Call print and plot helpers consecutively
        results = {
            "k": k_samples,
            "success": success,
            "success_rate": np.mean(success) * 100 if len(success) > 0 else 0,
            "flow_target": FLOW_TARGET,
            "Q_m3s": None  # required by helpers
        }
        # Print the fault-flow Monte Carlo summary.
        print_fracture_monte_carlo_report(results)
        # Plot the histogram of sampled fracture permeability values.
        plot_fracture_mc_histogram(results)
    else:
        print("Skipped fault-flow Monte Carlo (RUN_MONTECARLO_FAULT=False).")

    # ====================================================================================================================================================
    print("\n" + separator)
    print("3. CONFIGURATION SUMMARY".center(150))
    print(separator)
    # ====================================================================================================================================================

    porosity_front = POROSITY_FRONT / 100.0

    # - Load gridded topography and define the surface receiver locations used downstream.
    topo_xyz = np.loadtxt(str(topo_filename))
    receiver_locations = topo_xyz[:, 0:3]
    x = receiver_locations[:, 0]
    y = receiver_locations[:, 1]

    # - Build the 3D mesh from topography and gather basic mesh stats.
    # Build the 3D mesh for quantification from topography.
    mesh, nx, ny, nz = build_mesh_from_topography(topo_xyz, DX, DY, DZ, DEPTH_CORE)

    # Active cell coordinates and filtered property models
    xyz_active = export_data_density[:, :3]
    density_model_no_info = export_data_density[:, 3]
    magsus_model_no_info = export_data_magsus[:, 3]
    density_export_cells = density_model_no_info
    magsus_export_cells = magsus_model_no_info

    # Load external temperature CSV, interpolate onto active cells.
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
        DX,
        DY,
        DZ,
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
        save_svg=SAVE_SVG,
    )

    # Print a compact summary of the inversion/mesh configuration used for quantification.
    print_inversion_config(topo_filename, DX, DY, DZ, DIST_X, DIST_Y, DIST_Z)

    # Load the thermodynamic H2 production database for the target lithologies.
    thermo_data = load_h2_production_database(dsn_db, rock_codes=["HZ1", "LH1"])

    # ====================================================================================================================================================
    print("\n" + separator)
    print("4. SUMMARY OF VOLUMES AND DEGREE OF SERPENTINIZATION".center(150))
    print(separator + "\n")
    # ====================================================================================================================================================

    # Print the serpentinization volume/degree summary tables.
    serpentinization_corrections, total_volume = print_summary_serpentinization_section(
        volume_density_magsus,
        volume_at_temperature_total_100_500,
        mode_density,
        mode_magsus,
        volume_at_temperature,
        depths_for_temp_extremes,
        serpentinization_degree,
        serp_corr_percentage,
        temperature_ranges=list(temperature_ranges.keys()),
    )
    # Compute mean lithostatic pressures and depth stats per temperature range.
    mean_pressure_ranges, mean_depths = compute_mean_lithostatic_pressure_by_range(
        temperature_mesh,
        xyz_active,
        temperature_ranges,
        density_litho=DENSITY_LITHO,
        gravity=GRAVITY,
    )

    # Build a lookup of H2 production rates by temperature/pressure range.
    thermo_lookup = build_thermo_lookup_by_range(
        thermo_data=thermo_data,
        lithology_code=LITHOLOGY_CODE,
        waterrockratio=WATERROCKRATIO,
        temperature_ranges=temperature_ranges,
        mean_pressure_ranges=mean_pressure_ranges,
    )

    # ====================================================================================================================================================
    print("\n" + separator)
    print(f"5. ROCK DATABASE: Lithology: {LITHOLOGY_CODE} | W/R Ratio = {WATERROCKRATIO}".center(150))
    print(separator)
    # ====================================================================================================================================================

    # Print production-rate tables by temperature range.
    print_production_rate_volumetric_report(
        thermo_data=thermo_data,
        lithology_code=LITHOLOGY_CODE,
        waterrockratio=WATERROCKRATIO,
        temperature_ranges=temperature_ranges,
        mean_pressure_ranges=mean_pressure_ranges,
        lithologies_dict=lithologies_dict,
        thermo_lookup=thermo_lookup,
    )
    # Build a dict of production rates for downstream calculations.
    production_rate_volumetric = build_production_rate_volumetric_dict(
        thermo_data=thermo_data,
        lithology_code=LITHOLOGY_CODE,
        waterrockratio=WATERROCKRATIO,
        temperature_ranges=temperature_ranges,
        mean_pressure_ranges=mean_pressure_ranges,
        thermo_lookup=thermo_lookup,
    )


    # Derive serpentinization-front velocities per temperature range.
    serpentinization_front_velocities = compute_serpentinization_front_velocities(
        production_rate_volumetric,
        WATERROCKRATIO,
        T_REF_RANGE,
        V_REF_SYNTHETIC
    )

    # ====================================================================================================================================================
    print("\n" + separator)
    print(f"{'6. PARAMETERS FOR WATER FLOW & SERPENTINIZATION':^150}")
    print(separator + "\n")
    # ====================================================================================================================================================
             
    # Print the water-flow and serpentinization parameter summary.
    print_parameters_water_flow_serpentinization(
        FLOW_TARGET,
        INT_FRACTURE_SPACING,
        PERMEABILITY_FRACTURES,
        porosity_front,
        DENSITY_SERPENTINITE,
        WATERROCKRATIO,
        FLOW_TARGET_FRACTURE_CONFIG,
        DENSITY_LITHO,
        GRAVITY,
        YEARS,
        LITHOLOGY_CODE,
        MOLAR_MASS_H2,
        MOLAR_MASS_H2O
    )

    # ==================================================================================================================================================== 
    print()
    print("\n" + separator)
    print(f"{'7. H2 ESTIMATION NO SATURATION ':^130}")
    print(separator + "\n")
    # ====================================================================================================================================================
    # Full no-saturation volumetric pipeline (surface area -> MC -> summary/CSV)

    (   results_temp_volumetric, kg_rocks_dict, std_results_temp_volumetric, df_no_saturation,
        total_tons_no_sat, no_sat_csv_path, total_kg_rocks,
    # Run the no-saturation Monte Carlo workflow and export results.
    ) = run_no_saturation_workflow(
        temperature_ranges=temperature_ranges, serpentinization_corrections=serpentinization_corrections,
        serpentinization_front_velocities=serpentinization_front_velocities, density_serpentinite=DENSITY_SERPENTINITE,
        production_rate_volumetric=production_rate_volumetric, waterrockratio=WATERROCKRATIO,
        volume_at_temperature=volume_at_temperature, mean_pressure_ranges=mean_pressure_ranges,
        thermo_data=thermo_data, lithology_code=LITHOLOGY_CODE, porosity_front=porosity_front,
        results_path=results_path, years=YEARS,
        dist_x=DIST_X, dist_y=DIST_Y, dist_z=DIST_Z,
        mc_config=MC_NO_SATURATION_CONFIG,
        temp_bins=TEMP_BINS,
        molar_mass_h2=MOLAR_MASS_H2,
        molar_mass_h2o=MOLAR_MASS_H2O,
        thermo_lookup=thermo_lookup, run_prints=True,
    )

    # ====================================================================================================================================================
    print(separator)
    print(f"{'8. H2 ESTIMATION WITH SATURATION ':^130}")
    print(separator)
    # ====================================================================================================================================================
    # Full saturation volumetric pipeline (surface area -> MC -> summary/CSV)

    # Run the saturation-limited Monte Carlo workflow and export results.
    stats_mc_saturation, total_tons_sat, std_total_mc, mean_efficiency = run_saturation_workflow(
        volume_at_temperature=volume_at_temperature,
        df_saturation_table=df_no_saturation,
        mean_pressure_ranges=mean_pressure_ranges,
        serpentinization_degree=serpentinization_degree,
        int_fracture_spacing=INT_FRACTURE_SPACING,
        permeability_fractures=PERMEABILITY_FRACTURES,
        flow_target=FLOW_TARGET,
        production_rate_volumetric=production_rate_volumetric,
        years=YEARS,
        dist_x=DIST_X,
        dist_y=DIST_Y,
        dist_z=DIST_Z,
        kg_rocks_dict=kg_rocks_dict,
        total_kg_rocks=total_kg_rocks,
        total_tons_no_sat=total_tons_no_sat,
        results_path=results_path,
        mc_config=MC_SATURATION_CONFIG,
        n_cores=N_CORES,
        seed=SEED,
        porosity_front=porosity_front,
        density_serpentinite=DENSITY_SERPENTINITE,
        run_prints=True,
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
    mc_common_kwargs = dict(
        volume_at_temperature=volume_at_temperature,
        df_saturation_table=df_no_saturation,
        mean_pressure_ranges=mean_pressure_ranges,
        serpentinization_degree=serpentinization_degree,
        int_fracture_spacing=INT_FRACTURE_SPACING,
        permeability_fractures=PERMEABILITY_FRACTURES,
        flow_target=FLOW_TARGET,
        production_rate_volumetric=production_rate_volumetric,
        years=YEARS,
        dist_x=DIST_X,
        dist_y=DIST_Y,
        dist_z=DIST_Z,
        kg_rocks_dict=kg_rocks_dict,
        verbose=True,
        show_progress=True,
        mc_config=MC_SATURATION_CONFIG,
        n_cores=N_CORES,
        seed=SEED,
        porosity_front=porosity_front,
        density_serpentinite=DENSITY_SERPENTINITE,
    )

    # Ensure we pass a valid surface area per km³ (small fractures) to the no-sat MC.
    # If the global was not set (e.g., running univariate directly), recompute it here.
    _surface_area_no_sat = globals().get("surface_area_per_km3", None)
    if _surface_area_no_sat is None:
        original_volume_m3 = 1000 ** 3  # 1 km³
        voxel_volume_m3 = DIST_X * DIST_Y * DIST_Z
        total_voxels = original_volume_m3 / voxel_volume_m3
        voxel_surface_area = 2 * (DIST_X * DIST_Y + DIST_X * DIST_Z + DIST_Y * DIST_Z)
        _surface_area_no_sat = voxel_surface_area * total_voxels
        globals()["surface_area_per_km3"] = _surface_area_no_sat

    mc_no_sat_kwargs = dict(
        temperature_ranges=temperature_ranges,
        serpentinization_corrections=serpentinization_corrections,
        serpentinization_front_velocities=serpentinization_front_velocities,
        density_serpentinite=DENSITY_SERPENTINITE,
        production_rate_volumetric=production_rate_volumetric,
        waterrockratio=WATERROCKRATIO,
        volume_at_temperature=volume_at_temperature,
        surface_area_per_km3=_surface_area_no_sat,
        mc_config=MC_NO_SATURATION_CONFIG,
    )

    if RUN_UNIVARIATE_ANALYSIS_SAT:
        # Run univariate sensitivity sweeps for the saturation MC.
        run_saturation_univariate_sweep(
            mc_kwargs=mc_common_kwargs,
            base_config=MC_SATURATION_CONFIG,
            uni_config=UNIVARIATE_ANALYSIS_CONFIG,
            results_dir=results_path,
            n_cores=N_CORES,
            seed=SEED,
            mc_saturation_config=MC_SATURATION_CONFIG,
        )
    else:
        print("Skipped univariate sensitivity (RUN_UNIVARIATE_ANALYSIS_SAT=False)")


    if RUN_UNIVARIATE_ANALYSIS_NO_SAT:
        # Run univariate sensitivity sweeps for the no-saturation MC.
        run_no_saturation_univariate_sweep(
            mc_kwargs=mc_no_sat_kwargs,
            base_config=MC_NO_SATURATION_CONFIG,
            uni_config=UNIVARIATE_ANALYSIS_CONFIG,
            results_dir=results_path,
            n_cores=N_CORES,
            seed=SEED,
        )
    else:
        print("Skipped univariate sensitivity (no-saturation) (RUN_UNIVARIATE_ANALYSIS_NO_SAT=False)")


    if RUN_MC_CONVERGENCE_SWEEP or RUN_DT_CONVERGENCE_SWEEP:
        # ====================================================================================================================================================
        print("\n" + separator)
        print(f"{'CONVERGENCE SWEEPS':^150}")
        print(separator)
        # ====================================================================================================================================================

    if RUN_MC_CONVERGENCE_SWEEP:
        base_df_mc = stats_mc_saturation.attrs.get("df_all") if stats_mc_saturation is not None else None
        # Evaluate convergence vs iteration count for the saturation MC.
        run_mc_convergence_sweep(
            MC_CONVERGENCE_SWEEP_CONFIG,
            base_df=base_df_mc,
            mc_kwargs=mc_common_kwargs,
            results_dir=results_path
        )
    else:
        print("")
        print("Skipped MC convergence sweep (RUN_MC_CONVERGENCE_SWEEP=False).")


    if RUN_DT_CONVERGENCE_SWEEP:
        # Evaluate convergence vs time-step size for saturation MC.
        run_dt_convergence_sweep(
            DT_CONVERGENCE_SWEEP_CONFIG,
            mc_kwargs=mc_common_kwargs,
            results_dir=results_path,
            mc_saturation_config=MC_SATURATION_CONFIG,
            seed=SEED,
        )
    else:
        print("")
        print("Skipped dt_day convergence sweep (RUN_DT_CONVERGENCE_SWEEP=False).")


    if RUN_ANALYZE_LIMITING_FACTORS:
        # ====================================================================================================================================================
        print("\n" + separator)
        print(f"{'FLOW TARGETS':^150}")
        print(separator)
        # ====================================================================================================================================================

        # Analyze limiting factors across a range of flow-target values.
        analyze_limiting_factors_by_flow_target(
            FLOW_TARGET_FRACTURE_CONFIG=FLOW_TARGET_FRACTURE_CONFIG,
            volume_at_temperature=volume_at_temperature,
            df_saturation_table=df_no_saturation,
            mean_pressure_ranges=mean_pressure_ranges,
            serpentinization_degree=serpentinization_degree,
            int_fracture_spacing=INT_FRACTURE_SPACING,
            permeability_fractures=PERMEABILITY_FRACTURES,
            production_rate_volumetric=production_rate_volumetric,
            years=YEARS,
            dist_x=DIST_X,
            dist_y=DIST_Y,
            dist_z=DIST_Z,
            kg_rocks_dict=kg_rocks_dict,
            flow_target_log_min=FLOW_TARGET_LOG_MIN,
            flow_target_log_max=FLOW_TARGET_LOG_MAX,
            flow_target_n_samples=FLOW_TARGET_N_SAMPLES,
            mc_flow_target_config=MC_FLOW_TARGET_CONFIG,
            mc_saturation_config=MC_SATURATION_CONFIG,
            porosity_front=porosity_front,
            density_serpentinite=DENSITY_SERPENTINITE,
            results_path=results_path,
            total_tons_no_sat=total_tons_no_sat,
            n_cores=N_CORES,
            seed=SEED,
        )
    else:
        print("Skipped flow-target Monte Carlo analysis (RUN_ANALYZE_LIMITING_FACTORS=False)")

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
