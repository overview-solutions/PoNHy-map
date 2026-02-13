from __future__ import annotations

from dataclasses import asdict, dataclass, fields, MISSING, replace
from typing import Any, Dict, Optional, Sequence, Tuple, Union, get_args, get_origin

import yaml


@dataclass
class FlowTargetFractureConfig:
    L_fault: float
    width_min: float
    width_max: float
    fractured_length_min: float
    fractured_length_max: float
    D_min: float
    D_max: float
    fracture_density_min: float
    fracture_density_max: float
    deltaP_min: float
    deltaP_max: float
    mu_min: float
    mu_max: float
    connection_fraction_min: float
    connection_fraction_max: float


@dataclass
class McNoSaturationConfig:
    n_iter: int
    verbose: bool
    sampling: str
    sampling_seed: Optional[int]
    v_ref_range: Tuple[float, float]
    prod_rate_range: Tuple[float, float]
    volume_range: Tuple[float, float]
    serp_correction_range: Tuple[float, float]
    surface_area_range: Tuple[float, float]
    show_progress: bool


@dataclass
class McSaturationConfig:
    n_iter: int
    sampling: str
    sampling_seed: Optional[int]
    dt_day: float
    max_chunk_size: int
    vol_range: Tuple[float, float]
    mean_press_range: Tuple[float, float]
    serp_deg_range: Tuple[float, float]
    spacing_range: Tuple[float, float]
    perm_range: Tuple[float, float]
    prod_rate_range: Tuple[float, float]
    d_range: Tuple[float, float]
    kg_rocks_range: Tuple[float, float]
    solubility_scaling_range: Tuple[float, float]


@dataclass
class UnivariateAnalysisConfig:
    n_points: int
    n_rep: int
    baseline_n_iter: int
    sampling: str
    sampling_seed: Optional[int]
    show_progress: bool
    quiet: bool
    make_plot: bool
    worker_count: Optional[int]


@dataclass
class McConvergenceSweepConfig:
    iter_values: Sequence[int]
    reuse_from_single_run: bool
    result_column: str
    tolerance_mean_pct: float
    tolerance_std_pct: float
    save_plot: bool
    silence_runs: bool


@dataclass
class McFlowTargetConfig:
    n_iter: int
    verbose: bool
    show_progress: bool
    save_timeseries_plots: bool


@dataclass
class Config:
    run_inversion: bool
    run_h2_quantification: bool
    base_dir: str
    topo_file: str
    grav_file: str
    mag_file: str
    initial_model: str
    use_initial_model: bool
    background_dens: float
    background_susc: float
    db_dir: str
    temperature_file: str
    density_file: str
    magsus_file: str
    save_svg: bool
    dx: float
    dy: float
    dz: float
    depth_core: float
    expansion_percentage: float
    expansion_type: str
    dominant_side: str
    original_y: float
    original_x: float
    uncer_grav: float
    uncer_mag: float
    inclination: float
    declination: float
    strength: float
    unit_labels: Sequence[str]
    unit_dens_adj_list: Sequence[float]
    unit_magsus_list: Sequence[float]
    unit_dens_disp_list: Sequence[float]
    unit_magsus_disp_list: Sequence[float]
    vol_unit_list: Sequence[float]
    serpentinite_label: int
    low_dens_adj: float
    up_dens_adj: float
    lower_susceptibility: float
    upper_susceptibility: float
    max_iter: int
    max_iter_ls: int
    max_iter_cg: int
    tol_cg: float
    save_pred_residual_npy: bool
    alpha_pgi: float
    alpha_x: float
    alpha_y: float
    alpha_z: float
    alpha_xx: float
    alpha_yy: float
    alpha_zz: float
    alpha0_ratio_dens: float
    alpha0_ratio_susc: float
    beta0_ratio: float
    cooling_factor: float
    tolerance: float
    progress: float
    use_chi_small: bool
    chi_small: float
    wait_till_stable: bool
    update_gmm: bool
    kappa: Sequence[Sequence[float]]
    chi0_ratio: Sequence[float]
    seed: int
    use_global_seed: bool
    dist_x: float
    dist_y: float
    dist_z: float
    density_serpentinite: float
    porosity_front: float
    int_fracture_spacing: float
    permeability_fractures: float
    waterrockratio: float
    flow_target: float
    density_litho: float
    gravity: float
    lithology_code: str
    years: float
    molar_mass_h2: float
    molar_mass_h2o: float
    v_ref_synthetic: float
    t_ref_range: str
    n_cores: Optional[int]
    run_montecarlo_fault: bool
    fault_mc_n_iter: int
    flow_target_fracture_config: FlowTargetFractureConfig
    mc_no_saturation_config: McNoSaturationConfig
    mc_saturation_config: McSaturationConfig
    run_univariate_analysis_no_sat: bool
    run_univariate_analysis_sat: bool
    univariate_analysis_config: UnivariateAnalysisConfig
    run_mc_convergence_sweep: bool
    mc_convergence_sweep_config: McConvergenceSweepConfig
    run_analyze_limiting_factors: bool
    flow_target_log_min: float
    flow_target_log_max: float
    flow_target_n_samples: int
    mc_flow_target_config: McFlowTargetConfig


@dataclass
class InversionConfigSummary:
    topo_filename: str
    grav_filename: str
    mag_filename: str
    receiver_grav: Any
    receiver_mag: Any
    strength: float
    inclination: float
    declination: float
    nx: int
    ny: int
    nz: int
    total_distance_x: float
    total_distance_y: float
    total_distance_z: float
    dx: float
    dy: float
    dz: float
    use_initial_model: bool
    initial_model_filename: Optional[str]
    background_dens: float
    background_susc: float
    uncer_grav: float
    uncer_mag: float
    uncertainties_grav: Any
    uncertainties_mag: Any
    unit_labels: Sequence[str]
    unit_dens_adj: Sequence[float]
    unit_magsus: Sequence[float]
    unit_dens_disp: Sequence[float]
    unit_magsus_disp: Sequence[float]
    alpha0_ratio: Any
    beta: Any
    beta_it: Any
    targets: Any
    mref_in_smooth: Any
    update_smallness: Any
    scaling_init: Any
    separator: str = "="
    section_separator: str = "-"

    @classmethod
    def from_locals(cls, scope: Dict[str, Any]) -> "InversionConfigSummary":
        return _from_locals(cls, scope)


@dataclass
class InversionMeshConfig:
    topo_filename: str
    dx: float
    dy: float
    dz: float
    dist_x: float
    dist_y: float
    dist_z: float

    @classmethod
    def from_locals(cls, scope: Dict[str, Any]) -> "InversionMeshConfig":
        return _from_locals(cls, scope)


@dataclass
class InversionResultsSummary:
    separator: str
    section_separator: str
    results_path: str
    formatted_time: str
    opt_iter: Optional[int] = None
    opt_max_iter: Optional[int] = None

    @classmethod
    def from_locals(cls, scope: Dict[str, Any]) -> "InversionResultsSummary":
        return _from_locals(cls, scope)


@dataclass
class SerpentinizationSummaryParams:
    volume_density_magsus: Any
    volume_at_temperature: Dict[str, Any]
    volume_at_temperature_total_100_500: Any
    combined_masks: Any
    combined_mask_total_100_500: Any
    mode_density: float
    mode_magsus: float
    serpentinization_degree: float
    serp_corr_percentage: Dict[str, Any]
    temperature_ranges: Dict[str, Any]
    depths_for_temp_extremes: Dict[str, Any]

    @classmethod
    def from_locals(cls, scope: Dict[str, Any]) -> "SerpentinizationSummaryParams":
        return _from_locals(cls, scope)


@dataclass
class MeanLithostaticPressureParams:
    temperature_mesh: Any
    xyz_mesh_temperature: Any
    temperature_ranges: Dict[str, Any]
    density_litho: float
    gravity: float

    @classmethod
    def from_locals(cls, scope: Dict[str, Any]) -> "MeanLithostaticPressureParams":
        return _from_locals(cls, scope)


@dataclass
class ThermoLookupParams:
    thermo_data: Any
    lithology_code: str
    waterrockratio: float
    temperature_ranges: Sequence[str]
    mean_pressure_ranges: Dict[str, Any]

    @classmethod
    def from_locals(cls, scope: Dict[str, Any]) -> "ThermoLookupParams":
        return _from_locals(cls, scope)


@dataclass
class ProductionRateReportParams:
    thermo_data: Any
    lithology_code: str
    waterrockratio: float
    temperature_ranges: Sequence[str]
    mean_pressure_ranges: Dict[str, Any]
    lithologies_dict: Optional[Dict[str, Any]] = None
    thermo_lookup: Optional[Dict[str, Any]] = None

    @classmethod
    def from_locals(cls, scope: Dict[str, Any]) -> "ProductionRateReportParams":
        return _from_locals(cls, scope)


@dataclass
class ProductionRateVolumetricParams:
    thermo_data: Any
    lithology_code: str
    waterrockratio: float
    temperature_ranges: Sequence[str]
    mean_pressure_ranges: Dict[str, Any]
    thermo_lookup: Optional[Dict[str, Any]] = None

    @classmethod
    def from_locals(cls, scope: Dict[str, Any]) -> "ProductionRateVolumetricParams":
        return _from_locals(cls, scope)


@dataclass
class SerpentinizationFrontVelocityParams:
    production_rate_volumetric: Dict[str, Any]
    waterrockratio: float
    t_ref_range: str
    v_ref_synthetic: float

    @classmethod
    def from_locals(cls, scope: Dict[str, Any]) -> "SerpentinizationFrontVelocityParams":
        return _from_locals(cls, scope)


@dataclass
class WaterFlowSerpentinizationParams:
    flow_target: float
    int_fracture_spacing: float
    permeability_fractures: float
    porosity_front: float
    density_serpentinite: float
    waterrockratio: float
    flow_target_fracture_config: Dict[str, Any]
    density_litho: float
    gravity: float
    years: float
    lithology_code: str
    molar_mass_h2: float
    molar_mass_h2o: float

    @classmethod
    def from_cfg(cls, cfg: Config, **overrides: Any) -> "WaterFlowSerpentinizationParams":
        payload = {
            "flow_target": cfg.flow_target,
            "int_fracture_spacing": cfg.int_fracture_spacing,
            "permeability_fractures": cfg.permeability_fractures,
            "porosity_front": cfg.porosity_front,
            "density_serpentinite": cfg.density_serpentinite,
            "waterrockratio": cfg.waterrockratio,
            "flow_target_fracture_config": asdict(cfg.flow_target_fracture_config),
            "density_litho": cfg.density_litho,
            "gravity": cfg.gravity,
            "years": cfg.years,
            "lithology_code": cfg.lithology_code,
            "molar_mass_h2": cfg.molar_mass_h2,
            "molar_mass_h2o": cfg.molar_mass_h2o,
        }
        payload.update(overrides)
        return cls(**payload)


@dataclass
class NoSaturationWorkflowParams:
    temperature_ranges: Dict[str, Any]
    serpentinization_corrections: Dict[str, Any]
    serpentinization_front_velocities: Dict[str, Any]
    density_serpentinite: float
    production_rate_volumetric: Dict[str, Any]
    waterrockratio: float
    volume_at_temperature: Dict[str, Any]
    mean_pressure_ranges: Dict[str, Any]
    thermo_data: Any
    lithology_code: str
    porosity_front: float
    results_path: str
    years: float
    dist_x: float
    dist_y: float
    dist_z: float
    mc_config: Any
    temp_bins: Sequence[str]
    molar_mass_h2: float
    molar_mass_h2o: float
    thermo_lookup: Optional[Dict[str, Any]] = None
    surface_area_per_km3: Optional[float] = None
    run_prints: bool = True

    @classmethod
    def from_locals(cls, scope: Dict[str, Any]) -> "NoSaturationWorkflowParams":
        return _from_locals(cls, scope)


@dataclass
class SaturationWorkflowParams:
    volume_at_temperature: Dict[str, Any]
    df_saturation_table: Any
    mean_pressure_ranges: Dict[str, Any]
    serpentinization_degree: float
    int_fracture_spacing: float
    permeability_fractures: float
    flow_target: float
    production_rate_volumetric: Dict[str, Any]
    years: float
    dist_x: float
    dist_y: float
    dist_z: float
    kg_rocks_dict: Dict[str, Any]
    total_kg_rocks: float
    total_tons_no_sat: float
    results_path: str
    mc_config: Any
    n_cores: Optional[int]
    seed: Optional[int]
    porosity_front: float
    density_serpentinite: float
    run_prints: bool = True

    @classmethod
    def from_locals(cls, scope: Dict[str, Any]) -> "SaturationWorkflowParams":
        return _from_locals(cls, scope)


@dataclass
class SaturationSummaryParams:
    total_kg_rocks: float
    years: float
    total_tons_sat: float
    std_total_mc: float
    mean_efficiency: float

    @classmethod
    def from_locals(cls, scope: Dict[str, Any]) -> "SaturationSummaryParams":
        return _from_locals(cls, scope)


@dataclass
class NoSaturationSummaryReportParams:
    df_no_saturation: Any
    total_tons_no_sat: float
    total_kg_rocks: float
    years: float
    no_sat_csv_path: str

    @classmethod
    def from_locals(cls, scope: Dict[str, Any]) -> "NoSaturationSummaryReportParams":
        return _from_locals(cls, scope)


@dataclass
class FlowTargetLimitingFactorsParams:
    flow_target_fracture_config: Dict[str, Any]
    volume_at_temperature: Dict[str, float]
    df_saturation_table: Any
    mean_pressure_ranges: Dict[str, float]
    serpentinization_degree: float
    int_fracture_spacing: float
    permeability_fractures: float
    production_rate_volumetric: Dict[str, Any]
    years: float
    dist_x: float
    dist_y: float
    dist_z: float
    kg_rocks_dict: Optional[Dict[str, float]]
    flow_target_log_min: float
    flow_target_log_max: float
    flow_target_n_samples: int
    mc_flow_target_config: Dict[str, Any]
    mc_saturation_config: Dict[str, Any]
    porosity_front: float
    density_serpentinite: float
    results_path: Optional[str] = None
    total_tons_no_sat: float = 0.0
    n_cores: Optional[int] = None
    seed: Optional[int] = None


@dataclass
class SaturationMonteCarloReportParams:
    stats: Any
    years: float
    flow_target: float

    @classmethod
    def from_locals(cls, scope: Dict[str, Any]) -> "SaturationMonteCarloReportParams":
        return _from_locals(cls, scope)


def _from_locals(cls, scope: Dict[str, Any]):
    payload = {}
    for field in fields(cls):
        name = field.name
        if name in scope:
            payload[name] = scope[name]
            continue
        upper_name = name.upper()
        if upper_name in scope:
            payload[name] = scope[upper_name]
            continue
        raise KeyError(f"Missing '{name}' when building {cls.__name__}")
    return cls(**payload)


def _get_scope_value(scope: Dict[str, Any], name: str) -> Any:
    alias_map = {
        "grav_filename": "grav_data_filename",
        "mag_filename": "mag_data_filename",
        "unit_dens_adj": "unit_dens_adj_resolved",
        "unit_magsus": "unit_magsus_resolved",
        "unit_dens_disp": "unit_dens_disp_resolved",
        "unit_magsus_disp": "unit_magsus_disp_resolved",
    }
    if name in alias_map:
        alias_name = alias_map[name]
        if alias_name in scope:
            return scope[alias_name]
        upper_alias = alias_name.upper()
        if upper_alias in scope:
            return scope[upper_alias]
        lower_alias = alias_name.lower()
        if lower_alias in scope:
            return scope[lower_alias]
    if name in scope:
        return scope[name]
    upper_name = name.upper()
    if upper_name in scope:
        return scope[upper_name]
    lower_name = name.lower()
    if lower_name in scope:
        return scope[lower_name]
    raise KeyError(f"Missing '{name}' in scope")


def _build_from_scope(
    cls,
    scope: Dict[str, Any],
    *,
    extra_defaults: Optional[Dict[str, Any]] = None,
) -> Any:
    payload: Dict[str, Any] = {}
    for field in fields(cls):
        name = field.name
        try:
            payload[name] = _get_scope_value(scope, name)
            continue
        except KeyError:
            if extra_defaults and name in extra_defaults:
                payload[name] = extra_defaults[name]
                continue
            has_default = field.default is not MISSING or field.default_factory is not MISSING
            if not has_default:
                raise
    return cls(**payload)


def build_inversion_config_summary(scope: Dict[str, Any]) -> "InversionConfigSummary":
    return _build_from_scope(InversionConfigSummary, scope)


def build_inversion_mesh_config(scope: Dict[str, Any]) -> "InversionMeshConfig":
    return _build_from_scope(InversionMeshConfig, scope)


def build_inversion_results_summary(scope: Dict[str, Any]) -> "InversionResultsSummary":
    return _build_from_scope(InversionResultsSummary, scope)


def build_serpentinization_summary_params(scope: Dict[str, Any]) -> "SerpentinizationSummaryParams":
    return _build_from_scope(SerpentinizationSummaryParams, scope)


def build_mean_lithostatic_pressure_params(scope: Dict[str, Any]) -> "MeanLithostaticPressureParams":
    return _build_from_scope(MeanLithostaticPressureParams, scope)


def build_thermo_lookup_params(scope: Dict[str, Any]) -> "ThermoLookupParams":
    return _build_from_scope(ThermoLookupParams, scope)


def build_production_rate_report_params(scope: Dict[str, Any]) -> "ProductionRateReportParams":
    return _build_from_scope(ProductionRateReportParams, scope)


def build_production_rate_volumetric_params(scope: Dict[str, Any]) -> "ProductionRateVolumetricParams":
    return _build_from_scope(ProductionRateVolumetricParams, scope)


def build_serpentinization_front_velocity_params(scope: Dict[str, Any]) -> "SerpentinizationFrontVelocityParams":
    return _build_from_scope(SerpentinizationFrontVelocityParams, scope)


def build_saturation_monte_carlo_report_params(scope: Dict[str, Any]) -> "SaturationMonteCarloReportParams":
    return _build_from_scope(SaturationMonteCarloReportParams, scope)


def build_saturation_summary_params(scope: Dict[str, Any]) -> "SaturationSummaryParams":
    return _build_from_scope(SaturationSummaryParams, scope)


def build_no_saturation_workflow_params(
    cfg: Config,
    scope: Dict[str, Any],
) -> "NoSaturationWorkflowParams":
    mc_config_default = scope.get("mc_no_saturation_config", None) or asdict(cfg.mc_no_saturation_config)
    extra_defaults = {
        "density_serpentinite": cfg.density_serpentinite,
        "waterrockratio": cfg.waterrockratio,
        "porosity_front": cfg.porosity_front,
        "years": cfg.years,
        "molar_mass_h2": cfg.molar_mass_h2,
        "molar_mass_h2o": cfg.molar_mass_h2o,
        "mc_config": mc_config_default,
    }
    return _build_from_scope(NoSaturationWorkflowParams, scope, extra_defaults=extra_defaults)


def build_no_saturation_summary_report_params(scope: Dict[str, Any]) -> "NoSaturationSummaryReportParams":
    return _build_from_scope(NoSaturationSummaryReportParams, scope)


def build_flow_target_limiting_factors_params(
    scope: Dict[str, Any],
    *,
    cfg: Optional[Config] = None,
) -> "FlowTargetLimitingFactorsParams":
    extra_defaults = None
    if cfg is not None:
        extra_defaults = {
            "mc_flow_target_config": asdict(cfg.mc_flow_target_config),
            "mc_saturation_config": asdict(cfg.mc_saturation_config),
            "porosity_front": cfg.porosity_front,
            "density_serpentinite": cfg.density_serpentinite,
            "n_cores": cfg.n_cores,
            "seed": cfg.seed,
        }
    return _build_from_scope(FlowTargetLimitingFactorsParams, scope, extra_defaults=extra_defaults)


def build_mc_common_kwargs(scope: Dict[str, Any], *, cfg: Optional[Config] = None) -> Dict[str, Any]:
    if cfg is None:
        cfg = scope.get("cfg", None)
    return {
        "volume_at_temperature": scope.get("volume_at_temperature"),
        "df_saturation_table": scope.get("df_no_saturation"),
        "mean_pressure_ranges": scope.get("mean_pressure_ranges"),
        "serpentinization_degree": scope.get("serpentinization_degree"),
        "int_fracture_spacing": scope.get("int_fracture_spacing") if scope.get("int_fracture_spacing") is not None else getattr(cfg, "int_fracture_spacing", None),
        "permeability_fractures": scope.get("permeability_fractures") if scope.get("permeability_fractures") is not None else getattr(cfg, "permeability_fractures", None),
        "flow_target": scope.get("flow_target") if scope.get("flow_target") is not None else getattr(cfg, "flow_target", None),
        "production_rate_volumetric": scope.get("production_rate_volumetric"),
        "years": scope.get("years") if scope.get("years") is not None else getattr(cfg, "years", None),
        "dist_x": scope.get("dist_x") if scope.get("dist_x") is not None else getattr(cfg, "dist_x", None),
        "dist_y": scope.get("dist_y") if scope.get("dist_y") is not None else getattr(cfg, "dist_y", None),
        "dist_z": scope.get("dist_z") if scope.get("dist_z") is not None else getattr(cfg, "dist_z", None),
        "kg_rocks_dict": scope.get("kg_rocks_dict"),
        "verbose": True,
        "show_progress": True,
        "mc_config": scope.get("mc_saturation_config") if scope.get("mc_saturation_config") is not None else asdict(cfg.mc_saturation_config) if cfg is not None else None,
        "n_cores": scope.get("n_cores") if scope.get("n_cores") is not None else getattr(cfg, "n_cores", None),
        "seed": scope.get("seed") if scope.get("seed") is not None else getattr(cfg, "seed", None),
        "porosity_front": scope.get("porosity_front") if scope.get("porosity_front") is not None else getattr(cfg, "porosity_front", None),
        "density_serpentinite": scope.get("density_serpentinite") if scope.get("density_serpentinite") is not None else getattr(cfg, "density_serpentinite", None),
    }


def build_mc_no_sat_kwargs(scope: Dict[str, Any], *, surface_area_per_km3: Optional[float] = None, cfg: Optional[Config] = None) -> Dict[str, Any]:
    if cfg is None:
        cfg = scope.get("cfg", None)
    return {
        "temperature_ranges": scope.get("temperature_ranges"),
        "serpentinization_corrections": scope.get("serpentinization_corrections"),
        "serpentinization_front_velocities": scope.get("serpentinization_front_velocities"),
        "density_serpentinite": scope.get("density_serpentinite") if scope.get("density_serpentinite") is not None else getattr(cfg, "density_serpentinite", None),
        "production_rate_volumetric": scope.get("production_rate_volumetric"),
        "waterrockratio": scope.get("waterrockratio") if scope.get("waterrockratio") is not None else getattr(cfg, "waterrockratio", None),
        "volume_at_temperature": scope.get("volume_at_temperature"),
        "surface_area_per_km3": surface_area_per_km3,
        "mc_config": scope.get("mc_no_saturation_config") if scope.get("mc_no_saturation_config") is not None else asdict(cfg.mc_no_saturation_config) if cfg is not None else None,
    }


def build_saturation_workflow_params(
    cfg: Config,
    scope: Dict[str, Any],
) -> "SaturationWorkflowParams":
    mc_config_default = scope.get("mc_saturation_config", None) or asdict(cfg.mc_saturation_config)
    extra_defaults = {
        "int_fracture_spacing": cfg.int_fracture_spacing,
        "permeability_fractures": cfg.permeability_fractures,
        "flow_target": cfg.flow_target,
        "years": cfg.years,
        "porosity_front": cfg.porosity_front,
        "density_serpentinite": cfg.density_serpentinite,
        "mc_config": mc_config_default,
        "n_cores": cfg.n_cores,
        "seed": cfg.seed,
    }
    return _build_from_scope(SaturationWorkflowParams, scope, extra_defaults=extra_defaults)


def _as_tuple(value: Any) -> Tuple[float, float]:
    if value is None:
        return (1.0, 1.0)
    if isinstance(value, (list, tuple)):
        if len(value) == 2:
            return (float(value[0]), float(value[1]))
        raise ValueError(f"Expected a 2-item range, got {len(value)} values: {value}")
    return (float(value), float(value))


def _maybe_none(value: Any) -> Optional[int]:
    if value == 0 or value == "auto":
        return None
    if value is None:
        return None
    return int(value)


def _get(data: Dict[str, Any], key: str) -> Any:
    if key in data:
        return data[key]
    upper_key = key.upper()
    if upper_key in data:
        return data[upper_key]
    lower_key = key.lower()
    if lower_key in data:
        return data[lower_key]
    raise KeyError(f"Missing required config key: {key}")


def _has_key(data: Dict[str, Any], key: str) -> bool:
    return key in data or key.upper() in data or key.lower() in data


def _default_for_type(tp: Any) -> Any:
    origin = get_origin(tp)
    args = get_args(tp)
    if origin is tuple and len(args) == 2:
        return (1.0, 1.0)
    if origin in (list, tuple, Sequence):
        return []
    if origin is Union and type(None) in args:
        return None
    if tp is bool:
        return False
    if tp is int:
        return 0
    if tp is float:
        return 0.0
    if tp is str:
        return ""
    return None


def _default_dataclass_payload(cls) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for field in fields(cls):
        if field.name == "sampling":
            payload[field.name] = "uniform"
        elif field.name == "sampling_seed":
            payload[field.name] = None
        elif field.name == "n_iter":
            payload[field.name] = 0
        elif field.name in {
            "verbose",
            "show_progress",
            "reuse_from_single_run",
            "save_plot",
            "silence_runs",
            "save_timeseries_plots",
        }:
            payload[field.name] = False
        else:
            payload[field.name] = _default_for_type(field.type)
    return payload


def _seed_disabled_section_defaults(data: Dict[str, Any], *, run_inversion: bool, run_h2_quantification: bool) -> None:
    inversion_only_fields = {
        "grav_file",
        "mag_file",
        "initial_model",
        "use_initial_model",
        "background_dens",
        "background_susc",
        "dx",
        "dy",
        "dz",
        "depth_core",
        "expansion_percentage",
        "expansion_type",
        "dominant_side",
        "original_y",
        "original_x",
        "uncer_grav",
        "uncer_mag",
        "inclination",
        "declination",
        "strength",
        "unit_labels",
        "unit_dens_adj_list",
        "unit_magsus_list",
        "unit_dens_disp_list",
        "unit_magsus_disp_list",
        "vol_unit_list",
        "serpentinite_label",
        "low_dens_adj",
        "up_dens_adj",
        "lower_susceptibility",
        "upper_susceptibility",
        "max_iter",
        "max_iter_ls",
        "max_iter_cg",
        "tol_cg",
        "save_pred_residual_npy",
        "alpha_pgi",
        "alpha_x",
        "alpha_y",
        "alpha_z",
        "alpha_xx",
        "alpha_yy",
        "alpha_zz",
        "alpha0_ratio_dens",
        "alpha0_ratio_susc",
        "beta0_ratio",
        "cooling_factor",
        "tolerance",
        "progress",
        "use_chi_small",
        "chi_small",
        "wait_till_stable",
        "update_gmm",
        "kappa",
        "chi0_ratio",
    }
    quant_only_fields = {
        "db_dir",
        "temperature_file",
        "density_file",
        "magsus_file",
        "seed",
        "use_global_seed",
        "dist_x",
        "dist_y",
        "dist_z",
        "density_serpentinite",
        "porosity_front",
        "int_fracture_spacing",
        "permeability_fractures",
        "waterrockratio",
        "flow_target",
        "density_litho",
        "gravity",
        "lithology_code",
        "years",
        "molar_mass_h2",
        "molar_mass_h2o",
        "v_ref_synthetic",
        "t_ref_range",
        "n_cores",
        "run_montecarlo_fault",
        "fault_mc_n_iter",
        "flow_target_fracture_config",
        "mc_no_saturation_config",
        "mc_saturation_config",
        "run_univariate_analysis_no_sat",
        "run_univariate_analysis_sat",
        "univariate_analysis_config",
        "run_mc_convergence_sweep",
        "mc_convergence_sweep_config",
        "run_analyze_limiting_factors",
        "flow_target_log_min",
        "flow_target_log_max",
        "flow_target_n_samples",
        "mc_flow_target_config",
    }
    field_types = {field.name: field.type for field in fields(Config)}
    dataclass_defaults = {
        "flow_target_fracture_config": _default_dataclass_payload(FlowTargetFractureConfig),
        "mc_no_saturation_config": _default_dataclass_payload(McNoSaturationConfig),
        "mc_saturation_config": _default_dataclass_payload(McSaturationConfig),
        "univariate_analysis_config": _default_dataclass_payload(UnivariateAnalysisConfig),
        "mc_convergence_sweep_config": _default_dataclass_payload(McConvergenceSweepConfig),
        "mc_flow_target_config": _default_dataclass_payload(McFlowTargetConfig),
    }

    if not run_inversion:
        for field_name in inversion_only_fields:
            key = field_name.upper()
            if not _has_key(data, key):
                data[key] = _default_for_type(field_types.get(field_name))

    if not run_h2_quantification:
        for field_name in quant_only_fields:
            key = field_name.upper()
            if _has_key(data, key):
                continue
            if field_name in dataclass_defaults:
                data[key] = dataclass_defaults[field_name]
            else:
                data[key] = _default_for_type(field_types.get(field_name))


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Config YAML must contain a top-level mapping.")

    run_inversion = bool(_get(data, "run_inversion"))
    run_h2_quantification = bool(_get(data, "run_h2_quantification"))
    _seed_disabled_section_defaults(
        data,
        run_inversion=run_inversion,
        run_h2_quantification=run_h2_quantification,
    )

    flow_target_fracture_config = FlowTargetFractureConfig(**_get(data, "FLOW_TARGET_FRACTURE_CONFIG"))
    mc_no_saturation_data = _get(data, "MC_NO_SATURATION_CONFIG")
    mc_no_saturation_config = McNoSaturationConfig(
        n_iter=int(mc_no_saturation_data["n_iter"]),
        verbose=bool(mc_no_saturation_data["verbose"]),
        sampling=str(mc_no_saturation_data["sampling"]),
        sampling_seed=mc_no_saturation_data.get("sampling_seed", None),
        v_ref_range=_as_tuple(mc_no_saturation_data["v_ref_range"]),
        prod_rate_range=_as_tuple(mc_no_saturation_data["prod_rate_range"]),
        volume_range=_as_tuple(mc_no_saturation_data["volume_range"]),
        serp_correction_range=_as_tuple(mc_no_saturation_data["serp_correction_range"]),
        surface_area_range=_as_tuple(mc_no_saturation_data["surface_area_range"]),
        show_progress=bool(mc_no_saturation_data["show_progress"]),
    )
    mc_saturation_data = _get(data, "MC_SATURATION_CONFIG")
    mc_saturation_config = McSaturationConfig(
        n_iter=int(mc_saturation_data["n_iter"]),
        sampling=str(mc_saturation_data["sampling"]),
        sampling_seed=mc_saturation_data.get("sampling_seed", None),
        dt_day=float(mc_saturation_data["dt_day"]),
        max_chunk_size=int(mc_saturation_data["max_chunk_size"]),
        vol_range=_as_tuple(mc_saturation_data["vol_range"]),
        mean_press_range=_as_tuple(mc_saturation_data["mean_press_range"]),
        serp_deg_range=_as_tuple(mc_saturation_data["serp_deg_range"]),
        spacing_range=_as_tuple(mc_saturation_data["spacing_range"]),
        perm_range=_as_tuple(mc_saturation_data["perm_range"]),
        prod_rate_range=_as_tuple(mc_saturation_data["prod_rate_range"]),
        d_range=_as_tuple(mc_saturation_data["d_range"]),
        kg_rocks_range=_as_tuple(mc_saturation_data["kg_rocks_range"]),
        solubility_scaling_range=_as_tuple(mc_saturation_data["solubility_scaling_range"]),
    )
    univariate_data = _get(data, "UNIVARIATE_ANALYSIS_CONFIG")
    univariate_analysis_config = UnivariateAnalysisConfig(
        n_points=int(univariate_data["n_points"]),
        n_rep=int(univariate_data["n_rep"]),
        baseline_n_iter=int(univariate_data["baseline_n_iter"]),
        sampling=str(univariate_data["sampling"]),
        sampling_seed=univariate_data.get("sampling_seed", None),
        show_progress=bool(univariate_data["show_progress"]),
        quiet=bool(univariate_data["quiet"]),
        make_plot=bool(univariate_data["make_plot"]),
        worker_count=_maybe_none(univariate_data["worker_count"]),
    )
    convergence_data = _get(data, "MC_CONVERGENCE_SWEEP_CONFIG")
    mc_convergence_sweep_config = McConvergenceSweepConfig(
        iter_values=convergence_data["iter_values"],
        reuse_from_single_run=bool(convergence_data["reuse_from_single_run"]),
        result_column=str(convergence_data["result_column"]),
        tolerance_mean_pct=float(convergence_data["tolerance_mean_pct"]),
        tolerance_std_pct=float(convergence_data["tolerance_std_pct"]),
        save_plot=bool(convergence_data["save_plot"]),
        silence_runs=bool(convergence_data["silence_runs"]),
    )
    flow_target_data = _get(data, "MC_FLOW_TARGET_CONFIG")
    mc_flow_target_config = McFlowTargetConfig(
        n_iter=int(flow_target_data["n_iter"]),
        verbose=bool(flow_target_data["verbose"]),
        show_progress=bool(flow_target_data["show_progress"]),
        save_timeseries_plots=bool(flow_target_data["save_timeseries_plots"]),
    )

    cfg = Config(
        run_inversion=run_inversion,
        run_h2_quantification=run_h2_quantification,
        base_dir=str(_get(data, "base_dir")),
        topo_file=str(_get(data, "topo_file")),
        grav_file=str(_get(data, "grav_file")),
        mag_file=str(_get(data, "mag_file")),
        initial_model=str(_get(data, "initial_model")),
        use_initial_model=bool(_get(data, "use_initial_model")),
        background_dens=float(_get(data, "background_dens")),
        background_susc=float(_get(data, "background_susc")),
        db_dir=str(_get(data, "db_dir")),
        temperature_file=str(_get(data, "temperature_file")),
        density_file=str(_get(data, "density_file")),
        magsus_file=str(_get(data, "magsus_file")),
        save_svg=bool(_get(data, "save_svg")),
        dx=float(_get(data, "dx")),
        dy=float(_get(data, "dy")),
        dz=float(_get(data, "dz")),
        depth_core=float(_get(data, "depth_core")),
        expansion_percentage=float(_get(data, "expansion_percentage")),
        expansion_type=str(_get(data, "expansion_type")),
        dominant_side=str(_get(data, "dominant_side")),
        original_y=float(_get(data, "original_y")),
        original_x=float(_get(data, "original_x")),
        uncer_grav=float(_get(data, "uncer_grav")),
        uncer_mag=float(_get(data, "uncer_mag")),
        inclination=float(_get(data, "inclination")),
        declination=float(_get(data, "declination")),
        strength=float(_get(data, "strength")),
        unit_labels=_get(data, "unit_labels"),
        unit_dens_adj_list=_get(data, "unit_dens_adj_list"),
        unit_magsus_list=_get(data, "unit_magsus_list"),
        unit_dens_disp_list=_get(data, "unit_dens_disp_list"),
        unit_magsus_disp_list=_get(data, "unit_magsus_disp_list"),
        vol_unit_list=_get(data, "vol_unit_list"),
        serpentinite_label=int(_get(data, "serpentinite_label")),
        low_dens_adj=float(_get(data, "low_dens_adj")),
        up_dens_adj=float(_get(data, "up_dens_adj")),
        lower_susceptibility=float(_get(data, "lower_susceptibility")),
        upper_susceptibility=float(_get(data, "upper_susceptibility")),
        max_iter=int(_get(data, "max_iter")),
        max_iter_ls=int(_get(data, "max_iter_ls")),
        max_iter_cg=int(_get(data, "max_iter_cg")),
        tol_cg=float(_get(data, "tol_cg")),
        save_pred_residual_npy=bool(_get(data, "save_pred_residual_npy")),
        alpha_pgi=float(_get(data, "alpha_pgi")),
        alpha_x=float(_get(data, "alpha_x")),
        alpha_y=float(_get(data, "alpha_y")),
        alpha_z=float(_get(data, "alpha_z")),
        alpha_xx=float(_get(data, "alpha_xx")),
        alpha_yy=float(_get(data, "alpha_yy")),
        alpha_zz=float(_get(data, "alpha_zz")),
        alpha0_ratio_dens=float(_get(data, "alpha0_ratio_dens")),
        alpha0_ratio_susc=float(_get(data, "alpha0_ratio_susc")),
        beta0_ratio=float(_get(data, "beta0_ratio")),
        cooling_factor=float(_get(data, "cooling_factor")),
        tolerance=float(_get(data, "tolerance")),
        progress=float(_get(data, "progress")),
        use_chi_small=bool(_get(data, "use_chi_small")),
        chi_small=float(_get(data, "chi_small")),
        wait_till_stable=bool(_get(data, "wait_till_stable")),
        update_gmm=bool(_get(data, "update_gmm")),
        kappa=_get(data, "kappa"),
        chi0_ratio=_get(data, "chi0_ratio"),
        seed=int(_get(data, "seed")),
        use_global_seed=bool(_get(data, "use_global_seed")),
        dist_x=float(_get(data, "dist_x")),
        dist_y=float(_get(data, "dist_y")),
        dist_z=float(_get(data, "dist_z")),
        density_serpentinite=float(_get(data, "density_serpentinite")),
        porosity_front=float(_get(data, "porosity_front")),
        int_fracture_spacing=float(_get(data, "int_fracture_spacing")),
        permeability_fractures=float(_get(data, "permeability_fractures")),
        waterrockratio=float(_get(data, "waterrockratio")),
        flow_target=float(_get(data, "flow_target")),
        density_litho=float(_get(data, "density_litho")),
        gravity=float(_get(data, "gravity")),
        lithology_code=str(_get(data, "lithology_code")),
        years=float(_get(data, "years")),
        molar_mass_h2=float(_get(data, "molar_mass_h2")),
        molar_mass_h2o=float(_get(data, "molar_mass_h2o")),
        v_ref_synthetic=float(_get(data, "v_ref_synthetic")),
        t_ref_range=str(_get(data, "t_ref_range")),
        n_cores=_maybe_none(_get(data, "n_cores")),
        run_montecarlo_fault=bool(_get(data, "run_montecarlo_fault")),
        fault_mc_n_iter=int(_get(data, "fault_mc_n_iter")),
        flow_target_fracture_config=flow_target_fracture_config,
        mc_no_saturation_config=mc_no_saturation_config,
        mc_saturation_config=mc_saturation_config,
        run_univariate_analysis_no_sat=bool(_get(data, "run_univariate_analysis_no_sat")),
        run_univariate_analysis_sat=bool(_get(data, "run_univariate_analysis_sat")),
        univariate_analysis_config=univariate_analysis_config,
        run_mc_convergence_sweep=bool(_get(data, "run_mc_convergence_sweep")),
        mc_convergence_sweep_config=mc_convergence_sweep_config,
        run_analyze_limiting_factors=bool(_get(data, "run_analyze_limiting_factors")),
        flow_target_log_min=float(_get(data, "flow_target_log_min")),
        flow_target_log_max=float(_get(data, "flow_target_log_max")),
        flow_target_n_samples=int(_get(data, "flow_target_n_samples")),
        mc_flow_target_config=mc_flow_target_config,
    )

    if cfg.use_global_seed:
        seed = int(cfg.seed)
        cfg = replace(
            cfg,
            seed=seed,
            mc_no_saturation_config=replace(cfg.mc_no_saturation_config, sampling_seed=seed),
            mc_saturation_config=replace(cfg.mc_saturation_config, sampling_seed=seed),
            univariate_analysis_config=replace(cfg.univariate_analysis_config, sampling_seed=seed),
        )

    return cfg
