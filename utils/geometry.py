from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from discretize.utils import mesh_builder_xyz
from scipy.spatial import cKDTree


def build_mesh_from_topography(topo_xyz, dx, dy, dz, depth_core):
    """Create a mesh based on the given topography and parameters."""
    mesh = mesh_builder_xyz(
        xyz=topo_xyz,
        h=[dx, dy, dz],
        depth_core=depth_core,
    )
    nx, ny, nz = mesh.vnC
    return mesh, nx, ny, nz


def interpolate_nearest_neighbor(
    xyz_src: np.ndarray, values: np.ndarray, xyz_target: np.ndarray
) -> np.ndarray:
    """Perform nearest-neighbour interpolation."""
    tree = cKDTree(xyz_src)
    try:
        _, idx = tree.query(xyz_target, k=1, workers=-1)
    except TypeError:
        _, idx = tree.query(xyz_target, k=1)
    return values[idx]


def compute_temp_range_depths(temperature_mesh, xyz_mesh_temperature, temperature_ranges):
    """Compute min/max depths for each temperature range."""
    surface_level = np.max(xyz_mesh_temperature[:, 2])
    relative_depths = surface_level - xyz_mesh_temperature[:, 2]

    for temp_range in temperature_ranges:
        lower_bound, upper_bound = map(float, temp_range.split("_"))
        indices = np.where((temperature_mesh >= lower_bound) & (temperature_mesh < upper_bound))[0]

        if len(indices) > 0:
            min_temp_depth = relative_depths[indices][np.argmin(temperature_mesh[indices])]
            max_temp_depth = relative_depths[indices][np.argmax(temperature_mesh[indices])]
            temperature_ranges[temp_range] = (min_temp_depth, max_temp_depth)
        else:
            temperature_ranges[temp_range] = (np.nan, np.nan)

    return temperature_ranges


def compute_rock_volumes_and_masks(temperature_mesh, dx, dy, dz, mask_not_nan):
    """Compute serpentinite volumes and masks across temperature ranges."""
    cell_volume = (dx * dy * dz) / 1e9
    num_cells_meeting_conditions_gm = np.sum(mask_not_nan)
    volume_density_magsus = cell_volume * num_cells_meeting_conditions_gm
    temperature_ranges = [
        (100, 125),
        (125, 150),
        (150, 175),
        (175, 200),
        (200, 225),
        (225, 250),
        (250, 275),
        (275, 300),
        (300, 325),
        (325, 350),
        (350, 375),
        (375, 400),
        (400, 425),
        (425, 450),
        (450, 475),
        (475, 500),
    ]
    volume_at_temperature: Dict[str, float] = {}
    combined_masks: Dict[str, np.ndarray] = {}

    mask_temperature_total_100_500 = (temperature_mesh >= 100) & (temperature_mesh <= 500)
    combined_mask_total_100_500 = mask_not_nan & mask_temperature_total_100_500
    num_cells_meeting_conditions_total_100_500 = np.sum(combined_mask_total_100_500)
    volume_at_temperature_total_100_500 = cell_volume * num_cells_meeting_conditions_total_100_500

    for min_temp, max_temp in temperature_ranges:
        mask_temperature = (temperature_mesh >= min_temp) & (temperature_mesh <= max_temp)
        combined_mask = mask_not_nan & mask_temperature
        combined_masks[f"{min_temp}_{max_temp}"] = combined_mask
        num_cells = np.sum(combined_mask)
        volume_at_temperature[f"{min_temp}_{max_temp}"] = cell_volume * num_cells

    return (
        volume_density_magsus,
        volume_at_temperature,
        volume_at_temperature_total_100_500,
        combined_masks,
        combined_mask_total_100_500,
    )
