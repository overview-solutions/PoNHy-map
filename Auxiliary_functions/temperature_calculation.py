import fipy as fp
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from scipy.interpolate import interp1d
import random
from typing import Any, cast
import sys


def _select_data_dir() -> str:
    search_roots = [os.getcwd(), os.path.dirname(__file__), os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))]
    candidates = []
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            if name.lower().startswith("data"):
                path = os.path.join(root, name)
                if os.path.isdir(path):
                    candidates.append(path)

    # Remove duplicates while preserving order.
    seen = set()
    unique_candidates = []
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        unique_candidates.append(path)

    if not unique_candidates:
        raise FileNotFoundError("No Data/data folders found near the script or current directory.")
    if len(unique_candidates) == 1:
        return unique_candidates[0]

    print("\n[Temperature] Data folders found:")
    for idx, path in enumerate(unique_candidates, start=1):
        print(f"  {idx}) {path}")

    if not sys.stdin.isatty():
        raise RuntimeError("Multiple Data folders found but no interactive terminal to select one.")

    while True:
        choice = input("Select Data folder (number -> Enter): ").strip()
        if not choice:
            return unique_candidates[0]
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(unique_candidates):
                return unique_candidates[idx - 1]
        print("Invalid selection. Try again.")


def _resolve_topo_file(base_dir: str) -> str:
    topo_candidates = [
        os.path.join(base_dir, "Ext_Topo.txt"),
        os.path.join(base_dir, "Ext_Topo.csv"),
    ]
    for path in topo_candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "No Ext_Topo.txt or Ext_Topo.csv found in the selected Data folder."
    )


# Files to work with
topo_arg = sys.argv[1] if len(sys.argv) > 1 else None
if topo_arg:
    topo_file = topo_arg
    if not os.path.exists(topo_file):
        raise FileNotFoundError(f"Topo file not found: {topo_file}")
    if not topo_file.lower().endswith((".txt", ".csv")):
        raise ValueError("Topo file must be .txt or .csv")
    dir_path = os.path.dirname(topo_file) + os.path.sep
else:
    dir_path = _select_data_dir() + os.path.sep
    topo_file = _resolve_topo_file(dir_path)

generate_plots = False
if sys.stdin.isatty():
    response = input("Generate plots? [y/N]: ").strip().lower()
    generate_plots = response in {"y", "yes"}

def load_topo_xyz(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    if not lines:
        raise ValueError(f"{path} is empty.")

    first_line = lines[0]
    has_header = any(char.isalpha() for char in first_line)
    sample_line = lines[1] if has_header and len(lines) > 1 else first_line

    if "," in sample_line:
        delimiter = ","
    elif "\t" in sample_line:
        delimiter = "\t"
    else:
        delimiter = None  # split on any whitespace

    skip_rows = 1 if has_header else 0
    return np.loadtxt(str(path), delimiter=delimiter, skiprows=skip_rows)

topo_xyz = load_topo_xyz(str(topo_file))

output_dir = dir_path

# Extract columns
x_csv = topo_xyz[:, 0]
y_csv = topo_xyz[:, 1]
z_csv = topo_xyz[:, 2]
max_elevation = z_csv.max()
max_depth = 20000 + max_elevation

temp_sup = 15.0  # Surface temperature in Celsius degrees
heat_flow = 0.075  # Background heat flow in W/m² obtained from the International Heat Flow Database

dz_uniform = 50.0 # Cell size
nz = int(max_depth // dz_uniform)  # Number of cells
mesh = fp.Grid1D(nx=nz, dx=dz_uniform)

T = fp.CellVariable(name="temperature", mesh=mesh, value=temp_sup)  # Initial temperature

# Define intervals and conductivity values
intervals = [(0, 10), (10, 5000), (5000, 15000), (15000, max_depth)]
conductivity_values = [1.1, 2.50, 3.00, 3.50, 4.00]  # Corresponding conductivities

# Thermal conductivity function
def thermal_conductivity(z, intervals=intervals, values=conductivity_values):
    return (
        (values[0] + (values[1] - values[0]) * (z - intervals[0][0]) / (intervals[0][1] - intervals[0][0])) * (z <= intervals[0][1]) +
        (values[1] + (values[2] - values[1]) * (z - intervals[1][0]) / (intervals[1][1] - intervals[1][0])) * ((z > intervals[1][0]) & (z <= intervals[1][1])) +
        (values[2] + (values[3] - values[2]) * (z - intervals[2][0]) / (intervals[2][1] - intervals[2][0])) * ((z > intervals[2][0]) & (z <= intervals[2][1])) +
        (values[3] + (values[4] - values[3]) * (z - intervals[3][0]) / (intervals[3][1] - intervals[3][0])) * ((z > intervals[3][0]) & (z <= intervals[3][1])) +
        values[4] * (z > intervals[3][1])  # Constant conductivity at 4.5 W/mK beyond last interval
    )

# Create a uniform mesh with x-meter intervals up to max depth
z_uniform_mesh = np.arange(0, max_depth + dz_uniform, dz_uniform)

# Plot conductivity with depth (optional)
conductivities = [thermal_conductivity(z) for z in z_uniform_mesh]
if generate_plots:
    plt.figure(figsize=(5, 10))
    plt.plot(conductivities, z_uniform_mesh, label='Thermal Conductivity Profile', color='green')
    plt.xlabel('Thermal Conductivity (W/mK)')
    plt.ylabel('Depth (m)')
    plt.title('Thermal Conductivity with Depth')
    plt.gca().invert_yaxis()  # Invert y-axis for depth
    plt.grid(True)
    plt.legend()
    # Save the plot as PNG in the output directory
    plt.savefig(os.path.join(output_dir, "thermal_conductivity_profile.png"), dpi=300)
    plt.show()


n_simulations = 50  # Number of simulations for uncertainty analysis
all_temperatures = []
for i in range(n_simulations):
    # Apply random perturbation to conductivity values and heat flow
    perturbed_conductivity_values = [value + random.uniform(-0.3, 0.3) for value in conductivity_values]  # Fixed value perturbation
    perturbed_heat_flow = heat_flow + random.uniform(-0.01, 0.01)  # Fixed value perturbation
   
    k_face = thermal_conductivity(mesh.faceCenters[0], values=perturbed_conductivity_values)

    T.setValue(temp_sup)  # Reset temperature values
    eq = fp.DiffusionTerm(coeff=k_face)

    T.constrain(temp_sup, mesh.facesRight)  # Dirichlet boundary condition at the top
    flux_left = (-perturbed_heat_flow, )  # Neumann boundary condition at the bottom
    T.faceGrad.constrain(flux_left, mesh.facesLeft)

    eq.solve(var=T)

    # Invert the temperature values
    max_temp = T.value.max()
    min_temp = T.value.min()
    inverted_T_values = max_temp + min_temp - T.value

    # Interpolation function based on original depth and inverted temperature
    original_z_values = mesh.cellCenters[0]
    interp_func = interp1d(
        original_z_values,
        inverted_T_values,
        kind='linear',
        fill_value=cast(Any, "extrapolate"),
    )
    interpolated_temperature = interp_func(z_uniform_mesh)

    all_temperatures.append(interpolated_temperature)

# Calculate mean and standard deviation for uncertainty analysis
all_temperatures = np.array(all_temperatures)
mean_temperature = np.mean(all_temperatures, axis=0)
std_temperature = np.std(all_temperatures, axis=0)

# Plot temperature with uncertainty (optional)
if generate_plots:
    plt.figure(figsize=(5, 10))
    plt.plot(mean_temperature, z_uniform_mesh, label='Mean Temperature Profile', color='blue')
    plt.fill_betweenx(z_uniform_mesh, mean_temperature - std_temperature, mean_temperature + std_temperature, color='lightblue', alpha=0.5, label='Uncertainty Range (1σ)')

    plt.xlabel('Temperature (°C)')
    plt.ylabel('Depth (m)')
    plt.title('Temperature Profile with Uncertainty')
    plt.gca().invert_yaxis()  # Invert y-axis for depth
    plt.grid(True)
    plt.legend()
    # Save the plot as PNG in the output directory
    plt.savefig(os.path.join(output_dir, "temperature_profile_with_uncertainty.png"), dpi=300)
    plt.show()

# Function to export temperature to CSV with decimal notation
def export_topography_aligned_temperature_to_csv_v2(x_csv, y_csv, z_csv, z_values, T, max_depth_adjusted, file_path):
    export_data_list = []
    x_unique = np.unique(x_csv)
    y_unique = np.unique(y_csv)
    
    for x in x_unique:
        for y in y_unique:
            filtered_z = z_csv[np.isclose(x_csv, x, atol=1e-5) & np.isclose(y_csv, y, atol=1e-5)]
            if len(filtered_z) > 0:
                topo_z = filtered_z[0]
            else:
                continue
            adjusted_z_values = topo_z - z_values
            valid_indices = np.where(adjusted_z_values >= (topo_z - max_depth_adjusted))
            valid_z_values = adjusted_z_values[valid_indices]
            valid_temperature_values = T[valid_indices]
            for z, temp in zip(valid_z_values, valid_temperature_values):
                export_data_list.append([x, y, z, temp])

    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["X", "Y", "Z", "Temperature"])
        for row in export_data_list:
            csvwriter.writerow(row)

    return export_data_list

default_output_name = "Temperature_model_new.csv"
if sys.stdin.isatty():
    name_input = input(f"Output filename (default: {default_output_name}): ").strip()
    output_name = name_input or default_output_name
else:
    output_name = default_output_name
export_file_path = os.path.join(output_dir, output_name)
exported_data = export_topography_aligned_temperature_to_csv_v2(x_csv, y_csv, z_csv, z_uniform_mesh, mean_temperature, max_depth, export_file_path)
print(f"Temperature distribution saved to: {export_file_path}")

# Calculate the weighted average temperature gradient
gradients = []
weighted_sum = 0
total_length = 0
for i, (start, end) in enumerate(intervals):
    if i < len(intervals) - 1:
        mask = (z_uniform_mesh >= start) & (z_uniform_mesh < end)
    else:
        mask = (z_uniform_mesh >= start)
    
    if np.any(mask):
        z_interval = z_uniform_mesh[mask]
        T_interval = mean_temperature[mask]
        if len(z_interval) > 1:  # Ensure there are enough points to calculate a gradient
            gradient = abs((T_interval[-1] - T_interval[0]) / (z_interval[-1] - z_interval[0]))
            gradients.append(gradient * 1000)  # Convert gradient to °C/km
            length = z_interval[-1] - z_interval[0]
            weighted_sum += gradient * length
            total_length += length

# Calculate average gradient
weighted_average_gradient = (weighted_sum / total_length) * 1000  # Convert to °C/km

# Print the weighted average gradient
print(f'Weighted average temperature gradient: {weighted_average_gradient:.2f} °C/km')






