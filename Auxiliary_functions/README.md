# Auxiliary: Temperature Calculation (Pyrenees)

This directory contains the script `Temperature_Calculation_Pyrenees.py`, which computes a 1D temperature profile with uncertainty for thermal models in the Pyrenees using a vertical mesh and background heat flow. The script:

- Reads topography $(X, Y, Z)$ from a `.txt` or `.csv` file.
- Builds a thermal conductivity profile with depth.
- Runs $N$ simulations with random perturbations to conductivity and heat flow.
- Computes the mean temperature profile and its uncertainty ($1\sigma$).
- Exports results to a `.csv` aligned with the topography, applying the temperature model across all $(X, Y)$ columns in the input grid.
- Saves plots in `Ext_Data/`.

## Requirements

Use the Conda environment defined in `environment.yml`.

## Installation (new environment)

1. Create the environment from `environment.yml`:

```bash
conda env create -f environment.yml
```

2. Activate the environment:

```bash
conda activate temp_calc
```

## Usage

### Input

The script expects a topography file with $(X, Y, Z)$ columns. You can pass the file path as an argument:

```bash
python Temperature_Calculation_Pyrenees.py /path/to/Ext_Topo.txt
```

If no argument is provided, it uses the default path defined inside the script.

### Outputs

An `Ext_Data` directory is created next to the input file with:

- `thermal_conductivity_profile.png`: conductivity profile vs depth.
- `temperature_profile_with_uncertainty.png`: mean temperature profile with uncertainty band.
- `Temperature_model_new.csv`: exported temperatures by $(X, Y, Z)$.

## Notes

- Adjust `temp_sup`, `heat_flow`, `dz_uniform`, and the conductivity intervals inside the script for alternate scenarios.
- The number of simulations is controlled by `n_simulations`.
