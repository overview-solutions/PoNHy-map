#!/usr/bin/env bash
# Run PoNHy using the project venv so all dependencies are available.
# Usage: ./run_ponhy.sh   (from the PoNHy directory)
#
# By default this script limits CPU usage to reduce risk of overload/crashes.
# To allow more cores: PONHY_MAX_CORES=8 ./run_ponhy.sh
# To disable limits: PONHY_UNLIMITED=1 ./run_ponhy.sh

set -e
cd "$(dirname "$0")"

if [[ ! -d .venv ]]; then
  echo "No .venv found. Create it with:"
  echo "  python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

source .venv/bin/activate
export MPLCONFIGDIR="${MPLCONFIGDIR:-.venv/mplconfig}"

# Limit parallelism by default to avoid overloading the machine (inversion + MC are very heavy).
if [[ -z "$PONHY_UNLIMITED" ]]; then
  max_cores="${PONHY_MAX_CORES:-2}"
  export LOKY_MAX_CPU_COUNT="$max_cores"
  export OMP_NUM_THREADS="$max_cores"
  export OPENBLAS_NUM_THREADS="$max_cores"
  export MKL_NUM_THREADS="$max_cores"
  export NUMEXPR_NUM_THREADS="$max_cores"
fi

python ponhy.py "$@"
