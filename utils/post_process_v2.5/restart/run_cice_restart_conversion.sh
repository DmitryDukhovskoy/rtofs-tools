#!/bin/bash
set -euo pipefail

# ==============================================================================
# 0. Self-Contained Machine Detection
# ==============================================================================
HOSTNAME_F=$(hostname -f)
MACHINE_ID="UNKNOWN"

case $HOSTNAME_F in
    clogin*|dlogin*) MACHINE_ID="wcoss2" ;;
    ufe*)            MACHINE_ID="ursa" ;;
esac

if [[ "$MACHINE_ID" == "UNKNOWN" ]]; then
    echo "FATAL: This pipeline is only supported on WCOSS2 (clogin/dlogin) or Ursa (ufe)."
    echo "Detected hostname: $HOSTNAME_F"
    exit 1
fi

# ==============================================================================
# 1. Inputs & Path Construction
# ==============================================================================
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <IN_DIR> <OUT_DIR> <YYYYMMDD> [optional_args]"
    echo "Example: $0 /path/to/restart_in /path/to/restart_out 20260901 --ktherm 2"
    exit 1
fi

IN_DIR=$1
OUT_DIR=$2
TARGET_DATE=$3
shift 3 # Shift arguments so any extras (like --ktherm 2) can be passed directly to python

# Format Date for CICE6 file naming
YYYY=${TARGET_DATE:0:4}
MM=${TARGET_DATE:4:2}
DD=${TARGET_DATE:6:2}

INFILE="${IN_DIR}/rtofs_glo.t00z.n00.restart_cice"
OUTFILE="${OUT_DIR}/iced.${YYYY}-${MM}-${DD}-00000.nc"

if [ ! -f "$INFILE" ]; then
    echo "FATAL: Input file not found -> $INFILE"
    exit 1
fi

if [ ! -d "$OUT_DIR" ]; then
    echo "Creating output directory: $OUT_DIR"
    mkdir -p "$OUT_DIR"
fi

# ==============================================================================
# 2. Environment Setup (Python & Modules)
# ==============================================================================
echo "=================================================="
echo "Machine Detected: $MACHINE_ID ($HOSTNAME_F)"
echo "Input File      : $INFILE"
echo "Output File     : $OUTFILE"
echo "Setting up Python Environment..."
echo "=================================================="

module purge
module load python/3.11

# Find where the script actually lives to locate the python script and venv
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
VENV_DIR="${SCRIPT_DIR}/rtofs_venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating isolated virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    
    echo "Installing required Python packages (numpy, netCDF4)..."
    pip install --upgrade pip
    pip install numpy netCDF4
else
    source "$VENV_DIR/bin/activate"
fi

# ==============================================================================
# 3. Execution
# ==============================================================================
echo "Executing CICE conversion..."
python3 "${SCRIPT_DIR}/convert_cice_restart.py" "$INFILE" "$OUTFILE" "$TARGET_DATE" "$@"
