#!/usr/bin/env bash
# Set up a Python 3.12 virtual environment for the CCNSS 2026 tutorials
# (neural data analysis + synaptic plasticity).
# Run from the repo root: bash setup_env.sh
set -e

VENV=".venv"
PYTHON="${PYTHON:-python3.12}"

echo "=== Creating venv with $PYTHON ==="
$PYTHON -m venv $VENV
source $VENV/bin/activate
pip install --upgrade pip setuptools

echo "=== Step 1: numpy (required before ssm build) ==="
pip install "numpy==1.26.4"

echo "=== Step 2: Cython (required before ssm build) ==="
pip install Cython

echo "=== Step 3: ssm (Linderman lab — needs numpy+Cython at build time) ==="
pip install --no-build-isolation "ssm @ git+https://github.com/lindermanlab/ssm.git@master"

echo "=== Step 4: core requirements ==="
pip install -r requirements.txt

echo "=== Step 5: allensdk (installed --no-deps: declares numpy<1.24 and pandas==1.5.3"
echo "            which are incompatible with Python 3.12; runtime is fine with 1.26.4) ==="
pip install allensdk --no-deps

echo "=== Step 6: nlb-tools (installed --no-deps: declares pandas<=1.3.4) ==="
pip install nlb-tools --no-deps

echo "=== Step 7: ccnss_helpers package (neural data analysis tutorial) ==="
pip install -e neural-data-analysis/

echo "=== Step 8: register Jupyter kernel ==="
pip install ipykernel
python -m ipykernel install --user --name ccnss2026 --display-name "CCNSS 2026 (Python 3.12)"

echo ""
echo "=== Done! Activate with: source $VENV/bin/activate ==="
echo "=== Run fast tests:       pytest neural-data-analysis/tests/ -k 'not notebook' ==="
echo "=== Run notebooks locally: python neural-data-analysis/scripts/run_notebook_local.py neural-data-analysis/solutions/session1_solutions.ipynb ==="
