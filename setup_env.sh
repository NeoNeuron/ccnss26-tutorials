#!/usr/bin/env bash
# CCNSS 2026 — environment setup
# Installs Python 3.12, all tutorial packages, and a Jupyter kernel.
# No Anaconda/Miniconda or pre-installed Python required.
#
# Run from the repo root:
#   bash setup_env.sh
#
# After it finishes:
#   source .venv/bin/activate
#   jupyter notebook
set -euo pipefail

# ── 1. Ensure uv is installed ──────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "=== Installing uv (package + Python manager) ==="
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv $(uv --version)"

# ── 2. Install Python 3.12 and create virtual environment ─────────────────────
echo ""
echo "=== Downloading Python 3.12 (skipped if already installed) ==="
uv python install 3.12

echo "=== Creating .venv ==="
uv venv --python 3.12 .venv
# shellcheck source=/dev/null
source .venv/bin/activate

# ── 3. Build-time prereqs for ssm ─────────────────────────────────────────────
echo ""
echo "=== Installing numpy 1.26.4 + Cython + setuptools (build deps for ssm) ==="
uv pip install "numpy==1.26.4" Cython setuptools

# ── 4. ssm (Linderman lab HMM library — compiled from source) ─────────────────
# Requires numpy + Cython already present in the build env (hence --no-build-isolation).
echo ""
echo "=== Installing ssm from GitHub (~3 min) ==="
uv pip install --no-build-isolation \
    "ssm @ git+https://github.com/lindermanlab/ssm.git@master"

# ── 5. Core tutorial requirements ─────────────────────────────────────────────
echo ""
echo "=== Installing core requirements ==="
uv pip install -r requirements.txt

# ── 6. allensdk — bypass stale numpy<1.24 / pandas==1.5.3 / scipy<1.11 pins ──
# Install with --no-deps then add its actual runtime requirements manually.
echo ""
echo "=== Installing allensdk (--no-deps to bypass stale version pins) ==="
uv pip install allensdk --no-deps
uv pip install \
    SimpleITK \
    xarray \
    simplejson \
    nest-asyncio \
    psycopg2-binary \
    pynrrd \
    future \
    requests-toolbelt \
    scikit-image \
    statsmodels \
    seaborn \
    "marshmallow<4.0.0" \
    argschema \
    boto3 \
    semver \
    cachetools \
    sqlalchemy

# ── 7. nlb-tools — bypass stale pandas<=1.3.4 pin ────────────────────────────
echo ""
echo "=== Installing nlb-tools (--no-deps to bypass stale version pins) ==="
uv pip install nlb-tools --no-deps

# ── 8. Local helper package ────────────────────────────────────────────────────
echo ""
echo "=== Installing ccnss_helpers (local, editable) ==="
uv pip install -e neural-data-analysis/

# ── 9. Register Jupyter kernel ─────────────────────────────────────────────────
echo ""
echo "=== Registering Jupyter kernel 'CCNSS 2026' ==="
uv pip install ipykernel
python -m ipykernel install --user --name ccnss2026 --display-name "CCNSS 2026"

# ── 10. Smoke test ─────────────────────────────────────────────────────────────
echo ""
echo "=== Smoke test ==="
python - <<'PYEOF'
try:
    from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
    import ssm
    from ccnss_helpers import data, plotting
    print("✅  All imports OK.")
except Exception as e:
    print("❌  Import failed:", e)
    raise
PYEOF

# ── Done ───────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ✅  Setup complete!                                     ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  To activate this environment in future sessions:        ║"
echo "║      source .venv/bin/activate                           ║"
echo "║                                                          ║"
echo "║  Then launch Jupyter:                                    ║"
echo "║      jupyter notebook                                    ║"
echo "║                                                          ║"
echo "║  Inside Jupyter, select kernel: CCNSS 2026              ║"
echo "╚══════════════════════════════════════════════════════════╝"
