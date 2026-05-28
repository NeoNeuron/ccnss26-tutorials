"""Smoke-test: both solution notebooks execute end-to-end.

This runs locally / in CI with the data caches pre-warmed. In CI we skip
the heavy Allen + NLB downloads using env var SKIP_DOWNLOAD=1; cache files
must exist under ~/.allen_cache and ~/.nlb_cache (downloaded once on the runner).
"""
import os
import subprocess
import pytest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize("nb", [
    "solutions/session1_solutions.ipynb",
    "solutions/session2_solutions.ipynb",
])
def test_notebook_executes(nb, tmp_path):
    if os.environ.get("SKIP_NOTEBOOK_TESTS"):
        pytest.skip("notebook smoke tests skipped")
    out = tmp_path / "out.ipynb"
    result = subprocess.run(
        ["jupyter", "nbconvert", "--to", "notebook", "--execute",
         "--ExecutePreprocessor.timeout=600",
         str(REPO / nb), "--output", str(out)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
