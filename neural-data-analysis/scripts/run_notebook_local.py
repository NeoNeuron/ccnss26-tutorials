"""Execute a solutions notebook locally, overriding the Colab /content paths."""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

LOCAL_CACHE = Path.home() / ".ccnss_cache"
OVERRIDE_CELL = f"""\
# --- LOCAL EXECUTION OVERRIDE (injected by run_notebook_local.py) ---
import ccnss_helpers.data as _d, functools as _ft, pathlib as _pl
_allen_cache = _pl.Path("{LOCAL_CACHE}/allen")
_nlb_cache   = _pl.Path("{LOCAL_CACHE}/nlb")
_allen_cache.mkdir(parents=True, exist_ok=True)
_nlb_cache.mkdir(parents=True, exist_ok=True)
_d.load_allen_session  = _ft.partial(_d.load_allen_session,  cache_dir=_allen_cache)
_d.load_mc_maze_small  = _ft.partial(_d.load_mc_maze_small,  cache_dir=_nlb_cache)
_checkpoint_dir = _pl.Path("{LOCAL_CACHE}/checkpoints")
_checkpoint_dir.mkdir(parents=True, exist_ok=True)
import ccnss_helpers._checkpoint as _cp
_cp.DEFAULT_ROOT = _checkpoint_dir
print("Local cache:", _allen_cache, "|", _nlb_cache)
"""


def _strip_shell_magic(source: str) -> str:
    """Remove lines starting with ! (shell magic) from a cell source."""
    lines = [ln for ln in source.splitlines(keepends=True) if not ln.lstrip().startswith("!")]
    return "".join(lines)


def inject_and_run(nb_path: Path, kernel: str, timeout: int) -> int:
    with open(nb_path) as f:
        nb = json.load(f)

    # Strip !pip / !git / other shell magic from all code cells to avoid Colab setup re-runs
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            cleaned = _strip_shell_magic(src)
            cell["source"] = cleaned

    override = {
        "cell_type": "code",
        "id": str(uuid.uuid4())[:8],
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": OVERRIDE_CELL,
    }
    # Insert right after the first cell (usually markdown title)
    nb["cells"].insert(1, override)

    with tempfile.NamedTemporaryFile(suffix=".ipynb", delete=False, mode="w") as tmp:
        json.dump(nb, tmp)
        tmp_path = Path(tmp.name)

    try:
        cmd = [
            sys.executable, "-m", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--inplace",
            f"--ExecutePreprocessor.kernel_name={kernel}",
            f"--ExecutePreprocessor.timeout={timeout}",
            str(tmp_path),
        ]
        rc = subprocess.run(cmd).returncode
        if rc == 0:
            # Strip the injected override cell before saving back to the source notebook
            with open(tmp_path) as f:
                executed_nb = json.load(f)
            executed_nb["cells"] = [
                c for c in executed_nb["cells"]
                if "LOCAL EXECUTION OVERRIDE" not in "".join(c.get("source", []))
            ]
            with open(nb_path, "w") as f:
                json.dump(executed_nb, f, indent=1)
            print(f"Outputs saved back to {nb_path}")
        return rc
    finally:
        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("notebook", type=Path)
    p.add_argument("--kernel", default="ccnss2026")
    p.add_argument("--timeout", type=int, default=1800)
    args = p.parse_args()
    sys.exit(inject_and_run(args.notebook, args.kernel, args.timeout))
