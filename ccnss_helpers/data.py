"""Data loaders for Allen Neuropixels and NLB MC_Maze."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

# TENTATIVE — verify by running scripts/pin_allen_session.py on a machine
# with internet + AllenSDK installed.
ALLEN_SESSION_ID: int = 715093703

UNIT_SNR_MIN = 1.0
UNIT_ISI_MAX = 0.5


def _open_session(session_id: int, cache_dir: Path):
    """Open one Allen session via EcephysProjectCache. Indirection for tests."""
    from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = EcephysProjectCache.from_warehouse(manifest=cache_dir / "manifest.json")
    return cache.get_session_data(session_id)


def load_allen_session(
    *,
    session_id: int = ALLEN_SESSION_ID,
    cache_dir: Path | str = "/content/allen_cache",
) -> dict:
    """Load one Allen Visual Coding Neuropixels session.

    Returns a dict with keys:
        spike_times: {unit_id: np.ndarray of spike times in seconds}
        units: pd.DataFrame (filtered to good units)
        stim_table: pd.DataFrame (drifting_gratings presentations only)
        session_id: int
    """
    session = _open_session(session_id, Path(cache_dir))
    units = session.units
    good = units[(units["snr"] > UNIT_SNR_MIN) & (units["isi_violations"] < UNIT_ISI_MAX)]
    spike_times = {uid: np.asarray(session.spike_times[uid]) for uid in good.index}
    stim = session.stimulus_presentations
    stim = stim[stim["stimulus_name"] == "drifting_gratings"]
    return {
        "spike_times": spike_times,
        "units": good,
        "stim_table": stim,
        "session_id": session_id,
    }
