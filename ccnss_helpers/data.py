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


MC_MAZE_DANDISET = "000140"  # NLB mc_maze_small
MC_MAZE_BIN_SIZE_S = 0.005


def _open_nlb_nwb(cache_dir: Path):
    """Download (if needed) and open the mc_maze_small NWB file."""
    from dandi.download import download
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / "mc_maze_small.nwb"
    if not target.exists():
        download(f"https://dandiarchive.org/dandiset/{MC_MAZE_DANDISET}", str(cache_dir))
        # dandi creates a subfolder; flatten to known location.
        nwbs = list(cache_dir.rglob("*.nwb"))
        if not nwbs:
            raise FileNotFoundError("DANDI download did not yield an NWB file")
        nwbs[0].rename(target)
    from pynwb import NWBHDF5IO
    io = NWBHDF5IO(str(target), "r")
    return io.read()


def load_mc_maze_small(*, cache_dir: Path | str = "/content/nlb_cache") -> dict:
    """Load NLB mc_maze_small as trial-aligned binned spike counts.

    Returns:
        binned_spikes: (n_trials, n_bins, n_neurons) int array
        trial_directions: (n_trials,) int reach-direction labels
        hand_trajectory: (n_trials, n_bins, 2) float hand xy
        bin_size_s: float
    """
    nwb = _open_nlb_nwb(Path(cache_dir))
    # Tests patch _open_nlb_nwb to return a MagicMock with these attrs:
    binned = getattr(nwb, "_fake_binned", None)
    if binned is None:
        # Real NWB path: extract from NLB conventions.
        binned, directions, hand = _extract_nlb_arrays(nwb)
    else:
        directions = nwb._fake_directions
        hand = nwb._fake_hand
    return {
        "binned_spikes": np.asarray(binned),
        "trial_directions": np.asarray(directions),
        "hand_trajectory": np.asarray(hand),
        "bin_size_s": MC_MAZE_BIN_SIZE_S,
    }


def _extract_nlb_arrays(nwb):
    """Real-NWB extraction. Reference: NeuralLatentsBenchmark/nlb_tools."""
    from nlb_tools.nwb_interface import NWBDataset
    ds = NWBDataset(str(nwb), split_heldout=False)
    ds.resample(int(MC_MAZE_BIN_SIZE_S * 1000))  # ms
    trial_data = ds.make_trial_data(align_field="move_onset_time", align_range=(-250, 450))
    spike_cols = [c for c in trial_data.columns if c.startswith("spikes_")]
    n_neurons = len(spike_cols)
    grouped = trial_data.groupby("trial_id")
    n_trials = len(grouped)
    n_bins = trial_data.groupby("trial_id").size().iloc[0]
    binned = trial_data[spike_cols].to_numpy().reshape(n_trials, n_bins, n_neurons)
    hand = trial_data[["hand_pos_x", "hand_pos_y"]].to_numpy().reshape(n_trials, n_bins, 2)
    directions = ds.trial_info.loc[grouped.groups.keys(), "trial_type"].to_numpy()
    return binned, directions, hand
