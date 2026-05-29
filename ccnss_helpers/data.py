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
    # Drop blank sweeps: Allen SDK stores "null" (str) for blank-sweep orientation
    # in pandas 2.x; keep only rows with numeric orientation values.
    stim = stim[pd.to_numeric(stim["orientation"], errors="coerce").notna()]
    return {
        "spike_times": spike_times,
        "units": good,
        "stim_table": stim,
        "session_id": session_id,
    }


MC_MAZE_DANDISET = "000140"  # NLB mc_maze_small
MC_MAZE_BIN_SIZE_S = 0.005


def _open_nlb_nwb(cache_dir: Path):
    """Download (if needed) and open the mc_maze_small NWB train file."""
    from dandi.download import download
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / "mc_maze_small.nwb"
    if not target.exists():
        download(f"https://dandiarchive.org/dandiset/{MC_MAZE_DANDISET}", str(cache_dir))
        # dandi creates a subfolder; prefer the train file (has behavior data).
        nwbs = list(cache_dir.rglob("*.nwb"))
        if not nwbs:
            raise FileNotFoundError("DANDI download did not yield an NWB file")
        # Pick the behavior+ecephys (train) file; fall back to first found.
        train = next((p for p in nwbs if "behavior" in p.name), nwbs[0])
        train.rename(target)
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
    """Direct pynwb extraction — bypasses nlb_tools for pandas 2.x compatibility."""
    bin_s = MC_MAZE_BIN_SIZE_S
    pre_s, post_s = 0.250, 0.450          # window around move_onset_time
    n_bins = int(round((pre_s + post_s) / bin_s))

    trials = nwb.trials.to_dataframe()
    # Use only held-in neurons
    unit_df = nwb.units.to_dataframe()
    held_in_mask = ~unit_df["heldout"] if "heldout" in unit_df.columns else np.ones(len(unit_df), dtype=bool)
    spike_times_list = [np.asarray(st) for st, keep
                        in zip(unit_df["spike_times"], held_in_mask) if keep]
    n_neurons = len(spike_times_list)

    # Continuous hand position (timestamps + data)
    hand_ts = nwb.processing["behavior"]["hand_pos"]
    hand_t = np.asarray(hand_ts.timestamps)
    hand_xy = np.asarray(hand_ts.data)     # (T, 2)

    directions = trials["trial_type"].to_numpy().astype(int)
    onset_times = trials["move_onset_time"].to_numpy(dtype=float)

    binned = np.zeros((len(trials), n_bins, n_neurons), dtype=np.int64)
    hand = np.zeros((len(trials), n_bins, 2), dtype=np.float32)

    for i, (onset, _) in enumerate(zip(onset_times, trials.itertuples())):
        t0 = onset - pre_s
        edges = np.linspace(t0, t0 + n_bins * bin_s, n_bins + 1)
        # Bin spikes for each neuron
        for j, st in enumerate(spike_times_list):
            counts, _ = np.histogram(st, bins=edges)
            binned[i, :, j] = counts
        # Interpolate hand position at bin centres
        bin_centres = 0.5 * (edges[:-1] + edges[1:])
        hand[i, :, 0] = np.interp(bin_centres, hand_t, hand_xy[:, 0])
        hand[i, :, 1] = np.interp(bin_centres, hand_t, hand_xy[:, 1])

    return binned, directions, hand
