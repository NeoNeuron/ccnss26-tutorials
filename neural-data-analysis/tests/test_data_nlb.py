import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from ccnss_helpers import data


@pytest.fixture
def fake_nwb():
    """Fake pynwb.NWBFile-like object with binned spike counts + behavior."""
    nwb = MagicMock()
    # 50 trials, 100 time bins, 20 neurons
    spikes = np.random.RandomState(0).poisson(0.5, size=(50, 100, 20))
    nwb.units = MagicMock()
    nwb.units.spike_times_index = [np.array([0.1, 0.2]) for _ in range(20)]
    # NLB-style trial-aligned binned counts under intervals "trials"
    nwb._fake_binned = spikes
    nwb._fake_directions = np.tile(np.arange(8), 7)[:50]  # 8 directions
    nwb._fake_hand = np.random.RandomState(1).randn(50, 100, 2)
    return nwb


def test_load_mc_maze_returns_expected_keys(fake_nwb, tmp_path):
    with patch("ccnss_helpers.data._open_nlb_nwb", return_value=fake_nwb):
        out = data.load_mc_maze_small(cache_dir=tmp_path)
    assert set(out.keys()) >= {"binned_spikes", "trial_directions", "hand_trajectory", "bin_size_s"}


def test_load_mc_maze_shapes(fake_nwb, tmp_path):
    with patch("ccnss_helpers.data._open_nlb_nwb", return_value=fake_nwb):
        out = data.load_mc_maze_small(cache_dir=tmp_path)
    n_trials, n_bins, n_neurons = out["binned_spikes"].shape
    assert n_trials == 50
    assert n_neurons == 20
    assert out["trial_directions"].shape == (n_trials,)
    assert out["hand_trajectory"].shape == (n_trials, n_bins, 2)


def test_load_mc_maze_bin_size_is_5ms(fake_nwb, tmp_path):
    with patch("ccnss_helpers.data._open_nlb_nwb", return_value=fake_nwb):
        out = data.load_mc_maze_small(cache_dir=tmp_path)
    assert out["bin_size_s"] == pytest.approx(0.005)
