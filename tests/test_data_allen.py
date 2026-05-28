"""Contract tests for ccnss_helpers.data.load_allen_session.

We don't hit the real Allen warehouse from CI; instead we monkey-patch
EcephysProjectCache to return a tiny synthetic session and check shapes.
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from ccnss_helpers import data


@pytest.fixture
def fake_session():
    session = MagicMock()
    # 5 units across 2 areas
    session.units = pd.DataFrame({
        "snr": [2.0, 2.0, 2.0, 2.0, 2.0],
        "isi_violations": [0.1] * 5,
        "ecephys_structure_acronym": ["VISp", "VISp", "VISp", "VISl", "VISl"],
    }, index=[100, 101, 102, 200, 201])
    # Spike times per unit
    session.spike_times = {
        uid: np.array([0.1, 0.5, 1.0]) for uid in session.units.index
    }
    # Drifting-gratings stimulus table
    session.stimulus_presentations = pd.DataFrame({
        "stimulus_name": ["drifting_gratings"] * 4,
        "start_time": [0.0, 1.0, 2.0, 3.0],
        "stop_time": [1.0, 2.0, 3.0, 4.0],
        "orientation": [0, 45, 90, 135],
        "temporal_frequency": [2, 2, 2, 2],
    })
    return session


def test_load_allen_returns_expected_keys(fake_session, tmp_path):
    with patch("ccnss_helpers.data._open_session", return_value=fake_session):
        out = data.load_allen_session(cache_dir=tmp_path)
    assert set(out.keys()) >= {"spike_times", "units", "stim_table", "session_id"}
    assert out["session_id"] == data.ALLEN_SESSION_ID


def test_load_allen_filters_units_by_quality(fake_session, tmp_path):
    # Add a bad unit
    fake_session.units.loc[999] = [0.5, 0.9, "VISp"]  # snr=0.5 -> filtered out
    fake_session.spike_times[999] = np.array([0.0])
    with patch("ccnss_helpers.data._open_session", return_value=fake_session):
        out = data.load_allen_session(cache_dir=tmp_path)
    assert 999 not in out["units"].index
    assert len(out["units"]) == 5


def test_load_allen_keeps_only_drifting_gratings(fake_session, tmp_path):
    fake_session.stimulus_presentations.loc[10] = ["natural_scenes", 5, 6, np.nan, np.nan]
    with patch("ccnss_helpers.data._open_session", return_value=fake_session):
        out = data.load_allen_session(cache_dir=tmp_path)
    assert (out["stim_table"]["stimulus_name"] == "drifting_gratings").all()
