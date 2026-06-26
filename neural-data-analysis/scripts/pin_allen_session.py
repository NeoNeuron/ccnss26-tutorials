"""Pick one Allen Visual Coding Neuropixels Brain Observatory 1.1 session.

Criteria:
- session_type == "brain_observatory_1.1"
- contains all of V1, LM, AL, PM, AM in unit metadata
- >= 400 'good' units (snr > 1, isi_violations < 0.5)

Prints the chosen session_id, its unit count per area, and the manifest path.
Run once locally; copy the printed session_id into ccnss_helpers/data.py.
"""
from pathlib import Path
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

CACHE_DIR = Path.home() / ".allen_cache"
TARGET_AREAS = {"VISp", "VISl", "VISal", "VISpm", "VISam"}  # V1=VISp etc.


def main():
    cache = EcephysProjectCache.from_warehouse(manifest=CACHE_DIR / "manifest.json")
    sessions = cache.get_session_table()
    bo_sessions = sessions[sessions["session_type"] == "brain_observatory_1.1"]

    for sid in bo_sessions.index:
        units = cache.get_session_data(sid).units
        good = units[(units["snr"] > 1) & (units["isi_violations"] < 0.5)]
        areas = set(good["ecephys_structure_acronym"].unique())
        if TARGET_AREAS.issubset(areas) and len(good) >= 400:
            print(f"CHOSEN session_id = {sid}")
            print(good.groupby("ecephys_structure_acronym").size())
            return

    raise SystemExit("No session matched criteria; relax thresholds.")


if __name__ == "__main__":
    main()
