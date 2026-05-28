import nbformat as nbf
from pathlib import Path

Path("solutions").mkdir(exist_ok=True)

nb = nbf.v4.new_notebook()
cells = []

# Cell 1: Title markdown
cells.append(nbf.v4.new_markdown_cell("""# CCNSS 2026 — Session 1: Coding and Networks

**Theme:** From single-neuron coding to network interactions.

We will analyze one Allen Visual Coding Neuropixels session in three modules:
- **1A** — Tuning curves and PSTHs
- **1B** — Signal and noise correlations
- **1C** — Functional networks and graph theory

**Time budget:** 45 minutes. Each module: ~5 min intro → 5–8 min exercise → 2 min reveal."""))

# Cell 2: Setup code cell
cells.append(nbf.v4.new_code_cell("""# @title Setup (run once) { display-mode: "form" }
!pip install -q -r https://raw.githubusercontent.com/NeoNeuron/ccnss2026-neural-data-analysis/main/requirements.txt 2>&1 | tail -5
!git clone -q https://github.com/NeoNeuron/ccnss2026-neural-data-analysis.git || (cd ccnss2026-neural-data-analysis && git pull -q)
import sys; sys.path.insert(0, "/content/ccnss2026-neural-data-analysis")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ccnss_helpers import data, plotting, save_checkpoint, load_or_compute

print("✅ Setup complete.")"""))

# Cell 3: Data load cell
cells.append(nbf.v4.new_code_cell("""# Loads ~2 GB Allen NWB on first run (~3-5 min). Subsequent cells re-use the cache.
session = data.load_allen_session()
print(f"Session {session['session_id']}: {len(session['units'])} good units across "
      f"{session['units']['ecephys_structure_acronym'].nunique()} areas")
session["units"].groupby("ecephys_structure_acronym").size()"""))

# Cell 4: Framing raster cell
cells.append(nbf.v4.new_code_cell("""# Visualize ~100 simultaneous units for 5 seconds during drifting gratings.
sample_uids = session["units"].index[:100]
sample_spikes = {uid: session["spike_times"][uid] for uid in sample_uids}
t0 = session["stim_table"]["start_time"].iloc[0]
plotting.plot_raster(sample_spikes, t_start=t0, t_end=t0 + 5)
plt.title("100 simultaneously-recorded units, 5 s during drifting gratings");"""))

nb.cells = cells
nbf.write(nb, "solutions/session1_solutions.ipynb")
print("Written solutions/session1_solutions.ipynb")
