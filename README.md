# CCNSS 2026 — Tutorials

Hands-on tutorials for CCNSS 2026, combining two tracks:

- **[neural-data-analysis/](neural-data-analysis/)** — from raw spike trains to
  population-level computations (Allen Neuropixels tuning curves → noise
  correlations → functional networks; NLB MC_Maze PCA → linear dynamical
  systems → hidden Markov models).
- **[plasticity-tutorial/](plasticity-tutorial/)** — synaptic plasticity demos
  (Oja's rule / PCA receptive fields, EGHR / ICA, and TD-learning for reaching
  and dopamine).

Each track keeps its own `notebooks/` and utility scripts; the Python
environment is shared and set up from the repo root.

## Repository layout

```
ccnss26-tutorials/
├── requirements.txt              # unified Python dependencies
├── setup_env.sh                  # one-shot environment setup (run from root)
├── LICENSE
├── neural-data-analysis/
│   ├── notebooks/                # student notebooks (Colab)
│   ├── solutions/                # solution notebooks
│   ├── scripts/                  # authoring / data utilities
│   ├── ccnss_helpers/            # installable helper package
│   ├── pyproject.toml            # defines the ccnss_helpers package
│   ├── tests/  docs/  slides/
│   └── README.md                 # track-specific details
└── plasticity-tutorial/
    ├── notebooks/                # demo1–demo4 notebooks
    └── scripts/                  # MATLAB reference scripts (.m)
```

## Quick start (local — no Anaconda required)

`setup_env.sh` uses [uv](https://github.com/astral-sh/uv) to download Python 3.12
and install all packages into an isolated `.venv/`. You do **not** need Anaconda,
Miniconda, or a pre-existing Python installation.

```bash
bash setup_env.sh          # downloads Python 3.12 + all packages (~5–10 min)
source .venv/bin/activate  # activate the env (run this at the start of each session)
jupyter notebook           # then select kernel: CCNSS 2026
```

> **First time only:** if `uv` is not installed, the script installs it automatically
> via `curl`. On subsequent runs `setup_env.sh` is a fast no-op (packages are cached).

## Quick start (Google Colab)

Open any notebook directly from GitHub in Colab.
Before running the code, CLICK **Save a copy in Drive** for tracking your package installation and your own code edits.

### neural-data-analysis

|section|notebook|
|---|---|
|session 1|[![Open Session 1 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NeoNeuron/ccnss26-tutorials/blob/main/neural-data-analysis/notebooks/session1_coding_and_networks.ipynb)|
|session 2|[![Open Session 2 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NeoNeuron/ccnss26-tutorials/blob/main/neural-data-analysis/notebooks/session2_dynamics_and_states.ipynb)|

Then **run the first cell** — it
detects Colab automatically and installs all dependencies (~10 min first run).

### synaptic-plasticity and learning

|section|notebook|
|---|---|
|demo 1|[![Open Session 1 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NeoNeuron/ccnss26-tutorials/blob/main/plasticity-tutorial/notebooks/demo1_oja_pca.ipynb)|
|demo 2|[![Open Session 2 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NeoNeuron/ccnss26-tutorials/blob/main/plasticity-tutorial/notebooks/demo2_eghr_ica.ipynb)|
|demo 3|[![Open Session 3 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NeoNeuron/ccnss26-tutorials/blob/main/plasticity-tutorial/notebooks/demo3_td_reaching.ipynb)|
|demo 4|[![Open Session 4 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NeoNeuron/ccnss26-tutorials/blob/main/plasticity-tutorial/notebooks/demo4_td_dopamine.ipynb)|

`demo1`–`demo4` only use numpy / matplotlib / scikit-learn,
which are pre-installed in Colab — no setup cell needed.

## TA

Kai Chen.
