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

## Quick start (local)

```bash
# Creates a Python 3.12 venv, installs everything, registers a Jupyter kernel.
bash setup_env.sh
source .venv/bin/activate
```

Or, minimally:

```bash
pip install -r requirements.txt
pip install -e neural-data-analysis/      # for the ccnss_helpers package
```

## Quick start (Google Colab)

The neural-data-analysis notebooks are Colab-ready — see
[neural-data-analysis/README.md](neural-data-analysis/README.md) for the launch
badges and session schedule.

## TA

Kai Chen.
