"""Plotting utilities for the CCNSS 2026 tutorial."""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_raster(
    spike_times: dict[int, np.ndarray],
    *,
    t_start: float | None = None,
    t_end: float | None = None,
    ax=None,
) -> Figure:
    """Spike raster. spike_times maps unit_id -> 1D array of spike times (s)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure
    for row, (uid, spikes) in enumerate(spike_times.items()):
        if t_start is not None:
            spikes = spikes[spikes >= t_start]
        if t_end is not None:
            spikes = spikes[spikes <= t_end]
        ax.eventplot(spikes, lineoffsets=row, linelengths=0.8, colors="black")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Unit")
    ax.set_yticks(range(len(spike_times)))
    ax.set_yticklabels(list(spike_times.keys()))
    if t_start is not None and t_end is not None:
        ax.set_xlim(t_start, t_end)
    return fig


def plot_psth(
    bin_centers: np.ndarray,
    rates: np.ndarray,
    *,
    label: str | None = None,
    ax=None,
) -> Figure:
    """Plot one PSTH curve."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3))
    else:
        fig = ax.figure
    ax.plot(bin_centers, rates, label=label)
    ax.set_xlabel("Time relative to stimulus onset (s)")
    ax.set_ylabel("Firing rate (Hz)")
    if label:
        ax.legend()
    return fig


def plot_network(
    G,
    *,
    node_color="C0",
    node_size: int = 30,
    edge_alpha: float = 0.4,
    ax=None,
) -> Figure:
    """Plot a networkx graph with spring layout."""
    import networkx as nx
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure
    pos = nx.spring_layout(G, seed=0)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_color, node_size=node_size)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=edge_alpha)
    ax.set_axis_off()
    return fig


def plot_latents_3d(
    latents: np.ndarray,
    *,
    cmap: str = "hsv",
    ax=None,
) -> Figure:
    """3D latent trajectories, colored per condition.

    latents shape: (n_conditions, n_time, n_dims >= 3).
    """
    import matplotlib
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure
    n_cond = latents.shape[0]
    colors = matplotlib.colormaps[cmap].resampled(n_cond)
    for i in range(n_cond):
        ax.plot(latents[i, :, 0], latents[i, :, 1], latents[i, :, 2], color=colors(i))
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    return fig
