import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.figure
from ccnss_helpers import plotting
import networkx as nx


def test_plot_raster_returns_figure():
    spike_times = {0: np.array([0.1, 0.4, 0.7]), 1: np.array([0.2, 0.5])}
    fig = plotting.plot_raster(spike_times, t_start=0.0, t_end=1.0)
    assert isinstance(fig, matplotlib.figure.Figure)
    ax = fig.axes[0]
    # Each unit should produce one EventCollection.
    assert len(ax.collections) == 2


def test_plot_psth_returns_figure():
    bin_centers = np.linspace(0, 1, 50)
    rates = np.random.RandomState(0).rand(50)
    fig = plotting.plot_psth(bin_centers, rates, label="unit 42")
    assert isinstance(fig, matplotlib.figure.Figure)
    ax = fig.axes[0]
    assert ax.get_xlabel() != ""
    assert ax.get_ylabel() != ""


def test_plot_network_returns_figure():
    G = nx.erdos_renyi_graph(20, 0.2, seed=0)
    fig = plotting.plot_network(G, node_color="black")
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_latents_3d_returns_figure():
    latents = np.random.RandomState(0).randn(8, 100, 3)  # conditions x time x dims
    fig = plotting.plot_latents_3d(latents)
    assert isinstance(fig, matplotlib.figure.Figure)
    # Should be a single 3D axis.
    assert fig.axes[0].name == "3d"
