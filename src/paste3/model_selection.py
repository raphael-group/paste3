"""
This module provides functions to compute the optimal overlap percentage between
two partially overlapped slices.
"""

import logging

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from matplotlib.path import Path
from scipy.spatial import ConvexHull

from paste3.helper import (
    get_common_genes,
    glmpca_distance,
    to_dense_array,
)
from paste3.paste import pairwise_align

logger = logging.getLogger(__name__)


def generate_graph(slice, aligned_spots=None, degree=4):
    """
    Generates a graph using the networkx library where each node represents a spot
    from the given `slice` object, and edges are formed between each node and its
    closest neighbors based on spatial distance. The number of neighbors is controlled
    by the `degree` parameter.

    Parameters
    ----------
    slice : AnnData
        AnnData object containing data for a  slice.

    aligned_spots: dict, optional
        A dictionary mapping each node (spot index) to cluster labels. A cluster labeled 'True'
        means that a spot in slice A is aligned with another spot in slice B after computing
        pairwise alignment.

    degree : int, optional, default: 4
        The number of closest edges to connect each node to.

    Returns
    -------
    G : networkx.Graph
        A NetworkX graph where each node represents a spot, and each edge represents
        a connection to one of the closest neighbors. The nodes are indexed by the
        indices of the spots in the input slice.

    node_dict : dict
        A dictionary mapping each node (spot index) to its corresponding label (
        index from the `slice.obs.index`).

    """
    # Every spot will be added to the graph if alignment details is not provided
    aligned_spots = slice.obs.index if aligned_spots is None else aligned_spots

    distance = torch.cdist(
        torch.Tensor(slice.obsm["spatial"]).double(),
        torch.Tensor(slice.obsm["spatial"]).double(),
    )
    knn_spot_idx = np.argsort(distance, 1)[:, 1 : degree + 1]

    G = nx.Graph()
    for i, spots in enumerate(knn_spot_idx):
        for spot in spots:
            if slice.obs.index[int(spot)] in aligned_spots:
                G.add_edge(i, int(spot))

    return G, {n: aligned_spots[n] for n in G.nodes}


def edge_inconsistency_score(G, labels):
    """

    Compute the edge inconsistency score of a graph based on the provided labels (clusters).

    This function calculates a score that measures how inconsistent the edges in a graph are
    with respect to the cluster assignments of the nodes. The score is defined as the ratio of
    edges that connect nodes from different clusters to the total number of edges, excluding
    edges that connect nodes within the same cluster.

    Parameters
    ----------
    G : networkx.Graph
        A NetworkX graph where each node represents a spot, and each edge represents
        a connection to one of the closest neighbors.

    labels : dict
        A dictionary mapping each node (spot index) to cluster labels. A cluster labeled 'True'
        means that a spot in slice A is aligned with another spot in slice B after computing
        pairwise alignment.

    Returns
    -------
    float
        A float value representing the edge inconsistency score. A value close to 0 indicates that most edges
        are within clusters, while a value close to 1 indicates that most edges connect nodes from different clusters.
    """
    # Construct contiguity matrix that counts pairs of cluster edges
    C = pd.DataFrame(
        0, index=list(set(labels.values())), columns=list(set(labels.values()))
    )

    for edge in G.edges():
        C[labels[edge[0]]][labels[edge[1]]] += 1

    c_sum = C.values.sum()
    return float(c_sum - np.trace(C)) / c_sum


def convex_hull_edge_inconsistency(slice, pi, axis):
    """
    Computes the edge inconsistency score for a convex hull formed by the aligned spots
    in a slice, based on their probability masses (:math:`\pi`). This score reflects
    the inconsistency in edges within a subgraph of aligned spots.

    Specifically, let :math:`G = (V, E)` be a graph and let :math:`L = [l(i)]` be a labeling
    of nodes where :math:`l(i) \in {1, 2}` is the cluster label of node :math:`i`. Let
    :math:`E'` be the subset of the edges where the labelling of the nodes at the two
    ends are different, i.e. :math:`E'` is the cut of the graph. The edge inconsistency
    score is defined as :math:`H (G, L) = H(G, L) = \frac{|E'|}{|E|}`, which is the percentage
    of edges that are in the cut.

    A high inconsistency score means most of the edges are in the cut, indicating hte labeling
    of the nodes has low spatial coherence, while a low inconsistency score means that the two
    classes are nodes are mostly contiguous in graph.

    Parameters
    ----------
    slice : AnnData
        AnnData object containing data for a slice.

    pi : torch.Tensor
       Optimal transport plan for aligning two slices.

    axis : int
        The axis along which the probability mass (`pi`) is summed to determine the
        alignment status of each spot. Axis = 1 determines mass distribution for spots in
        the first slice, while axis = 0 determines mass distribution for spots in the second
        slices.

    Returns
    -------
    float
        The edge inconsistency score of the graph formed by the aligned spots. This score
        quantifies the irregularity or inconsistency of edges between aligned regions.

    """

    slice_mass = torch.sum(pi, axis=axis)
    spatial_data = slice.obsm["spatial"]

    slice.obs["aligned"] = [str(float(mass) > 0) for mass in slice_mass]
    mapped_points = [spatial_data[i] for i, mass in enumerate(slice_mass) if mass > 0]

    hull = ConvexHull(mapped_points)
    hull_path = Path(np.array(mapped_points)[hull.vertices])
    hull_adata = slice[slice.obs.index[hull_path.contains_points(spatial_data)]]

    graph, label = generate_graph(hull_adata, hull_adata.obs["aligned"])
    return edge_inconsistency_score(graph, label)


def plot_edge_curve(overlap_fractions, inconsistency_scores, ax, title):
    ax.plot(overlap_fractions, inconsistency_scores)
    ax.set_xlim(1, 0)
    ax.set_xticks([0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    ax.set_xlabel("Overlap Fraction")
    ax.set_ylabel("Edge Inconsistency Score")
    ax.set_title(title)


def select_overlap_fraction(
    a_slice, b_slice, alpha=0.1, show_plot=True, numItermax=1000
):
    """
    Selects the optimal overlap fraction between two slices, `a_slice` and
    `b_slice`, using a pairwise alignment approach. The function evaluates the
    edge inconsistency scores for different overlap fractions and estimates the
    best overlap fraction by finding the one that minimizes the edge inconsistency
    in both slices. The function also optionally visualizes the edge curves for
    both slices.

    Parameters
    ----------
    a_slice : anndata.AnnData
        AnnData object containing data for the first slice.

    b_slice : anndata.AnnData
        AnnData object containing data or the second slice.

    alpha : float, optional, default: 0.1
        Regularization parameter balancing transcriptional dissimilarity and spatial distance among aligned spots.
        Setting alpha = 0 uses only transcriptional information, while alpha = 1 uses only spatial coordinates.

    show_plot : bool, optional, default: True
        Whether to plot the edge inconsistency curves for both slices. If `True`,
        the function will display two plots: one for the source slice (`a_slice`) and
        one for the target slice (`b_slice`).

    numItermax : int, optional, default: 1000
        Maximum number of iterations for the optimization.

    Returns
    -------
    float
        The estimated overlap fraction between the two slices, representing the
        proportion of spatial overlap between the two slices. The value is between
        0 and 1, with 1 indicating a perfect overlap.

    """
    overlap_frac = np.concatenate((np.arange(0.05, 0.99, 0.05), [0.99]))

    (a_slice, b_slice), _ = get_common_genes([a_slice, b_slice])
    a_exp_dissim = to_dense_array(a_slice.X).double()
    b_exp_dissim = to_dense_array(b_slice.X).double()

    exp_dissim_matrix = glmpca_distance(a_exp_dissim, b_exp_dissim, maxIter=numItermax)

    edge_a, edge_b = [], []
    for frac in overlap_frac:
        logger.info(f"Running PASTE2 with s = {frac}.")
        pi, log = pairwise_align(
            a_slice,
            b_slice,
            overlap_fraction=frac,
            exp_dissim_matrix=exp_dissim_matrix,
            alpha=alpha,
            norm=True,
            numItermax=numItermax,
            maxIter=numItermax,
        )
        edge_a.append(convex_hull_edge_inconsistency(a_slice, pi, axis=1))
        edge_b.append(convex_hull_edge_inconsistency(b_slice, pi, axis=0))

    if show_plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        plot_edge_curve(overlap_frac, edge_a, ax=ax1, title="Source Slice")
        plot_edge_curve(overlap_frac, edge_b, ax=ax2, title="Target Slice")
        plt.show()

    half_estimate_a = overlap_frac[np.argmax(edge_a)]
    half_estimate_b = overlap_frac[np.argmax(edge_b)]

    estimated_overlap_fraction = min(2 * min(half_estimate_a, half_estimate_b), 1)
    logger.info(f"Estimation of overlap percentage is {estimated_overlap_fraction}.")
    return estimated_overlap_fraction
