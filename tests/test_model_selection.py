from pathlib import Path

import numpy as np
import scanpy as sc

from paste3.model_selection import (
    calculate_convex_hull_edge_inconsistency,
    create_graph,
    edge_inconsistency_score,
    generate_graph_from_labels,
    select_overlap_fraction,
)
from paste3.paste import pairwise_align

test_dir = Path(__file__).parent / "data"
input_dir = test_dir / "input"
output_dir = test_dir / "output"


def test_create_graph(slices):
    graph, _ = create_graph(slices[0])

    expected_result = np.load(test_dir / "create_graph.npz")

    assert np.allclose(expected_result["edges"], graph.edges)
    assert np.allclose(expected_result["nodes"], graph.nodes)


def test_generate_graph_from_labels():
    adata = sc.read_h5ad(output_dir / "source_hull_adata.h5ad")

    graph, labels = generate_graph_from_labels(adata, adata.obs["aligned"])

    expected_result = np.load(test_dir / "generate_graph_from_labels.npz")

    assert np.allclose(expected_result["edges"], graph.edges)
    assert np.allclose(expected_result["nodes"], graph.nodes)


def test_edge_inconsistency_score():
    adata = sc.read_h5ad(output_dir / "source_hull_adata.h5ad")

    graph, labels = generate_graph_from_labels(adata, adata.obs["aligned"])
    measure_a = edge_inconsistency_score(graph, labels)
    assert measure_a == 0.0


def test_calculate_convex_hull_edge_inconsistency(slices):
    pairwise_info, _ = pairwise_align(
        slices[0],
        slices[1],
        overlap_fraction=0.7,
        exp_dissim_metric="glmpca",
        norm=True,
        maxIter=10,
    )
    measure_a, measure_b = calculate_convex_hull_edge_inconsistency(
        slices[0], slices[1], pairwise_info
    )
    assert measure_a == 0.18269230769230768
    assert measure_b == 0.20970873786407768


def test_select_overlap_fraction(slices):
    fraction = select_overlap_fraction(
        slices[0], slices[1], show_plot=False, numItermax=10
    )
    assert fraction == 0.3
