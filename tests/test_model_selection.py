from pathlib import Path

import numpy as np
import pytest
import scanpy as sc

from paste3.model_selection import (
    convex_hull_edge_inconsistency,
    edge_inconsistency_score,
    generate_graph,
    select_overlap_fraction,
)
from paste3.paste import pairwise_align
from tests.test_paste import assert_checksum_equals

test_dir = Path(__file__).parent
input_dir = test_dir / "data/input"
output_dir = test_dir / "data/output"


def test_create_graph(slices, tmp_path):
    graph, _ = generate_graph(slices[0])

    np.savetxt(tmp_path / "create_graph_edges.csv", graph.edges, delimiter=",")
    np.savetxt(tmp_path / "create_graph_nodes.csv", graph.nodes, delimiter=",")

    assert_checksum_equals(tmp_path, "create_graph_edges.csv")
    assert_checksum_equals(tmp_path, "create_graph_nodes.csv")


def test_generate_graph_from_labels(tmp_path):
    adata = sc.read_h5ad(output_dir / "source_hull_adata.h5ad")

    graph, labels = generate_graph(adata, adata.obs["aligned"])

    np.savetxt(
        tmp_path / "generate_graph_from_labels_edges.csv", graph.edges, delimiter=","
    )
    np.savetxt(
        tmp_path / "generate_graph_from_labels_nodes.csv", graph.nodes, delimiter=","
    )

    assert_checksum_equals(tmp_path, "generate_graph_from_labels_edges.csv")
    assert_checksum_equals(tmp_path, "generate_graph_from_labels_nodes.csv")


def test_edge_inconsistency_score():
    adata = sc.read_h5ad(output_dir / "source_hull_adata.h5ad")
    # adata.obs['aligned'] used in be in str formatted boolean values
    # The code is now compatible with boolean dtypes
    # TODO: Update test data
    adata.obs["aligned"] = [bool(i) for i in adata.obs["aligned"]]
    graph, labels = generate_graph(adata, adata.obs["aligned"])
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
    measure_a = convex_hull_edge_inconsistency(slices[0], pairwise_info, axis=1)
    measure_b = convex_hull_edge_inconsistency(slices[1], pairwise_info, axis=0)
    assert measure_a == 0.18269230769230768
    assert measure_b == 0.20970873786407768


def test_select_overlap_fraction(slices):
    fraction = select_overlap_fraction(
        slices[0], slices[1], show_plot=False, numItermax=10
    )
    assert fraction == pytest.approx(0.3)
