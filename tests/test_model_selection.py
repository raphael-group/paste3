import numpy as np
from pathlib import Path
import pytest
from tests.test_paste import assert_checksum_equals
from paste2.model_selection import (
    create_graph,
    generate_graph_from_labels,
    edge_inconsistency_score,
    calculate_convex_hull_edge_inconsistency,
    select_overlap_fraction,
    select_overlap_fraction_plotting,
)
from matplotlib import pyplot as plt
import scanpy as sc
from paste2.PASTE2 import partial_pairwise_align

test_dir = Path(__file__).parent
input_dir = test_dir / "data/input"
output_dir = test_dir / "data/output"


def test_create_graph(slices, tmp_path):
    graph, _ = create_graph(slices[0])

    np.savetxt(tmp_path / "create_graph_edges.csv", graph.edges, delimiter=",")
    np.savetxt(tmp_path / "create_graph_nodes.csv", graph.nodes, delimiter=",")

    assert_checksum_equals(tmp_path, "create_graph_edges.csv")
    assert_checksum_equals(tmp_path, "create_graph_nodes.csv")


def test_generate_graph_from_labels(tmp_path):
    adata = sc.read_h5ad(output_dir / "source_hull_adata.h5ad")

    graph, labels = generate_graph_from_labels(adata, adata.obs["aligned"])

    np.savetxt(
        tmp_path / "generate_graph_from_labels_edges.csv", graph.edges, delimiter=","
    )
    np.savetxt(
        tmp_path / "generate_graph_from_labels_nodes.csv", graph.nodes, delimiter=","
    )

    assert_checksum_equals(tmp_path, "generate_graph_from_labels_edges.csv")
    assert_checksum_equals(tmp_path, "generate_graph_from_labels_nodes.csv")


def test_edge_inconsistency_score(slices, tmp_path):
    adata = sc.read_h5ad(output_dir / "source_hull_adata.h5ad")

    graph, labels = generate_graph_from_labels(adata, adata.obs["aligned"])
    measure_a = edge_inconsistency_score(graph, labels)
    assert measure_a == 0.0


# TODO: need to figure out where the randomness is coming from in the following three functions
# TODO: they also take a long time to work
@pytest.mark.skip
def test_calculate_convex_hull_edge_inconsistency(slices, tmp_path):
    pairwise_info = partial_pairwise_align(slices[0], slices[1], s=0.7)
    measure_a, measure_b = calculate_convex_hull_edge_inconsistency(
        slices[0], slices[1], pairwise_info
    )

    assert measure_a == 0.17177914110429449
    assert measure_b == 0.18404907975460122


@pytest.mark.skip
def test_select_overlap_fraction(slices):
    fraction = select_overlap_fraction(slices[0], slices[1])
    assert fraction == 0.5


@pytest.mark.skip
def test_select_overlap_fraction_plotting(slices):
    fraction = select_overlap_fraction_plotting(slices[0], slices[1])

    assert fraction == 0.4