from pathlib import Path

import numpy as np
import pytest
import torch

from paste3.helper import (
    dissimilarity_metric,
    get_common_genes,
    glmpca_distance,
    high_umi_gene_distance,
    kl_divergence,
    match_spots_using_spatial_heuristic,
    norm_and_center_coordinates,
    pca_distance,
    to_dense_array,
)

test_dir = Path(__file__).parent / "data"


def test_intersect(slices):
    common_genes = slices[1].var.index.intersection(slices[2].var.index)
    data = np.load(test_dir / "common_genes_s1_s2.npz", allow_pickle=True)

    assert np.all(np.equal(data["common_genes"], common_genes))


def test_kl_divergence_backend():
    X = torch.Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])).double()
    Y = torch.Tensor(np.array([[2, 4, 6], [8, 10, 12], [14, 16, 28]])).double()

    kl_divergence_matrix = kl_divergence(X, Y)
    expected_kl_divergence_matrix = np.array(
        [
            [0.0, 0.03323784, 0.01889736],
            [0.03607688, 0.0, 0.01442773],
            [0.05534049, 0.00193493, 0.02355472],
        ]
    )
    assert np.allclose(
        kl_divergence_matrix,
        expected_kl_divergence_matrix,
    )


def test_kl_divergence():
    X = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).double()
    Y = torch.Tensor([[2, 4, 6], [8, 10, 12], [14, 16, 28]]).double()

    kl_divergence_matrix = kl_divergence(X, Y)
    expected_kl_divergence_matrix = np.array(
        [
            [0.0, 0.03323784, 0.01889736],
            [0.03607688, 0.0, 0.01442773],
            [0.05534049, 0.00193493, 0.02355472],
        ]
    )
    assert np.allclose(
        kl_divergence_matrix,
        expected_kl_divergence_matrix,
    )


def test_filter_for_common_genes(slices):
    slices, _ = get_common_genes(slices)

    data = np.load(test_dir / "common_genes.npz", allow_pickle=True)
    for slice in slices:
        assert np.all(np.equal(data["common_genes"], slice.var.index))


def test_generalized_kl_divergence():
    X = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).double()
    Y = torch.Tensor([[2, 4, 6], [8, 10, 12], [14, 16, 28]]).double()

    generalized_kl_divergence_matrix = kl_divergence(X, Y, is_generalized=True)
    expected_kl_divergence_matrix = np.array(
        [
            [1.84111692, 14.54279955, 38.50128292],
            [0.88830648, 4.60279229, 22.93052383],
            [5.9637042, 0.69099319, 13.3879729],
        ]
    )
    assert np.allclose(
        generalized_kl_divergence_matrix,
        expected_kl_divergence_matrix,
    )


def test_glmpca_distance():
    data = np.load(test_dir / "glmpca_distance.npz")

    glmpca_distance_matrix = glmpca_distance(
        torch.Tensor(data["a_exp_dissim"]).double(),
        torch.Tensor(data["b_exp_dissim"]).double(),
        latent_dim=10,
        filter=True,
        maxIter=10,
    )
    assert np.allclose(glmpca_distance_matrix, data["glmpca_distance_matrix"])


def test_pca_distance(slices2):
    common_genes = slices2[1].var.index.intersection(slices2[2].var.index)
    sliceA = slices2[1][:, common_genes]
    sliceB = slices2[2][:, common_genes]

    pca = pca_distance(sliceA, sliceB, 2000, 20)
    # Saving test data for the entire matrix takes too much space,
    # so we only save a sample of 100 randomly-selected rows to compare
    random_indices = np.random.choice(pca.shape[0], 100, replace=False)
    pca = pca[random_indices, :]

    assert np.allclose(
        np.load(test_dir / "pca_distance.npz")["pca"],
        pca,
        rtol=1e-5,
        atol=1e-5,
    )


def test_high_umi_gene_distance():
    data = np.load(test_dir / "high_umi_gene_distance.npz")

    high_umi_gene_distance_matrix = high_umi_gene_distance(
        torch.Tensor(data["a_exp_dissim"]).double(),
        torch.Tensor(data["b_exp_dissim"]).double(),
        n=2000,
    )

    assert np.allclose(
        high_umi_gene_distance_matrix, data["high_umi_gene_distance_matrix"]
    )


@pytest.mark.parametrize(
    ("_use_ot", "filename"),
    [(True, "spots_mapping_true"), (False, "spots_mapping_false")],
)
def test_match_spots_using_spatial_heuristic(slices, _use_ot, filename):  # noqa: PT019
    slices, _ = get_common_genes(slices)

    data = np.load(test_dir / "match_spots_using_spatial_heuristic.npz")

    spots_mapping = match_spots_using_spatial_heuristic(
        data["a_spatial_data"], data["b_spatial_data"], use_ot=bool(_use_ot)
    )
    np.allclose(data[filename], spots_mapping)


def test_norm_and_center_coordinates():
    data = np.load(test_dir / "norm_and_center_coordinates.npz")

    X = norm_and_center_coordinates(data["a_exp_dissim"])
    Y = norm_and_center_coordinates(data["b_exp_dissim"])

    assert np.allclose(X, data["normalized_X"])
    assert np.allclose(Y, data["normalized_Y"])


@pytest.mark.parametrize(
    # Note: There's already a dedicated test for glmpca dissimilarity metric,
    # (test_glmpca_distance), so we don't include it here
    "dissimilarity",
    ["euc", "gkl", "kl", "selection_kl", "pca"],
)
def test_dissimilarity_metric(slices2, dissimilarity):
    sliceA, sliceB = slices2[:2]
    common_genes = sliceA.var.index.intersection(sliceB.var.index)
    sliceA = sliceA[:, common_genes]
    sliceB = sliceB[:, common_genes]

    A_X, B_X = to_dense_array(sliceA.X), to_dense_array(sliceB.X)
    M = dissimilarity_metric(
        dissimilarity,
        sliceA,
        sliceB,
        A_X,
        B_X,
    )

    # Saving test data for the entire matrix takes too much space,
    # so we only save a sample of 100 randomly-selected rows to compare
    random_indices = np.random.choice(M.shape[0], 100, replace=False)
    M = M[random_indices, :]

    assert np.allclose(
        np.load(test_dir / "dissimilarity_metric.npz")[dissimilarity],
        M,
        rtol=1e-5,
        atol=1e-5,
    )
