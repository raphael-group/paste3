from pathlib import Path
import numpy as np
import ot.backend
import pandas as pd
from pandas.testing import assert_frame_equal
from paste.helper import (
    intersect,
    kl_divergence_backend,
    to_dense_array,
    extract_data_matrix,
    kl_divergence,
    filter_for_common_genes,
    match_spots_using_spatial_heuristic,
)

from paste2.helper import (
    generalized_kl_divergence,
    glmpca_distance,
    pca_distance,
    high_umi_gene_distance,
    norm_and_center_coordinates,
)

test_dir = Path(__file__).parent
input_dir = test_dir / "data/input"
output_dir = test_dir / "data/output"


def test_intersect(slices):
    common_genes = intersect(slices[1].var.index, slices[2].var.index)
    assert np.all(
        np.equal(
            common_genes,
            list(np.genfromtxt(output_dir / "common_genes_s1_s2.csv", dtype=str)),
        )
    )


def test_kl_divergence_backend(slices):
    nx = ot.backend.NumpyBackend()

    common_genes = intersect(slices[1].var.index, slices[2].var.index)
    sliceA = slices[1][:, common_genes]
    sliceB = slices[2][:, common_genes]

    slice1_X = nx.from_numpy(to_dense_array(extract_data_matrix(sliceA, None)))
    slice2_X = nx.from_numpy(to_dense_array(extract_data_matrix(sliceB, None)))

    kl_divergence_matrix = kl_divergence_backend(slice1_X + 0.01, slice2_X + 0.01)
    assert_frame_equal(
        pd.DataFrame(kl_divergence_matrix, columns=[str(i) for i in range(264)]),
        pd.read_csv(output_dir / "kl_divergence_backend_matrix.csv"),
        check_names=False,
        check_dtype=False,
        rtol=1e-04,
    )


def test_kl_divergence(slices):
    common_genes = intersect(slices[1].var.index, slices[2].var.index)
    sliceA = slices[1][:, common_genes]
    sliceB = slices[2][:, common_genes]

    sliceA_X, sliceB_X = to_dense_array(
        extract_data_matrix(sliceA, None)
    ), to_dense_array(extract_data_matrix(sliceB, None))

    kl_divergence_matrix = kl_divergence(sliceA_X + 0.01, sliceB_X + 0.01)
    assert_frame_equal(
        pd.DataFrame(kl_divergence_matrix, columns=[str(i) for i in range(264)]),
        pd.read_csv(output_dir / "kl_divergence_matrix.csv"),
        check_names=False,
        check_dtype=False,
        rtol=1e-03,
    )


def test_filter_for_common_genes(slices):
    # creating a copy of the original list
    slices = list(slices)
    filter_for_common_genes(slices)

    common_genes = list(np.genfromtxt(output_dir / "common_genes.csv", dtype=str))
    for slice in slices:
        assert np.all(np.equal(common_genes, slice.var.index))


def test_generalized_kl_divergence(slices):
    common_genes = intersect(slices[1].var.index, slices[2].var.index)
    sliceA = slices[1][:, common_genes]
    sliceB = slices[2][:, common_genes]

    sliceA_X, sliceB_X = to_dense_array(
        extract_data_matrix(sliceA, None)
    ), to_dense_array(extract_data_matrix(sliceB, None))

    generalized_kl_divergence_matrix = generalized_kl_divergence(
        sliceA_X + 0.01, sliceB_X + 0.01
    )
    assert_frame_equal(
        pd.DataFrame(
            generalized_kl_divergence_matrix, columns=[str(i) for i in range(264)]
        ),
        pd.read_csv(output_dir / "generalized_kl_divergence_matrix.csv"),
        check_names=False,
        check_dtype=False,
        rtol=1e-01,
    )


def test_glmpca_distance():
    np.random.seed(0)
    sliceA_X = np.genfromtxt(input_dir / "sliceA_X.csv", delimiter=",", skip_header=1)[
        10:, :1000
    ]
    sliceB_X = np.genfromtxt(input_dir / "sliceB_X.csv", delimiter=",", skip_header=1)[
        10:, :1000
    ]

    glmpca_distance_matrix = glmpca_distance(
        sliceA_X, sliceB_X, latent_dim=10, filter=True
    )

    assert_frame_equal(
        pd.DataFrame(glmpca_distance_matrix, columns=[str(i) for i in range(254)]),
        pd.read_csv(output_dir / "glmpca_distance_matrix.csv"),
        check_names=False,
        check_dtype=False,
        rtol=1e-04,
    )


def test_pca_distance(slices2):
    common_genes = intersect(slices2[1].var.index, slices2[2].var.index)
    sliceA = slices2[1][:, common_genes]
    sliceB = slices2[2][:, common_genes]

    pca_distance_matrix = pca_distance(sliceA, sliceB, 2000, 20)
    # TODO: need to add file for this
    # TODO: its too large need to introduce compression
    # assert_frame_equal(
    #     pd.DataFrame(pca_distance_matrix, columns=[str(i) for i in range(2873)]),
    #     pd.read_csv(output_dir / "pca_distance_matrix.csv"),
    #     check_names=False,
    #     check_dtype=False,
    #     rtol=1e-04,
    # )


def test_high_umi_gene_distance(slices):
    common_genes = intersect(slices[1].var.index, slices[2].var.index)
    sliceA = slices[1][:, common_genes]
    sliceB = slices[2][:, common_genes]

    sliceA_X, sliceB_X = to_dense_array(
        extract_data_matrix(sliceA, None)
    ), to_dense_array(extract_data_matrix(sliceB, None))

    high_umi_gene_distance_matrix = high_umi_gene_distance(sliceA_X, sliceB_X, n=2000)
    assert_frame_equal(
        pd.DataFrame(
            high_umi_gene_distance_matrix, columns=[str(i) for i in range(264)]
        ),
        pd.read_csv(output_dir / "high_umi_gene_distance_matrix.csv"),
        check_names=False,
        check_dtype=False,
        rtol=1e-03,
    )


def test_match_spots_using_spatial_heuristic(slices):
    # creating a copy of the original list
    slices = list(slices)
    filter_for_common_genes(slices)

    spots_mapping = match_spots_using_spatial_heuristic(
        slices[0].X, slices[1].X, use_ot=True
    )
    assert_frame_equal(
        pd.DataFrame(spots_mapping, columns=[str(i) for i in range(251)]),
        pd.read_csv(output_dir / "spots_mapping.csv"),
        check_names=False,
        check_dtype=False,
        rtol=1e-04,
    )


def test_norm_and_center_coordinates(slices):
    common_genes = intersect(slices[1].var.index, slices[2].var.index)
    sliceA = slices[1][:, common_genes]
    sliceB = slices[2][:, common_genes]

    sliceA_X, sliceB_X = to_dense_array(
        extract_data_matrix(sliceA, None)
    ), to_dense_array(extract_data_matrix(sliceB, None))

    X = norm_and_center_coordinates(sliceA_X)
    Y = norm_and_center_coordinates(sliceB_X)

    assert_frame_equal(
        pd.DataFrame(X, columns=[str(i) for i in range(6886)]),
        pd.read_csv(output_dir / "normalized_X.csv"),
        check_names=False,
        check_dtype=False,
        rtol=1e-04,
    )
    assert_frame_equal(
        pd.DataFrame(Y, columns=[str(i) for i in range(6886)]),
        pd.read_csv(output_dir / "normalized_Y.csv"),
        check_names=False,
        check_dtype=False,
        rtol=1e-04,
    )
