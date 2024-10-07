from pathlib import Path
import pandas as pd
import numpy as np
from paste3.paste2 import (
    partial_pairwise_align,
    partial_pairwise_align_given_cost_matrix,
    partial_pairwise_align_histology,
    gwgrad_partial,
    gwloss_partial,
)
from paste3.helper import intersect
import pytest
from unittest.mock import patch
from scipy.spatial import distance
from pandas.testing import assert_frame_equal
from paste3.paste import my_fused_gromov_wasserstein

test_dir = Path(__file__).parent
input_dir = test_dir / "data/input"
output_dir = test_dir / "data/output"


@patch("paste3.paste2.dissimilarity_metric")
def test_partial_pairwise_align_glmpca(fn, slices2):
    # Load pre-computed dissimilarity metrics,
    # since it is time-consuming to compute.
    data = np.load(output_dir / "test_partial_pairwise_align.npz")
    fn.return_value = data["glmpca"]

    pi_BC = partial_pairwise_align(
        slices2[0],
        slices2[1],
        s=0.7,
        dissimilarity="glmpca",
        verbose=True,
        maxIter=10,
    )

    assert_frame_equal(
        pd.DataFrame(pi_BC, columns=[str(i) for i in range(pi_BC.shape[1])]),
        pd.read_csv(output_dir / "partial_pairwise_align_glmpca.csv"),
        rtol=1e-03,
        atol=1e-03,
    )


def test_partial_pairwise_align_given_cost_matrix(slices):
    common_genes = intersect(slices[1].var.index, slices[2].var.index)
    sliceA = slices[1][:, common_genes]
    sliceB = slices[2][:, common_genes]

    glmpca_distance_matrix = np.genfromtxt(
        input_dir / "glmpca_distance_matrix.csv", delimiter=",", skip_header=1
    )

    pairwise_info, log = partial_pairwise_align_given_cost_matrix(
        sliceA,
        sliceB,
        s=0.85,
        M=glmpca_distance_matrix,
        alpha=0.1,
        armijo=False,
        norm=True,
        return_obj=True,
        verbose=True,
        numItermax=10,
    )

    assert_frame_equal(
        pd.DataFrame(pairwise_info, columns=[str(i) for i in range(264)]),
        pd.read_csv(output_dir / "align_given_cost_matrix_pairwise_info.csv"),
        rtol=1e-05,
    )
    assert log == pytest.approx(40.86494022326222)


def test_partial_pairwise_align_histology(slices2):
    pairwise_info, log = partial_pairwise_align_histology(
        slices2[0],
        slices2[1],
        s=0.7,
        return_obj=True,
        dissimilarity="euclidean",
        verbose=True,
        numItermax=10,
    )
    assert log == pytest.approx(88.06713721008786)
    assert_frame_equal(
        pd.DataFrame(pairwise_info, columns=[str(i) for i in range(2877)]),
        pd.read_csv(output_dir / "partial_pairwise_align_histology.csv"),
        rtol=1e-05,
    )


@pytest.mark.parametrize(
    "armijo, expected_log, filename",
    [
        (
            False,
            {
                "err": [0.047201842558232954],
                "loss": [
                    174.40490055003175,
                    52.31031712851437,
                    35.35388862002473,
                    30.84819243143108,
                    30.770197475353303,
                    30.764346125679705,
                    30.76336403641352,
                    30.76332791868975,
                    30.762808654741757,
                    30.762727812006336,
                    30.762727812006336,
                ],
                "partial_fgw_cost": 30.762727812006336,
            },
            "partial_fused_gromov_wasserstein.csv",
        ),
        (
            True,
            {
                "err": [0.047201842558232954, 9.659795787581263e-08],
                "loss": [
                    174.40490055003175,
                    53.40351168112147,
                    35.56234792074645,
                    30.897730857089122,
                    30.772178816776396,
                    30.764588004718327,
                    30.76338000971795,
                    30.763328599181595,
                    30.762818343959903,
                    30.762728863994308,
                    30.762727822540885,
                    30.762727812111688,
                ],
                "partial_fgw_cost": 30.76272781211168,
            },
            "partial_fused_gromov_wasserstein_true.csv",
        ),
    ],
)
def test_partial_fused_gromov_wasserstein(slices, armijo, expected_log, filename):
    common_genes = intersect(slices[1].var.index, slices[2].var.index)
    sliceA = slices[1][:, common_genes]
    sliceB = slices[2][:, common_genes]

    distance_a = distance.cdist(sliceA.obsm["spatial"], sliceA.obsm["spatial"])
    distance_b = distance.cdist(sliceB.obsm["spatial"], sliceB.obsm["spatial"])

    distance_a /= distance_a[distance_a > 0].min().min()
    distance_b /= distance_b[distance_b > 0].min().min()

    glmpca_distance_matrix = np.genfromtxt(
        input_dir / "glmpca_distance_matrix.csv", delimiter=",", skip_header=1
    )

    distance_a /= distance_a[distance_a > 0].max()
    distance_a *= glmpca_distance_matrix.max()
    distance_b /= distance_b[distance_b > 0].max()
    distance_b *= glmpca_distance_matrix.max()

    pairwise_info, log = my_fused_gromov_wasserstein(
        glmpca_distance_matrix,
        distance_a,
        distance_b,
        np.ones((sliceA.shape[0],)) / sliceA.shape[0],
        np.ones((sliceB.shape[0],)) / sliceB.shape[0],
        alpha=0.1,
        m=0.7,
        G0=None,
        loss_fun="square_loss",
        armijo=armijo,
        log=True,
    )

    assert_frame_equal(
        pd.DataFrame(pairwise_info, columns=[str(i) for i in range(264)]),
        pd.read_csv(output_dir / filename),
        rtol=1e-05,
    )

    for k, v in expected_log.items():
        assert np.allclose(log[k], v, rtol=1e-05)


def test_gloss_partial(slices):
    common_genes = intersect(slices[1].var.index, slices[2].var.index)
    sliceA = slices[1][:, common_genes]
    sliceB = slices[2][:, common_genes]

    distance_a = distance.cdist(sliceA.obsm["spatial"], sliceA.obsm["spatial"])
    distance_b = distance.cdist(sliceB.obsm["spatial"], sliceB.obsm["spatial"])

    distance_a /= distance_a[distance_a > 0].min().min()
    distance_b /= distance_b[distance_b > 0].min().min()

    glmpca_distance_matrix = np.genfromtxt(
        input_dir / "glmpca_distance_matrix.csv", delimiter=",", skip_header=1
    )

    distance_a /= distance_a[distance_a > 0].max()
    distance_a *= glmpca_distance_matrix.max()
    distance_b /= distance_b[distance_b > 0].max()
    distance_b *= glmpca_distance_matrix.max()

    G0 = np.outer(
        np.ones((sliceA.shape[0],)) / sliceA.shape[0],
        np.ones((sliceB.shape[0],)) / sliceB.shape[0],
    )

    output = gwloss_partial(distance_a, distance_b, G0)

    expected_output = pytest.approx(1135.0163192178504)
    assert output == expected_output


@pytest.mark.parametrize(
    "loss_fun, filename",
    [
        ("square_loss", "gwloss_partial.csv"),
        ("kl_loss", "gwloss_partial_kl_loss.csv"),
    ],
)
def test_gwloss_partial(slices, loss_fun, filename):
    common_genes = intersect(slices[1].var.index, slices[2].var.index)
    sliceA = slices[1][:, common_genes]
    sliceB = slices[2][:, common_genes]

    distance_a = distance.cdist(sliceA.obsm["spatial"], sliceA.obsm["spatial"])
    distance_b = distance.cdist(sliceB.obsm["spatial"], sliceB.obsm["spatial"])

    distance_a /= distance_a[distance_a > 0].min().min()
    distance_b /= distance_b[distance_b > 0].min().min()

    glmpca_distance_matrix = np.genfromtxt(
        input_dir / "glmpca_distance_matrix.csv", delimiter=",", skip_header=1
    )

    distance_a /= distance_a[distance_a > 0].max()
    distance_a *= glmpca_distance_matrix.max()
    distance_b /= distance_b[distance_b > 0].max()
    distance_b *= glmpca_distance_matrix.max()

    G0 = np.outer(
        np.ones((sliceA.shape[0],)) / sliceA.shape[0],
        np.ones((sliceB.shape[0],)) / sliceB.shape[0],
    )

    output = gwgrad_partial(distance_a, distance_b, G0, loss_fun=loss_fun)

    assert_frame_equal(
        pd.DataFrame(output, columns=[str(i) for i in range(264)]),
        pd.read_csv(output_dir / filename),
    )
