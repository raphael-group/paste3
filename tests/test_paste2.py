from pathlib import Path
import pandas as pd
import numpy as np
from paste3.paste2 import (
    partial_pairwise_align,
    partial_pairwise_align_given_cost_matrix,
    partial_pairwise_align_histology,
    partial_fused_gromov_wasserstein,
    gwgrad_partial,
    gwloss_partial,
)
from paste3.helper import intersect
import pytest
from scipy.spatial import distance
from pandas.testing import assert_frame_equal

test_dir = Path(__file__).parent
input_dir = test_dir / "data/input"
output_dir = test_dir / "data/output"


def pytest_generate_tests(metafunc):
    if "loss_fun" in metafunc.fixturenames:
        metafunc.parametrize(
            "loss_fun, filename",
            [
                ("square_loss", "gwloss_partial.csv"),
                ("kl_loss", "gwloss_partial_kl_loss.csv"),
            ],
        )
    if "dissimilarity" in metafunc.fixturenames:
        metafunc.parametrize(
            "dissimilarity, filename",
            [
                ("euc", "partial_pairwise_align_euc.csv"),
                ("gkl", "partial_pairwise_align_gkl.csv"),
                ("kl", "partial_pairwise_align_kl.csv"),
                ("selection_kl", "partial_pairwise_align_selection_kl.csv"),
                ("pca", "partial_pairwise_align_pca.csv"),
                ("glmpca", "partial_pairwise_align_glmpca.csv"),
            ],
        )
    if "armijo" in metafunc.fixturenames:
        metafunc.parametrize(
            "armijo, expected_log, filename",
            [
                (
                    False,
                    {
                        "err": [0.047201842558232954],
                        "loss": [
                            52.31031712851437,
                            35.35388862002473,
                            30.84819243143108,
                            30.770197475353303,
                            30.7643461256797,
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
                            53.40351168112148,
                            35.56234792074653,
                            30.897730857089122,
                            30.77217881677637,
                            30.764588004718373,
                            30.763380009717963,
                            30.76332859918154,
                            30.762818343959903,
                            30.762728863994322,
                            30.76272782254089,
                            30.76272781211168,
                        ],
                        "partial_fgw_cost": 30.76272781211168,
                    },
                    "partial_fused_gromov_wasserstein_true.csv",
                ),
            ],
        )


def test_partial_pairwise_align(slices2, dissimilarity, filename):
    pi_BC = partial_pairwise_align(
        slices2[0], slices2[1], s=0.7, dissimilarity=dissimilarity
    )
    pd.DataFrame(pi_BC).to_csv(output_dir / filename, index=False)

    assert_frame_equal(
        pd.DataFrame(pi_BC, columns=[str(i) for i in range(pi_BC.shape[1])]),
        pd.read_csv(output_dir / filename),
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
        verbose=False,
    )

    expected_log = 40.86486220302934

    assert_frame_equal(
        pd.DataFrame(pairwise_info, columns=[str(i) for i in range(264)]),
        pd.read_csv(output_dir / "align_given_cost_matrix_pairwise_info.csv"),
        rtol=1e-05,
    )
    assert log == pytest.approx(expected_log)


def test_partial_pairwise_align_histology(slices2):
    pairwise_info, log = partial_pairwise_align_histology(
        slices2[0], slices2[1], s=0.7, return_obj=True, dissimilarity="euclidean"
    )
    assert round(log, 3) == round(78.30015827691841, 3)
    assert_frame_equal(
        pd.DataFrame(pairwise_info, columns=[str(i) for i in range(2877)]),
        pd.read_csv(output_dir / "partial_pairwise_align_histology.csv"),
        rtol=1e-05,
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

    pairwise_info, log = partial_fused_gromov_wasserstein(
        glmpca_distance_matrix,
        distance_a,
        distance_b,
        np.ones((sliceA.shape[0],)) / sliceA.shape[0],
        np.ones((sliceB.shape[0],)) / sliceB.shape[0],
        armijo=armijo,
        alpha=0.1,
        m=0.7,
        G0=None,
        loss_fun="square_loss",
        log=True,
    )

    assert_frame_equal(
        pd.DataFrame(pairwise_info, columns=[str(i) for i in range(264)]),
        pd.read_csv(output_dir / filename),
        rtol=1e-05,
    )

    for k, v in expected_log.items():
        assert np.all(np.isclose(log[k], v, rtol=1e-05))


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

    expected_output = 1135.0163192178504
    assert output == expected_output


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
