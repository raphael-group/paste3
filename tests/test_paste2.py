from pathlib import Path
import pandas as pd
import numpy as np
from paste2.PASTE2 import (
    partial_pairwise_align,
    partial_pairwise_align_given_cost_matrix,
    partial_pairwise_align_histology,
    partial_fused_gromov_wasserstein,
    gwgrad_partial,
    gwloss_partial,
)
from paste2.helper import (
    intersect,
)
import pytest
from scipy.spatial import distance

test_dir = Path(__file__).parent
input_dir = test_dir / "data/input"
output_dir = test_dir / "data/output"

from pandas.testing import assert_frame_equal


def test_partial_pairwise_align(slices2):
    pi_BC = partial_pairwise_align(slices2[0], slices2[1], s=0.7)

    assert_frame_equal(
        pd.DataFrame(pi_BC, columns=[str(i) for i in range(pi_BC.shape[1])]),
        pd.read_csv(output_dir / "partial_pairwise_align.csv"),
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
    assert log == expected_log


@pytest.mark.skip
def test_partial_pairwise_align_histology(slices2):
    # TODO: this function doesn't seem to be called anywhere and also seems to be incomplete

    pairwise_info, log = partial_pairwise_align_histology(
        slices2[0], slices2[1], return_obj=True, dissimilarity="euclidean"
    )


def test_partial_fused_gromov_wasserstein(slices):
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
        alpha=0.1,
        m=0.7,
        G0=None,
        loss_fun="square_loss",
        log=True,
    )
    expected_log = {
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
    }

    assert_frame_equal(
        pd.DataFrame(pairwise_info, columns=[str(i) for i in range(264)]),
        pd.read_csv(output_dir / "partial_fused_gromov_wasserstein.csv"),
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


def test_gwloss_partial(slices):
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

    output = gwgrad_partial(distance_a, distance_b, G0, loss_fun="square_loss")

    assert_frame_equal(
        pd.DataFrame(output, columns=[str(i) for i in range(264)]),
        pd.read_csv(output_dir / "gwloss_partial.csv"),
    )
