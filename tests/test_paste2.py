from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from paste3.paste import my_fused_gromov_wasserstein, pairwise_align

test_dir = Path(__file__).parent / "data"


@patch("paste3.paste.dissimilarity_metric")
def test_partial_pairwise_align_glmpca(fn, slices2):
    # Load pre-computed dissimilarity metrics,
    # since it is time-consuming to compute.
    data = np.load(test_dir / "test_partial_pairwise_align.npz")
    fn.return_value = torch.Tensor(data["glmpca"]).double()

    pi_BC, _ = pairwise_align(
        slices2[0],
        slices2[1],
        overlap_fraction=0.7,
        exp_dissim_metric="glmpca",
        norm=True,
        maxIter=10,
    )

    assert np.allclose(pi_BC.cpu().numpy(), data["pi_BC"], atol=1e-7)


def test_partial_pairwise_align_given_cost_matrix(slices):
    data = np.load(test_dir / "alignment_given_cost_matrix.npz", allow_pickle=True)
    sliceA = slices[1][:, data["common_genes"]]
    sliceB = slices[2][:, data["common_genes"]]

    glmpca_distance_matrix = torch.Tensor(data["exp_dissim_matrix"]).double()

    pairwise_info, log = pairwise_align(
        sliceA,
        sliceB,
        overlap_fraction=0.85,
        exp_dissim_matrix=glmpca_distance_matrix,
        alpha=0.1,
        norm=True,
        numItermax=10,
        maxIter=10,
    )
    assert np.allclose(data["expected_pairwise_info"], pairwise_info)
    assert log["loss"][-1].cpu().numpy() == pytest.approx(40.86494022326222)


def test_partial_pairwise_align_histology(slices2):
    data = np.load(test_dir / "partial_pairwise_align_histology.npz")
    pairwise_info, log = pairwise_align(
        slices2[0],
        slices2[1],
        overlap_fraction=0.7,
        alpha=0.1,
        exp_dissim_metric="euclidean",
        norm=True,
        numItermax=10,
        maxIter=10,
        do_histology=True,
    )
    assert log["loss"][-1].cpu().numpy() == pytest.approx(88.06713721008786)
    assert np.allclose(pairwise_info.cpu().numpy(), data["expected_pi"])


@pytest.mark.parametrize(
    ("armijo", "expected_log", "filename"),
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
            "partial_fused_gromov_wasserstein",
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
            "partial_fused_gromov_wasserstein_true",
        ),
    ],
)
def test_partial_fused_gromov_wasserstein(armijo, expected_log, filename):
    data = np.load(test_dir / "partial_fused_gromov_wasserstein.npz")

    pairwise_info, log = my_fused_gromov_wasserstein(
        torch.Tensor(data["exp_dissim_matrix"]).double(),
        torch.Tensor(data["distance_a"]).double(),
        torch.Tensor(data["distance_b"]).double(),
        torch.Tensor(data["a_weight"]).double(),
        torch.Tensor(data["b_weight"]).double(),
        alpha=0.1,
        overlap_fraction=0.7,
        pi_init=None,
        loss_fun="square_loss",
        armijo=armijo,
    )

    assert np.allclose(pairwise_info, data[filename], atol=1e-7)

    for k, v in expected_log.items():
        if k == "partial_fgw_cost":
            assert np.allclose(log["loss"][-1], v, rtol=1e-05)
        else:
            assert np.allclose(log[k], v, rtol=1e-05)
