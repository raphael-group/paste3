import hashlib
from pathlib import Path

import numpy as np
import ot.backend
import pandas as pd
import pytest
import torch
from pandas.testing import assert_frame_equal

from paste3.paste import (
    center_align,
    center_NMF,
    center_ot,
    line_search_partial,
    my_fused_gromov_wasserstein,
    pairwise_align,
)

test_dir = Path(__file__).parent / "data"
output_dir = test_dir / "output"


def assert_checksum_equals(temp_dir, filename, loose=False):
    generated_file = temp_dir / filename
    oracle = output_dir / filename

    if loose:
        assert_frame_equal(pd.read_csv(generated_file), pd.read_csv(oracle))
    else:
        with (
            Path.open(generated_file) as generated_file_f,
            Path.open(oracle) as oracle_f,
        ):
            assert (
                hashlib.md5(
                    "".join(generated_file_f.readlines()).encode("utf8")
                ).hexdigest()
                == hashlib.md5("".join(oracle_f.readlines()).encode("utf8")).hexdigest()
            )


def test_pairwise_alignment(slices):
    outcome, _ = pairwise_align(
        slices[0],
        slices[1],
        alpha=0.1,
        exp_dissim_metric="kl",
        pi_init=None,
        a_spots_weight=slices[0].obsm["weights"].astype(slices[0].X.dtype),
        b_spots_weight=slices[1].obsm["weights"].astype(slices[1].X.dtype),
        use_gpu=True,
    )
    expected_result = np.load(test_dir / "pairwise_alignment.npz", allow_pickle=True)
    assert np.allclose(expected_result["outcome"], outcome.cpu().numpy())


def test_center_alignment(slices):
    # Make a copy of the list
    slices = list(slices)
    n_slices = len(slices)
    center_slice, pairwise_info = center_align(
        slices[0],
        slices,
        slice_weights=n_slices * [1.0 / n_slices],
        alpha=0.1,
        n_components=15,
        random_seed=0,
        threshold=0.001,
        max_iter=2,
        exp_dissim_metric="kl",
        use_gpu=True,
        spots_weights=[
            slices[i].obsm["weights"].astype(slices[i].X.dtype)
            for i in range(len(slices))
        ],
    )
    expected_result = np.load(test_dir / "center_alignment.npz", allow_pickle=True)
    assert np.allclose(expected_result["paste_W"], center_slice.uns["paste_W"])
    assert np.allclose(expected_result["paste_H"], center_slice.uns["paste_H"])

    for i, pi in enumerate(pairwise_info):
        np.allclose(pi, expected_result[f"pi_{i}"])


def test_center_ot(slices):
    data = np.load(test_dir / "center_ot.npz", allow_pickle=True)
    pairwise_info, r = center_ot(
        feature_matrix=data["feature_matrix"],
        coeff_matrix=data["coeff_matrix"],
        slices=slices,
        center_coordinates=data["center_coordinates"],
        common_genes=data["common_genes"],
        use_gpu=True,
        alpha=0.1,
        exp_dissim_metric="kl",
        norm=False,
        pi_inits=[None for _ in range(len(slices))],
    )

    expected_r = [
        -25.08051355206619,
        -26.139415232102213,
        -25.728504876394076,
        -25.740615316378296,
    ]

    assert np.allclose(expected_r, r)

    for i, pi in enumerate(pairwise_info):
        assert np.allclose(pi, data[f"pi_{i}"])


def test_center_NMF(intersecting_slices):
    n_slices = len(intersecting_slices)

    data = np.load(test_dir / "center_NMF.npz")
    pairwise_info = [torch.Tensor(data[f"pi_{i}"]).double() for i in range(n_slices)]

    _W, _H = center_NMF(
        feature_matrix=data["feature_matrix"],
        slices=intersecting_slices,
        pis=pairwise_info,
        slice_weights=n_slices * [1.0 / n_slices],
        n_components=15,
        random_seed=0,
    )
    assert np.allclose(data["W"], _W)
    assert np.allclose(data["H"], _H)


def test_fused_gromov_wasserstein(spot_distance_matrix):
    data = np.load(test_dir / "fused_gromov_wasserstein.npz")
    M = torch.Tensor(data["exp_dissim_matrix"]).double()
    pairwise_info, log = my_fused_gromov_wasserstein(
        M,
        spot_distance_matrix[0],
        spot_distance_matrix[1],
        a_spots_weight=torch.ones((254,)).double() / 254,
        b_spots_weight=torch.ones((251,)).double() / 251,
        alpha=0.1,
        pi_init=None,
        loss_fun="square_loss",
        numItermax=10,
    )
    assert np.allclose(data["pairwise_info"], pairwise_info)


def test_gromov_linesearch(spot_distance_matrix):
    data = np.load(test_dir / "gromov_linesearch.npz")
    G = torch.Tensor(data["G"]).double()
    deltaG = torch.Tensor(data["deltaG"]).double()
    costG = 6.0935270338235075

    alpha, fc, cost_G = ot.gromov.solve_gromov_linesearch(
        G=G,
        deltaG=deltaG,
        cost_G=costG,
        C1=spot_distance_matrix[1],
        C2=spot_distance_matrix[2],
        M=0.0,
        reg=2 * 1.0,
    )
    assert alpha == 1.0
    assert fc == 1
    assert pytest.approx(cost_G) == -11.20545


def test_line_search_partial(spot_distance_matrix):
    d1, d2 = spot_distance_matrix[1], spot_distance_matrix[2]

    data = np.load(test_dir / "line_search_partial.npz")

    G = torch.Tensor(data["G"]).double()
    deltaG = torch.Tensor(data["deltaG"]).double()
    M = torch.Tensor(data["exp_dissim_matrix"]).double()

    alpha = 0.1

    def f_cost(pi):
        p, q = torch.sum(pi, axis=1), torch.sum(pi, axis=0)
        constC, hC1, hC2 = ot.gromov.init_matrix(d1, d2, p, q)
        return (1 - alpha) * torch.sum(M * pi) + alpha * ot.gromov.gwloss(
            constC, hC1, hC2, pi
        )

    def f_gradient(pi):
        p, q = torch.sum(pi, axis=1), torch.sum(pi, axis=0)
        constC, hC1, hC2 = ot.gromov.init_matrix(d1, d2, p, q)
        return ot.gromov.gwggrad(constC, hC1, hC2, pi)

    minimal_cost, a, cost_G = line_search_partial(
        alpha=alpha,
        exp_dissim_matrix=M,
        pi=G,
        a_spatial_dist=spot_distance_matrix[1],
        b_spatial_dist=spot_distance_matrix[2],
        pi_diff=deltaG,
        f_cost=f_cost,
        f_gradient=f_gradient,
    )
    assert minimal_cost == 1.0
    assert pytest.approx(a) == 0.4858849047237918
    assert pytest.approx(cost_G) == 102.6333512778727
