import hashlib
from pathlib import Path

import numpy as np
import ot.backend
from ot.lp import emd
import pandas as pd
import tempfile

from paste.PASTE import (
    pairwise_align,
    center_align,
    center_ot,
    intersect,
    center_NMF,
    extract_data_matrix,
    kl_divergence_backend,
    to_dense_array,
    my_fused_gromov_wasserstein,
    solve_gromov_linesearch,
)
from pandas.testing import assert_frame_equal

test_dir = Path(__file__).parent
input_dir = test_dir / "data/input"
output_dir = test_dir / "data/output"


def assert_checksum_equals(temp_dir, filename):
    generated_file = temp_dir / filename
    oracle = output_dir / filename

    assert (
        hashlib.md5(
            "".join(open(generated_file, "r").readlines()).encode("utf8")
        ).hexdigest()
        == hashlib.md5(
            "".join(open(oracle, "r").readlines()).encode("utf8")
        ).hexdigest()
    )


def test_pairwise_alignment(slices):
    temp_dir = Path(tempfile.mkdtemp())
    outcome = pairwise_align(
        slices[0],
        slices[1],
        alpha=0.1,
        dissimilarity="kl",
        a_distribution=slices[0].obsm["weights"],
        b_distribution=slices[1].obsm["weights"],
        G_init=None,
    )
    pd.DataFrame(
        outcome, index=slices[0].obs.index, columns=slices[1].obs.index
    ).to_csv(temp_dir / "slices_1_2_pairwise.csv")
    assert_checksum_equals(temp_dir, "slices_1_2_pairwise.csv")


def test_center_alignment(slices):
    temp_dir = Path(tempfile.mkdtemp())

    # Make a copy of the list
    slices = list(slices)
    n_slices = len(slices)
    center_slice, pairwise_info = center_align(
        slices[0],
        slices,
        lmbda=n_slices * [1.0 / n_slices],
        alpha=0.1,
        n_components=15,
        random_seed=0,
        threshold=0.001,
        dissimilarity="kl",
        distributions=[slices[i].obsm["weights"] for i in range(len(slices))],
    )
    assert_frame_equal(
        pd.DataFrame(
            center_slice.uns["paste_W"],
            index=center_slice.obs.index,
            columns=[str(i) for i in range(15)],
        ),
        pd.read_csv(output_dir / "W_center.csv", index_col=0),
        check_names=False,
        rtol=1e-05,
        atol=1e-08,
    )
    assert_frame_equal(
        pd.DataFrame(center_slice.uns["paste_H"], columns=center_slice.var.index),
        pd.read_csv(output_dir / "H_center.csv"),
        rtol=1e-05,
        atol=1e-08,
    )

    for i, pi in enumerate(pairwise_info):
        pd.DataFrame(
            pi, index=center_slice.obs.index, columns=slices[i].obs.index
        ).to_csv(temp_dir / f"center_slice{i + 1}_pairwise.csv")
        assert_checksum_equals(temp_dir, f"center_slice{i + 1}_pairwise.csv")


def test_center_ot(slices):
    temp_dir = Path(tempfile.mkdtemp())

    common_genes = slices[0].var.index
    for slice in slices[1:]:
        common_genes = intersect(common_genes, slice.var.index)

    intersecting_slice = slices[0][:, common_genes]
    pairwise_info, r = center_ot(
        W=np.genfromtxt(input_dir / "W_intermediate.csv", delimiter=","),
        H=np.genfromtxt(input_dir / "H_intermediate.csv", delimiter=","),
        slices=slices,
        center_coordinates=intersecting_slice.obsm["spatial"],
        common_genes=common_genes,
        use_gpu=False,
        alpha=0.1,
        backend=ot.backend.NumpyBackend(),
        dissimilarity="kl",
        norm=False,
        G_inits=[None for _ in range(len(slices))],
    )

    expected_r = [
        -25.08051355206619,
        -26.139415232102213,
        -25.728504876394076,
        -25.740615316378296,
    ]

    assert np.all(np.isclose(expected_r, r, rtol=1e-05, atol=1e-08, equal_nan=True))

    for i, pi in enumerate(pairwise_info):
        pd.DataFrame(
            pi, index=intersecting_slice.obs.index, columns=slices[i].obs.index
        ).to_csv(temp_dir / f"center_ot{i + 1}_pairwise.csv")
        assert_checksum_equals(temp_dir, f"center_ot{i + 1}_pairwise.csv")


def test_center_NMF(intersecting_slices):
    n_slices = len(intersecting_slices)

    pairwise_info = [
        np.genfromtxt(input_dir / f"center_ot{i+1}_pairwise.csv", delimiter=",")
        for i in range(n_slices)
    ]

    _W, _H = center_NMF(
        W=np.genfromtxt(input_dir / "W_intermediate.csv", delimiter=","),
        H=np.genfromtxt(input_dir / "H_intermediate.csv", delimiter=","),
        slices=intersecting_slices,
        pis=pairwise_info,
        lmbda=n_slices * [1.0 / n_slices],
        n_components=15,
        random_seed=0,
    )

    assert_frame_equal(
        pd.DataFrame(
            _W,
            index=intersecting_slices[0].obs.index,
            columns=[str(i) for i in range(15)],
        ),
        pd.read_csv(output_dir / "W_center_NMF.csv", index_col=0),
        rtol=1e-05,
        atol=1e-08,
    )
    assert_frame_equal(
        pd.DataFrame(_H, columns=intersecting_slices[0].var.index),
        pd.read_csv(output_dir / "H_center_NMF.csv"),
        rtol=1e-05,
        atol=1e-08,
    )


def test_fused_gromov_wasserstein(slices):
    np.random.seed(0)
    temp_dir = Path(tempfile.mkdtemp())

    common_genes = intersect(slices[0].var.index, slices[1].var.index)
    sliceA = slices[0][:, common_genes]
    sliceB = slices[1][:, common_genes]

    nx = ot.backend.NumpyBackend()
    slice1_dist = ot.dist(
        nx.from_numpy(sliceA.obsm["spatial"]),
        nx.from_numpy(sliceA.obsm["spatial"]),
        metric="euclidean",
    )
    slice2_dist = ot.dist(
        nx.from_numpy(sliceB.obsm["spatial"]),
        nx.from_numpy(sliceB.obsm["spatial"]),
        metric="euclidean",
    )
    slice1_distr = nx.ones((sliceA.shape[0],)) / sliceA.shape[0]
    slice2_distr = nx.ones((sliceB.shape[0],)) / sliceB.shape[0]

    slice1_X = nx.from_numpy(to_dense_array(extract_data_matrix(sliceA, None)))
    slice2_X = nx.from_numpy(to_dense_array(extract_data_matrix(sliceB, None)))

    M = nx.from_numpy(kl_divergence_backend(slice1_X + 0.01, slice2_X + 0.01))

    pairwise_info, log = my_fused_gromov_wasserstein(
        M,
        slice1_dist,
        slice2_dist,
        slice1_distr,
        slice2_distr,
        G_init=None,
        loss_fun="square_loss",
        alpha=0.1,
        log=True,
        numItermax=200,
    )
    pd.DataFrame(pairwise_info).to_csv(temp_dir / "fused_gromov_wasserstein.csv")
    # TODO: Need to figure out where the randomness is coming from
    # assert_checksum_equals(temp_dir, "fused_gromov_wasserstein.csv")


def test_gromov_linesearch(slices):
    common_genes = intersect(slices[1].var.index, slices[2].var.index)
    sliceA = slices[1][:, common_genes]
    sliceB = slices[2][:, common_genes]

    nx = ot.backend.NumpyBackend()
    slice1_dist = ot.dist(
        nx.from_numpy(sliceA.obsm["spatial"]),
        nx.from_numpy(sliceA.obsm["spatial"]),
        metric="euclidean",
    )
    slice2_dist = ot.dist(
        nx.from_numpy(sliceB.obsm["spatial"]),
        nx.from_numpy(sliceB.obsm["spatial"]),
        metric="euclidean",
    )
    slice1_distr = nx.ones((sliceA.shape[0],)) / sliceA.shape[0]
    slice2_distr = nx.ones((sliceB.shape[0],)) / sliceB.shape[0]

    slice1_X = nx.from_numpy(to_dense_array(extract_data_matrix(sliceA, None)))
    slice2_X = nx.from_numpy(to_dense_array(extract_data_matrix(sliceB, None)))

    M = nx.from_numpy(kl_divergence_backend(slice1_X + 0.01, slice2_X + 0.01))
    slice1_distr, slice2_distr = ot.utils.list_to_array(slice1_distr, slice2_distr)

    constC, hC1, hC2 = ot.gromov.init_matrix(
        slice1_dist, slice2_dist, slice1_distr, slice2_distr, loss_fun="square_loss"
    )

    G = slice1_distr[:, None] * slice2_distr[None, :]
    Mi = M + 0.1 + ot.gromov.gwggrad(constC, hC1, hC2, G)
    Mi = Mi + nx.min(Mi)

    Gc = emd(slice1_distr, slice2_distr, Mi)
    deltaG = Gc - G
    costG = nx.sum(M * G) + 0.1 * ot.gromov.gwloss(constC, hC1, hC2, G)
    alpha, fc, cost_G = solve_gromov_linesearch(
        G, deltaG, costG, slice1_dist, slice2_dist, M=0.0, reg=1.0, nx=nx
    )
    assert alpha == 1.0
    assert fc == 1
    assert round(cost_G, 5) == -11.20545
