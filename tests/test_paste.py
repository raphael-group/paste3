import hashlib
from pathlib import Path
import numpy as np
import torch
import ot.backend
import pandas as pd
import tempfile
from paste3.paste import (
    pairwise_align,
    center_align,
    center_ot,
    intersect,
    center_NMF,
    my_fused_gromov_wasserstein,
    solve_gromov_linesearch,
    line_search_partial,
)
from pandas.testing import assert_frame_equal
import pytest

test_dir = Path(__file__).parent
input_dir = test_dir / "data/input"
output_dir = test_dir / "data/output"


def assert_checksum_equals(temp_dir, filename, loose=False):
    generated_file = temp_dir / filename
    oracle = output_dir / filename

    if loose:
        assert_frame_equal(pd.read_csv(generated_file), pd.read_csv(oracle))
    else:
        assert (
            hashlib.md5(
                "".join(open(generated_file, "r").readlines()).encode("utf8")
            ).hexdigest()
            == hashlib.md5(
                "".join(open(oracle, "r").readlines()).encode("utf8")
            ).hexdigest()
        )


def test_pairwise_alignment(slices):
    outcome = pairwise_align(
        slices[0],
        slices[1],
        alpha=0.1,
        exp_dissim_metric="kl",
        a_spots_weight=slices[0].obsm["weights"].astype(slices[0].X.dtype),
        b_spots_weight=slices[1].obsm["weights"].astype(slices[1].X.dtype),
        pis_init=None,
        use_gpu=True,
        backend=ot.backend.TorchBackend(),
    )
    probability_mapping = pd.DataFrame(
        outcome.cpu().numpy(), index=slices[0].obs.index, columns=slices[1].obs.index
    )
    true_probability_mapping = pd.read_csv(
        output_dir / "slices_1_2_pairwise.csv", index_col=0
    )
    assert_frame_equal(probability_mapping, true_probability_mapping, check_dtype=False)


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
        backend=ot.backend.TorchBackend(),
        spots_weights=[
            slices[i].obsm["weights"].astype(slices[i].X.dtype)
            for i in range(len(slices))
        ],
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
        atol=1e-04,
        check_dtype=False,
    )
    assert_frame_equal(
        pd.DataFrame(center_slice.uns["paste_H"], columns=center_slice.var.index),
        pd.read_csv(output_dir / "H_center.csv", index_col=0),
        rtol=1e-05,
        atol=1e-04,
        check_dtype=False,
    )

    for i, pi in enumerate(pairwise_info):
        pairwise_mapping = pd.DataFrame(
            pi.cpu().numpy(), index=center_slice.obs.index, columns=slices[i].obs.index
        )
        true_pairwise_mapping = pd.read_csv(
            output_dir / f"center_slice{i + 1}_pairwise.csv", index_col=0
        )
        assert_frame_equal(pairwise_mapping, true_pairwise_mapping, check_dtype=False)


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
        use_gpu=True,
        alpha=0.1,
        backend=ot.backend.TorchBackend(),
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

    assert np.allclose(expected_r, r)

    for i, pi in enumerate(pairwise_info):
        pd.DataFrame(
            pi.cpu().numpy(),
            index=intersecting_slice.obs.index,
            columns=slices[i].obs.index,
        ).to_csv(temp_dir / f"center_ot{i + 1}_pairwise.csv")
        assert_checksum_equals(temp_dir, f"center_ot{i + 1}_pairwise.csv", loose=True)


def test_center_NMF(intersecting_slices):
    n_slices = len(intersecting_slices)

    pairwise_info = [
        torch.Tensor(
            np.genfromtxt(input_dir / f"center_ot{i+1}_pairwise.csv", delimiter=",")
        ).double()
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


def test_fused_gromov_wasserstein(slices, spot_distance_matrix):
    temp_dir = Path(tempfile.mkdtemp())

    nx = ot.backend.TorchBackend()

    M = torch.Tensor(
        np.genfromtxt(input_dir / "gene_distance.csv", delimiter=",")
    ).double()
    pairwise_info, log = my_fused_gromov_wasserstein(
        M,
        spot_distance_matrix[0],
        spot_distance_matrix[1],
        p=nx.ones((254,)).double() / 254,
        q=nx.ones((251,)).double() / 251,
        alpha=0.1,
        G0=None,
        loss_fun="square_loss",
        numItermax=10,
    )
    pd.DataFrame(pairwise_info).to_csv(
        temp_dir / "fused_gromov_wasserstein.csv", index=False
    )
    assert_checksum_equals(temp_dir, "fused_gromov_wasserstein.csv")


def test_gromov_linesearch(slices, spot_distance_matrix):
    nx = ot.backend.TorchBackend()

    G = 1.509115054931788e-05 * torch.ones((251, 264)).double()
    deltaG = torch.Tensor(
        np.genfromtxt(input_dir / "deltaG.csv", delimiter=",")
    ).double()
    costG = 6.0935270338235075

    alpha, fc, cost_G = solve_gromov_linesearch(
        G,
        deltaG,
        costG,
        spot_distance_matrix[1],
        spot_distance_matrix[2],
        M=0.0,
        reg=1.0,
        nx=nx,
    )
    assert alpha == 1.0
    assert fc == 1
    assert pytest.approx(cost_G) == -11.20545


def test_line_search_partial(slices, spot_distance_matrix):
    G = 1.509115054931788e-05 * torch.ones((251, 264)).double()
    deltaG = torch.Tensor(
        np.genfromtxt(input_dir / "deltaG.csv", delimiter=",")
    ).double()
    M = torch.Tensor(
        np.genfromtxt(input_dir / "euc_dissimilarity.csv", delimiter=",")
    ).double()

    alpha, a, cost_G = line_search_partial(
        reg=0.1,
        M=M,
        G=G,
        C1=spot_distance_matrix[1],
        C2=spot_distance_matrix[2],
        deltaG=deltaG,
    )
    assert alpha == 1.0
    assert pytest.approx(a) == 0.4858849047237918
    assert pytest.approx(cost_G) == 102.6333512778727
