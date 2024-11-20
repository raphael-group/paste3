from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from pandas.testing import assert_frame_equal

from paste3.visualization import (
    generalized_procrustes_analysis,
    stack_slices_center,
    stack_slices_pairwise,
)

test_dir = Path(__file__).parent
input_dir = test_dir / "data/input"
output_dir = test_dir / "data/output"


def test_stack_slices_pairwise(slices):
    n_slices = len(slices)

    pairwise_info = [
        torch.Tensor(
            np.genfromtxt(input_dir / f"slices_{i}_{i + 1}_pairwise.csv", delimiter=",")
        ).double()
        for i in range(1, n_slices)
    ]

    new_slices, thetas, translations = stack_slices_pairwise(slices, pairwise_info)

    for i, slice in enumerate(new_slices, start=1):
        assert_frame_equal(
            pd.DataFrame(slice.obsm["spatial"], columns=["0", "1"]),
            pd.read_csv(output_dir / f"aligned_spatial_{i}_{i + 1}.csv"),
            atol=1e-6,
        )

    expected_thetas = [-0.25086326614894794, 0.5228805289947901, 0.02478065908672744]
    expected_translations = [
        ([16.44623233, 16.73757875], [19.80709569, 15.74706369]),
        ([-2.90017423e-08, -1.19685091e-08], [16.32537929, 17.43314825]),
        ([1.58526981e-07, 6.97949045e-07], [19.49901545, 17.35546584]),
    ]

    assert np.allclose(expected_thetas, thetas, rtol=1e-05, atol=1e-08)
    assert np.allclose(expected_translations, translations, rtol=1e-05, atol=1e-08)


def test_stack_slices_center(slices):
    center_slice = sc.read_h5ad(input_dir / "center_slice.h5ad")

    pairwise_info = [
        torch.Tensor(
            np.genfromtxt(input_dir / f"center_slice{i}_pairwise.csv", delimiter=",")
        ).double()
        for i in range(1, len(slices) + 1)
    ]

    new_center, new_slices, thetas, translations = stack_slices_center(
        center_slice, slices, pairwise_info
    )
    assert_frame_equal(
        pd.DataFrame(new_center.obsm["spatial"], columns=["0", "1"]),
        pd.read_csv(output_dir / "aligned_spatial_center.csv"),
        atol=1e-6,
    )

    for i, slice in enumerate(new_slices):
        assert_frame_equal(
            pd.DataFrame(slice.obsm["spatial"], columns=["0", "1"]),
            pd.read_csv(output_dir / f"slice{i}_stack_slices_center.csv"),
            atol=1e-6,
        )

    expected_thetas = [
        0.0,
        -0.24633847994675845,
        0.5083563603453264,
        0.0245843732567813,
    ]
    expected_translations = [
        ([16.44623224, 16.73757867], [16.44623224, 16.73757867]),
        ([16.44623233, 16.73757876], [19.80709569, 15.7470637]),
        ([16.44623251, 16.73757894], [16.325379, 17.43314794]),
        ([16.44623234, 16.73757877], [19.49901525, 17.35546567]),
    ]

    assert np.allclose(expected_thetas, thetas, rtol=1e-05, atol=1e-08)
    assert np.allclose(expected_translations, translations, rtol=1e-05, atol=1e-08)


def test_generalized_procrustes_analysis(slices):
    center_slice = sc.read_h5ad(input_dir / "center_slice.h5ad")

    pairwise_info = torch.Tensor(
        np.genfromtxt(input_dir / "center_slice1_pairwise.csv", delimiter=",")
    ).double()

    aligned_center, aligned_slice, theta, translation_x, translation_y = (
        generalized_procrustes_analysis(
            torch.Tensor(center_slice.obsm["spatial"]).double(),
            torch.Tensor(slices[0].obsm["spatial"]).double(),
            pairwise_info,
        )
    )

    assert_frame_equal(
        pd.DataFrame(aligned_center, columns=["0", "1"]),
        pd.read_csv(output_dir / "aligned_center.csv"),
        atol=1e-6,
    )
    assert_frame_equal(
        pd.DataFrame(aligned_slice, columns=["0", "1"]),
        pd.read_csv(output_dir / "aligned_slice.csv"),
        atol=1e-6,
    )
    expected_theta = 0.0
    expected_translation_x = [16.44623228, 16.73757874]
    expected_translation_y = [16.44623228, 16.73757874]

    assert np.all(
        np.isclose(expected_theta, theta, rtol=1e-05, atol=1e-08, equal_nan=True)
    )
    assert np.all(
        np.isclose(
            expected_translation_x,
            translation_x,
            rtol=1e-05,
            atol=1e-08,
            equal_nan=True,
        )
    )
    assert np.all(
        np.isclose(
            expected_translation_y,
            translation_y,
            rtol=1e-05,
            atol=1e-08,
            equal_nan=True,
        )
    )


def test_partial_stack_slices_pairwise(slices):
    n_slices = len(slices)

    pairwise_info = [
        torch.Tensor(
            np.genfromtxt(input_dir / f"slices_{i}_{i + 1}_pairwise.csv", delimiter=",")
        ).double()
        for i in range(1, n_slices)
    ]

    new_slices, _, _ = stack_slices_pairwise(slices, pairwise_info, is_partial=True)

    for i, slice in enumerate(new_slices, start=1):
        assert_frame_equal(
            pd.DataFrame(slice.obsm["spatial"], columns=["0", "1"]),
            pd.read_csv(output_dir / f"aligned_spatial_{i}_{i + 1}.csv"),
            atol=1e-6,
        )


def test_partial_procrustes_analysis(slices2):
    data = np.load(output_dir / "partial_procrustes_analysis.npz")

    assert torch.sum(torch.Tensor(data["pi"])) < 0.99999999

    x_aligned, y_aligned, _, _, _ = generalized_procrustes_analysis(
        torch.Tensor(slices2[0].obsm["spatial"]).double(),
        torch.Tensor(slices2[1].obsm["spatial"]).double(),
        torch.Tensor(data["pi"]).double(),
        is_partial=True,
    )
    assert np.allclose(x_aligned.cpu().numpy(), data["x_aligned"])
    assert np.allclose(y_aligned.cpu().numpy(), data["y_aligned"], atol=1e-06)
