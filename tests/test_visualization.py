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

    new_slices, thetas, translations = stack_slices_pairwise(
        slices, pairwise_info, return_params=True
    )

    for i, slice in enumerate(new_slices, start=1):
        assert_frame_equal(
            pd.DataFrame(slice.obsm["spatial"], columns=["0", "1"]),
            pd.read_csv(output_dir / f"aligned_spatial_{i}_{i + 1}.csv"),
            atol=1e-6,
        )

    expected_thetas = [-0.25086326614894794, 0.5228805289947901, 0.02478065908672744]
    expected_translations = [
        [16.44623228, 16.73757874],
        [19.80709562, 15.74706375],
        [16.32537879, 17.43314773],
        [19.49901527, 17.35546565],
    ]

    assert np.all(
        np.isclose(expected_thetas, thetas, rtol=1e-05, atol=1e-08, equal_nan=True)
    )
    assert np.all(
        np.isclose(
            expected_translations, translations, rtol=1e-05, atol=1e-08, equal_nan=True
        )
    )


def test_stack_slices_center(slices):
    center_slice = sc.read_h5ad(input_dir / "center_slice.h5ad")

    pairwise_info = [
        torch.Tensor(
            np.genfromtxt(input_dir / f"center_slice{i}_pairwise.csv", delimiter=",")
        ).double()
        for i in range(1, len(slices) + 1)
    ]

    new_center, new_slices, thetas, translations = stack_slices_center(
        center_slice, slices, pairwise_info, output_params=True
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
        [16.44623228, 16.73757874],
        [19.80709562, 15.74706375],
        [16.32537879, 17.43314773],
        [19.49901527, 17.35546565],
    ]

    assert np.all(
        np.isclose(expected_thetas, thetas, rtol=1e-05, atol=1e-08, equal_nan=True)
    )
    assert np.all(
        np.isclose(
            expected_translations, translations, rtol=1e-05, atol=1e-08, equal_nan=True
        )
    )


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
            return_params=True,
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

    new_slices = stack_slices_pairwise(slices, pairwise_info, is_partial=True)

    for i, slice in enumerate(new_slices, start=1):
        assert_frame_equal(
            pd.DataFrame(slice.obsm["spatial"], columns=["0", "1"]),
            pd.read_csv(output_dir / f"aligned_spatial_{i}_{i + 1}.csv"),
            atol=1e-6,
        )


def test_partial_procrustes_analysis(slices):
    center_slice = sc.read_h5ad(input_dir / "center_slice.h5ad")

    pairwise_info = torch.Tensor(
        np.genfromtxt(input_dir / "center_slice1_pairwise.csv", delimiter=",")
    ).double()

    aligned_center, aligned_slice = generalized_procrustes_analysis(
        torch.Tensor(center_slice.obsm["spatial"]).double(),
        torch.Tensor(slices[0].obsm["spatial"]).double(),
        pairwise_info,
        is_partial=True,
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
