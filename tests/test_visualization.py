from pathlib import Path

import numpy as np
import scanpy as sc
import torch

from paste3.visualization import (
    generalized_procrustes_analysis,
    stack_slices_center,
    stack_slices_pairwise,
)

test_dir = Path(__file__).parent / "data"


def test_stack_slices_pairwise(slices):
    n_slices = len(slices)

    data = np.load(test_dir / "stack_slices_pairwise.npz")
    pairwise_info = [
        torch.Tensor(data[f"pi_{i}_{i+1}"]).double() for i in range(n_slices - 1)
    ]

    new_slices, thetas, translations = stack_slices_pairwise(slices, pairwise_info)

    for i, slice in enumerate(new_slices):
        assert np.allclose(slice.obsm["spatial"], data[f"aligned_{i}"])

    expected_thetas = [-0.25086326663252634, 0.5228805221288383, 0.024780658882267766]
    expected_translations = [
        ([16.44623233, 16.73757875], [19.80709569, 15.74706369]),
        ([-2.90017423e-08, -1.19685091e-08], [16.32537929, 17.43314825]),
        ([1.58526981e-07, 6.97949045e-07], [19.49901545, 17.35546584]),
    ]

    assert np.allclose(expected_thetas, thetas)
    assert np.allclose(expected_translations, translations)


def test_stack_slices_center(slices):
    center_slice = sc.read_h5ad(test_dir / "center_slice.h5ad")
    data = np.load(test_dir / "stack_slices_center.npz")
    pairwise_info = [
        torch.Tensor(data[f"pi_{i}_{i+1}"]).double() for i in range(len(slices))
    ]

    new_center, new_slices, thetas, translations = stack_slices_center(
        center_slice, slices, pairwise_info
    )

    assert np.allclose(new_center.obsm["spatial"], data["aligned"])

    for i, slice in enumerate(new_slices):
        assert np.allclose(data[f"new_slice_{i}"], slice.obsm["spatial"])

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

    assert np.allclose(expected_thetas, thetas)
    assert np.allclose(expected_translations, translations)


def test_generalized_procrustes_analysis(slices):
    center_slice = sc.read_h5ad(test_dir / "center_slice.h5ad")
    data = np.load(test_dir / "generalized_procrustes_analysis.npz")

    aligned_center, aligned_slice, theta, translation_x, translation_y = (
        generalized_procrustes_analysis(
            torch.Tensor(center_slice.obsm["spatial"]).double(),
            torch.Tensor(slices[0].obsm["spatial"]).double(),
            torch.Tensor(data["pi"]).double(),
        )
    )
    assert np.allclose(aligned_center, data["aligned_center"])
    assert np.allclose(aligned_slice, data["aligned_slice"])

    expected_theta = 0.0
    expected_translation_x = [16.44623228, 16.73757874]
    expected_translation_y = [16.44623228, 16.73757874]

    assert np.allclose(expected_theta, theta)
    assert np.allclose(expected_translation_x, translation_x)
    assert np.allclose(expected_translation_y, translation_y)


def test_partial_stack_slices_pairwise(slices):
    n_slices = len(slices)
    data = np.load(test_dir / "partial_stack_slices_pairwise.npz")

    pairwise_info = [
        torch.Tensor(data[f"pi_{i}_{i+1}"]).double() for i in range(n_slices - 1)
    ]

    new_slices, _, _ = stack_slices_pairwise(slices, pairwise_info, is_partial=True)

    for i, slice in enumerate(new_slices):
        assert np.allclose(slice.obsm["spatial"], data[f"aligned_{i}"])


def test_partial_procrustes_analysis(slices2):
    data = np.load(test_dir / "partial_procrustes_analysis.npz")

    assert torch.sum(torch.Tensor(data["pi"])) < 0.99999999

    x_aligned, y_aligned, _, _, _ = generalized_procrustes_analysis(
        torch.Tensor(slices2[0].obsm["spatial"]).double(),
        torch.Tensor(slices2[1].obsm["spatial"]).double(),
        torch.Tensor(data["pi"]).double(),
        is_partial=True,
    )
    assert np.allclose(x_aligned.cpu().numpy(), data["x_aligned"])
    assert np.allclose(y_aligned.cpu().numpy(), data["y_aligned"], atol=1e-06)
