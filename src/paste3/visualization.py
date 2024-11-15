import logging

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from anndata import AnnData

logger = logging.getLogger(__name__)


"""
    Functions to plot slices and align spatial coordinates after obtaining a mapping from PASTE.
"""


def stack_slices_pairwise(
    slices: list[AnnData], pis: list[np.ndarray], is_partial: bool = False
) -> tuple[list[AnnData], list[float] | None, list[np.ndarray] | None]:
    """
    Align spatial coordinates of sequential pairwise slices.

    In other words, align:

        slices[0] --> slices[1] --> slices[2] --> ...



    Args:
        slices: List of slices.
        pis: List of pi (``pairwise_align()`` output) between consecutive slices.
        return_params: If ``True``, addtionally return angles of rotation (theta) and translations for each slice.
        is_partial: Boolean of whether this is partial pairwise analysis or a total one

    Returns:
        - List of slices with aligned spatial coordinates.

        If ``output_params = True``, additionally return:

        - List of angles of rotation (theta) for each slice.
        - List of translations [x_translation, y_translation] for each slice.
    """
    assert (
        len(slices) == len(pis) + 1
    ), "'slices' should have length one more than 'pis'. Please double check."
    assert len(slices) > 1, "You should have at least 2 layers."
    aligned_coordinates = []
    rotation_angles = []
    translations = []
    (
        source_coordinates,
        target_coordinates,
        rotation_angle,
        x_translation,
        y_translation,
    ) = generalized_procrustes_analysis(
        torch.Tensor(slices[0].obsm["spatial"]).to(pis[0].dtype).to(pis[0].device),
        torch.Tensor(slices[1].obsm["spatial"]).to(pis[0].dtype).to(pis[0].device),
        pis[0],
        is_partial=is_partial,
    )
    rotation_angles.append(rotation_angle)
    translations.append((x_translation, y_translation))
    aligned_coordinates.append(source_coordinates)
    aligned_coordinates.append(target_coordinates)
    for i in range(1, len(slices) - 1):
        (
            source_coordinates,
            target_coordinates,
            rotation_angle,
            x_translation,
            y_translation,
        ) = generalized_procrustes_analysis(
            aligned_coordinates[i],
            torch.Tensor(slices[i + 1].obsm["spatial"])
            .to(pis[i].dtype)
            .to(pis[i].device),
            pis[i],
            is_partial=is_partial,
        )
        rotation_angles.append(rotation_angle)
        translations.append((x_translation, y_translation))
        if is_partial:
            shift = aligned_coordinates[i][0, :] - source_coordinates[0, :]
            target_coordinates = target_coordinates + shift
        aligned_coordinates.append(target_coordinates)

    new_slices = []
    for i in range(len(slices)):
        _slice = slices[i].copy()
        _slice.obsm["spatial"] = aligned_coordinates[i].cpu().numpy()
        new_slices.append(_slice)

    return new_slices, rotation_angles, translations


def stack_slices_center(
    center_slice: AnnData, slices: list[AnnData], pis: list[np.ndarray]
) -> tuple[AnnData, list[AnnData], list[float] | None, list[np.ndarray] | None]:
    """
    Align spatial coordinates of a list of slices to a center_slice.

    In other words, align:

        slices[0] --> center_slice

        slices[1] --> center_slice

        slices[2] --> center_slice

        ...

    Args:
        center_slice: Inferred center slice.
        slices: List of original slices to be aligned.
        pis: List of pi (``center_align()`` output) between center_slice and slices.
        output_params: If ``True``, additionally return angles of rotation (theta) and translations for each slice.
        matrix: If ``True`` and output_params is also ``True``, the rotation is
            return as a matrix instead of an angle for each slice.

    Returns:
        - Center slice with aligned spatial coordinates.
        - List of other slices with aligned spatial coordinates.

        If ``output_params = True``, additionally return:

        - List of angles of rotation (theta) for each slice.
        - List of translations [x_translation, y_translation] for each slice.
    """
    assert len(slices) == len(
        pis
    ), "'slices' should have the same length 'pis'. Please double check."
    aligned_coordinates = []
    rotation_angles = []
    translations = []

    for i in range(len(slices)):
        (
            source_coordinates,
            target_coordinates,
            rotation_angle,
            x_translation,
            y_translation,
        ) = generalized_procrustes_analysis(
            torch.Tensor(center_slice.obsm["spatial"])
            .to(pis[i].dtype)
            .to(pis[i].device),
            torch.Tensor(slices[i].obsm["spatial"]).to(pis[i].dtype).to(pis[i].device),
            pis[i],
        )
        rotation_angles.append(rotation_angle)
        translations.append((x_translation, y_translation))
        aligned_coordinates.append(target_coordinates)

    new_slices = []
    for i in range(len(slices)):
        _slice = slices[i].copy()
        _slice.obsm["spatial"] = aligned_coordinates[i].cpu().numpy()
        new_slices.append(_slice)

    new_center = center_slice.copy()
    new_center.obsm["spatial"] = source_coordinates.cpu().numpy()
    return new_center, new_slices, rotation_angles, translations


def plot_slice(
    slice: AnnData, color, ax: plt.Axes | None = None, s: float = 100
) -> None:
    """
    Plots slice spatial coordinates.

    Args:
        slice: Slice to be plotted.
        color: Scatterplot color, any format accepted by ``matplotlib``.
        ax: Pre-existing axes for the plot. Otherwise, call ``matplotlib.pyplot.gca()`` internally.
        s: Size of spots.
    """
    sns.scatterplot(
        x=slice.obsm["spatial"][:, 0],
        y=slice.obsm["spatial"][:, 1],
        linewidth=0,
        s=s,
        marker=".",
        color=color,
        ax=ax,
    )
    if ax:
        ax.invert_yaxis()
        ax.axis("off")


def generalized_procrustes_analysis(
    source_coordinates, target_coordinates, pi, is_partial=False
):
    """
    Finds and applies optimal rotation between spatial coordinates of two layers (may also do a reflection).

    Args:
        source_coordinates: np array of spatial coordinates (ex: sliceA.obs['spatial'])
        target_coordinates: np array of spatial coordinates (ex: sliceB.obs['spatial'])
        pi: mapping between the two layers output by PASTE
        return_params: Boolean of whether to return rotation angle and translations along with spatial coordiantes.
        return_as_matrix: Boolean of whether to return the rotation as a matrix or an angle.
        is_partial: Boolean of whether this is partial pairwise analysis or a total one


    Returns:
        Aligned spatial coordinates of X, Y, rotation angle, translation of X, translation of Y.
    """
    assert source_coordinates.shape[1] == 2
    assert target_coordinates.shape[1] == 2

    weighted_source = pi.sum(axis=1).matmul(source_coordinates)
    weighted_target = pi.sum(axis=0).matmul(target_coordinates)

    if is_partial:
        m = torch.sum(pi)
        weighted_source = weighted_source * (1.0 / m)
        weighted_target = weighted_target * (1.0 / m)

    source_coordinates = source_coordinates - weighted_source
    target_coordinates = target_coordinates - weighted_target

    covariance_matrix = target_coordinates.T.matmul(pi.T.matmul(source_coordinates))
    U, S, Vt = torch.linalg.svd(covariance_matrix, full_matrices=True)
    rotation_matrix = Vt.T.matmul(U.T)
    target_coordinates = rotation_matrix.matmul(target_coordinates.T).T
    M = torch.Tensor([[0, -1], [1, 0]]).double()
    rotation_angle = torch.arctan(
        torch.trace(M.matmul(covariance_matrix)) / torch.trace(covariance_matrix)
    )
    return (
        source_coordinates,
        target_coordinates,
        rotation_angle,
        weighted_source,
        weighted_target,
    )
