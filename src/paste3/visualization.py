from typing import List, Tuple, Optional
from anndata import AnnData
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


"""
    Functions to plot slices and align spatial coordinates after obtaining a mapping from PASTE.
"""


def stack_slices_pairwise(
    slices: List[AnnData],
    pis: List[np.ndarray],
    output_params: bool = False,
    matrix: bool = False,
    is_partial: bool = False,
) -> Tuple[List[AnnData], Optional[List[float]], Optional[List[np.ndarray]]]:
    """
    Align spatial coordinates of sequential pairwise slices.

    In other words, align:

        slices[0] --> slices[1] --> slices[2] --> ...

    Args:
        slices: List of slices.
        pis: List of pi (``pairwise_align()`` output) between consecutive slices.
        output_params: If ``True``, addtionally return angles of rotation (theta) and translations for each slice.
        matrix: If ``True`` and output_params is also ``True``, the rotation is
            return as a matrix instead of an angle for each slice.
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
    new_coor = []
    thetas = []
    translations = []
    result = generalized_procrustes_analysis(
        torch.Tensor(slices[0].obsm["spatial"]).to(pis[0].dtype).to(pis[0].device),
        torch.Tensor(slices[1].obsm["spatial"]).to(pis[0].dtype).to(pis[0].device),
        pis[0],
        is_partial=is_partial,
        output_params=output_params,
    )
    if output_params:
        S1, S2, theta, tX, tY = result
        thetas.append(theta)
        translations.append(tX)
        translations.append(tY)
    else:
        S1, S2 = result
    new_coor.append(S1)
    new_coor.append(S2)
    for i in range(1, len(slices) - 1):
        result = generalized_procrustes_analysis(
            new_coor[i],
            torch.Tensor(slices[i + 1].obsm["spatial"])
            .to(pis[i].dtype)
            .to(pis[i].device),
            pis[i],
            is_partial=is_partial,
            output_params=output_params,
        )
        if output_params:
            x, y, theta, tX, tY = result
            thetas.append(theta)
            translations.append(tY)
        else:
            x, y = result

        if is_partial:
            shift = new_coor[i][0, :] - x[0, :]
            y = y + shift
        new_coor.append(y)

    new_slices = []
    for i in range(len(slices)):
        s = slices[i].copy()
        s.obsm["spatial"] = new_coor[i].cpu().numpy()
        new_slices.append(s)

    if not output_params:
        return new_slices
    else:
        return new_slices, thetas, translations


def stack_slices_center(
    center_slice: AnnData,
    slices: List[AnnData],
    pis: List[np.ndarray],
    matrix: bool = False,
    output_params: bool = False,
) -> Tuple[AnnData, List[AnnData], Optional[List[float]], Optional[List[np.ndarray]]]:
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
    new_coor = []
    thetas = []
    translations = []

    for i in range(len(slices)):
        if not output_params:
            c, y = generalized_procrustes_analysis(
                torch.Tensor(center_slice.obsm["spatial"])
                .to(pis[i].dtype)
                .to(pis[i].device),
                torch.Tensor(slices[i].obsm["spatial"])
                .to(pis[i].dtype)
                .to(pis[i].device),
                pis[i],
            )
        else:
            c, y, theta, tX, tY = generalized_procrustes_analysis(
                torch.Tensor(center_slice.obsm["spatial"])
                .to(pis[i].dtype)
                .to(pis[i].device),
                torch.Tensor(slices[i].obsm["spatial"])
                .to(pis[i].dtype)
                .to(pis[i].device),
                pis[i],
                output_params=output_params,
                matrix=matrix,
            )
            thetas.append(theta)
            translations.append(tY)
        new_coor.append(y)

    new_slices = []
    for i in range(len(slices)):
        s = slices[i].copy()
        s.obsm["spatial"] = new_coor[i].cpu().numpy()
        new_slices.append(s)

    new_center = center_slice.copy()
    new_center.obsm["spatial"] = c.cpu().numpy()
    if not output_params:
        return new_center, new_slices
    else:
        return new_center, new_slices, thetas, translations


def plot_slice(
    sliceX: AnnData, color, ax: Optional[plt.Axes] = None, s: float = 100
) -> None:
    """
    Plots slice spatial coordinates.

    Args:
        sliceX: Slice to be plotted.
        color: Scatterplot color, any format accepted by ``matplotlib``.
        ax: Pre-existing axes for the plot. Otherwise, call ``matplotlib.pyplot.gca()`` internally.
        s: Size of spots.
    """
    sns.scatterplot(
        x=sliceX.obsm["spatial"][:, 0],
        y=sliceX.obsm["spatial"][:, 1],
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
    X, Y, pi, output_params=False, matrix=False, is_partial=False
):
    """
    Finds and applies optimal rotation between spatial coordinates of two layers (may also do a reflection).

    Args:
        X: np array of spatial coordinates (ex: sliceA.obs['spatial'])
        Y: np array of spatial coordinates (ex: sliceB.obs['spatial'])
        pi: mapping between the two layers output by PASTE
        output_params: Boolean of whether to return rotation angle and translations along with spatial coordiantes.
        matrix: Boolean of whether to return the rotation as a matrix or an angle.
        is_partial: Boolean of whether this is partial pairwise analysis or a total one


    Returns:
        Aligned spatial coordinates of X, Y, rotation angle, translation of X, translation of Y.
    """
    assert X.shape[1] == 2 and Y.shape[1] == 2

    tX = pi.sum(axis=1).matmul(X)
    tY = pi.sum(axis=0).matmul(Y)
    X = X - tX
    Y = Y - tY
    if is_partial:
        m = torch.sum(pi)
        X = X * (1.0 / m)
        Y = Y * (1.0 / m)
    H = Y.T.matmul(pi.T.matmul(X))
    U, S, Vt = torch.linalg.svd(H, full_matrices=True)
    R = Vt.T.matmul(U.T)
    Y = R.matmul(Y.T).T
    if output_params and not matrix:
        M = torch.Tensor([[0, -1], [1, 0]]).double()
        theta = torch.arctan(torch.trace(M.matmul(H)) / torch.trace(H))
        return X, Y, theta, tX, tY
    elif output_params and matrix:
        return X, Y, R, tX, tY
    else:
        return X, Y
