"""
This module provides functions to compute an optimal transport plan that aligns multiple tissue slices
using result of an ST experiment that includes a p genes by n spots transcript count matrix and coordinate
matrix of the spots
"""

import logging
from typing import Any

import numpy as np
import ot
import torch
from anndata import AnnData
from ot.lp import emd
from sklearn.decomposition import NMF
from torchnmf.nmf import NMF as TorchNMF

from paste3.helper import (
    dissimilarity_metric,
    to_dense_array,
)

logger = logging.getLogger(__name__)


def pairwise_align(
    a_slice: AnnData,
    b_slice: AnnData,
    overlap_fraction: float | None = None,
    exp_dissim_matrix: np.ndarray = None,
    alpha: float = 0.1,
    exp_dissim_metric: str = "kl",
    pi_init=None,
    a_spots_weight=None,
    b_spots_weight=None,
    norm: bool = False,
    numItermax: int = 200,
    use_gpu: bool = True,
    return_obj: bool = False,
    maxIter=1000,
    optimizeTheta=True,
    eps=1e-4,
    do_histology: bool = False,
) -> tuple[np.ndarray, int | None]:
    r"""
    Returns a mapping :math:`( \Pi = [\pi_{ij}] )` between spots in one slice and spots in another slice
    while preserving gene expression and spatial distances of mapped spots, where :math:`\pi_{ij}` describes the probability that
    a spot i in the first slice is aligned to a spot j in the second slice.

    Given slices :math:`(X, D, g)` and :math:`(X', D', g')` containing :math:`n` and :math:`n'` spots, respectively,
    over the same :math:`p` genes, an expression cost function :math:`c`, and a parameter :math:`(0 \leq \alpha \leq 1)`,
    this function finds a mapping :math:`( \Pi \in \Gamma(g, g') )` that minimizes the following transport cost:

    .. math::
        F(\Pi; X, D, X', D', c, \alpha) = (1 - \alpha) \sum_{i,j} c(x_i, x'_j) \pi_{ij} + \alpha \sum_{i,j,k,l} (d_{ik} - d'_{jl})^2 \pi_{ij} \pi_{kl}'. \tag{1}

    subject to the regularity constraint that :math:`\pi` has to be a probabilistic coupling between :math:`g` and :math:`g'`:

    .. math::
        \pi \in \mathcal{F}(g, g') = \left\{ \pi \in \mathbb{R}^{n \times n'} \mid \pi \geq 0, \pi 1_{n'} = g, \pi^T 1_n = g' \right\}. \tag{2}

    Where:

        - :math:`X` and :math:`X'` represent the gene expression data for each slice,
        - :math:`D` and :math:`D'` represent the spatial distance matrices for each slice,
        - :math:`c` is a cost function applied to expression differences, and
        - :math:`\alpha` is a parameter that balances expression and spatial distance preservation in the mapping.
        - :math:`g` and :math:`g'` represent probability distribution over the spots in slice :math:`X` and :math:`X'`, respectively

    .. note::
        When the value for :math:`\textit {overlap_fraction}` is provided, this function solves the :math:`\textit{partial pairwise slice alignment problem}`
        by minimizing the same objective function as Equation (1), but with a different set of constraints that allow for unmapped spots.
        Given a parameter :math:`s \in [0, 1]` describing the fraction of mass to transport between :math:`g` and :math:`g'`, we define a set
        :math:`\mathcal{P}(g, g', s)` of :math:`s`-:math:`\textit{partial}` couplings between distributions :math:`g` and :math:`g'` as:

        .. math::
            \mathcal{P}(g, g', s) = \left\{ \pi \in \mathbb{R}^{n \times n'} \mid \pi \geq 0, \pi 1_{n'} \leq g, \pi^T 1_n \leq g', 1_n^T \pi 1_{n'} = s \right\}. \tag{3}

        Where:

            - :math:`s \in [0, 1]` is the overlap percentage between the two slices to align. (The constraint :math:`1_n^T \pi 1_{n'} = s` ensures that only the fraction of :math:`s` probability mass is transported)

    Parameters
    ----------
    a_slice : AnnData
        AnnData object containing data for the first slice.
    b_slice : AnnData
        AnnData object containing data for the second slice.
    overlap_fraction : float, optional
        Fraction of overlap between the two slices, must be between 0 and 1. If None, full alignment is performed.
    exp_dissim_matrix : np.ndarray, optional
        Precomputed expression dissimilarity matrix between two slices. If None, it will be computed.
    alpha : float, default=0.1
        Regularization parameter balancing transcriptional dissimilarity and spatial distance among aligned spots.
        Setting alpha = 0 uses only transcriptional information, while alpha = 1 uses only spatial coordinates.
    exp_dissim_metric : str, default="kl"
        Metric used to compute the expression dissimilarity with the following options:
        - 'kl' for Kullback-Leibler divergence between slices,
        - 'euc' for Euclidean distance,
        - 'gkl' for generalized Kullback-Leibler divergence,
        - 'selection_kl' for a selection-based KL approach,
        - 'pca' for Principal Component Analysis,
        - 'glmpca' for Generalized Linear Model PCA.
    pi_init : np.ndarray, optional
        Initial transport plan. If None, it will be computed.
    a_spots_weight : np.ndarray, optional
        Weight distribution for the spots in the first slice. If None, uniform weights are used.
    b_spots_weight : np.ndarray, optional
        Weight distribution for the spots in the second slice. If None, uniform weights are used.
    norm : bool, default=False
        If True, normalizes spatial distances.
    numItermax : int, default=200
        Maximum number of iterations for the optimization.
    use_gpu : bool, default=True
        Whether to use GPU for computations. If True but no GPU is available, will default to CPU.
    return_obj : bool, default=False
        If True, returns the optimization object along with the transport plan.
    maxIter : int, default=1000
        Maximum number of iterations for the dissimilarity calculation.
    optimizeTheta : bool, default=True
        Whether to optimize theta during dissimilarity calculation.
    eps : float, default=1e-4
        Tolerance level for convergence.
    do_histology : bool, default=False
        If True, incorporates RGB dissimilarity from histology data.

    Returns
    -------
    Tuple[np.ndarray, Optional[int]]
        - pi : np.ndarray
          Optimal transport plan for aligning the two slices.r
        - info : Optional[int]
          Information on the optimization process (if `return_obj` is True), else None.
    """
    if use_gpu and not torch.cuda.is_available():
        logger.info("GPU is not available, resorting to torch CPU.")
        use_gpu = False

    # subset for common genes
    common_genes = a_slice.var.index.intersection(b_slice.var.index)
    a_slice = a_slice[:, common_genes]
    b_slice = b_slice[:, common_genes]

    # check if slices are valid
    for slice in [a_slice, b_slice]:
        if not len(slice):
            raise ValueError(f"Found empty `AnnData`:\n{a_slice}.")

    # Backend
    nx = ot.backend.TorchBackend()

    # Calculate spatial distances
    a_coordinates = a_slice.obsm["spatial"].copy()
    a_coordinates = nx.from_numpy(a_coordinates)
    b_coordinates = b_slice.obsm["spatial"].copy()
    b_coordinates = nx.from_numpy(b_coordinates)

    a_spatial_dist = ot.dist(a_coordinates, a_coordinates, metric="euclidean")
    b_spatial_dist = ot.dist(b_coordinates, b_coordinates, metric="euclidean")

    a_spatial_dist = a_spatial_dist.double()
    b_spatial_dist = b_spatial_dist.double()
    if use_gpu:
        a_spatial_dist = a_spatial_dist.cuda()
        b_spatial_dist = b_spatial_dist.cuda()

    # Calculate expression dissimilarity
    a_exp_dissim = to_dense_array(a_slice.X)
    b_exp_dissim = to_dense_array(b_slice.X)

    if use_gpu:
        a_exp_dissim = a_exp_dissim.cuda()
        b_exp_dissim = b_exp_dissim.cuda()

    if exp_dissim_matrix is None:
        exp_dissim_matrix = dissimilarity_metric(
            exp_dissim_metric,
            a_slice,
            b_slice,
            a_exp_dissim,
            b_exp_dissim,
            latent_dim=50,
            filter=True,
            maxIter=maxIter,
            eps=eps,
            optimizeTheta=optimizeTheta,
        )

    if do_histology:
        # Calculate RGB dissimilarity
        rgb_dissim_matrix = (
            torch.cdist(
                torch.Tensor(a_slice.obsm["rgb"]).double(),
                torch.Tensor(b_slice.obsm["rgb"]).double(),
            )
            .to(exp_dissim_matrix.dtype)
            .to(exp_dissim_matrix.device)
        )

        # Scale M_exp and rgb_dissim_matrix, obtain M by taking half from each
        rgb_dissim_matrix /= rgb_dissim_matrix[rgb_dissim_matrix > 0].max()
        rgb_dissim_matrix *= exp_dissim_matrix.max()
        exp_dissim_matrix = 0.5 * exp_dissim_matrix + 0.5 * rgb_dissim_matrix

    # init distributions
    if a_spots_weight is None:
        a_spots_weight = nx.ones((a_slice.shape[0],)) / a_slice.shape[0]
    else:
        a_spots_weight = nx.from_numpy(a_spots_weight)

    if b_spots_weight is None:
        b_spots_weight = nx.ones((b_slice.shape[0],)) / b_slice.shape[0]
    else:
        b_spots_weight = nx.from_numpy(b_spots_weight)

    exp_dissim_matrix = exp_dissim_matrix.double()
    a_spots_weight = a_spots_weight.double()
    b_spots_weight = b_spots_weight.double()
    if use_gpu:
        exp_dissim_matrix = exp_dissim_matrix.cuda()
        a_spots_weight = a_spots_weight.cuda()
        b_spots_weight = b_spots_weight.cuda()

    if norm:
        a_spatial_dist /= nx.min(a_spatial_dist[a_spatial_dist > 0])
        b_spatial_dist /= nx.min(b_spatial_dist[b_spatial_dist > 0])
        if overlap_fraction:
            a_spatial_dist /= a_spatial_dist[a_spatial_dist > 0].max()
            a_spatial_dist *= exp_dissim_matrix.max()
            b_spatial_dist /= b_spatial_dist[b_spatial_dist > 0].max()
            b_spatial_dist *= exp_dissim_matrix.max()

    # Run OT
    if pi_init is not None and use_gpu:
        pi_init.cuda()
    pi, info = my_fused_gromov_wasserstein(
        exp_dissim_matrix,
        a_spatial_dist,
        b_spatial_dist,
        a_spots_weight,
        b_spots_weight,
        alpha=alpha,
        overlap_fraction=overlap_fraction,
        pi_init=pi_init,
        loss_fun="square_loss",
        numItermax=maxIter if overlap_fraction else numItermax,
        use_gpu=use_gpu,
    )
    if not overlap_fraction:
        info = info["fgw_dist"].item()

    if return_obj:
        return pi, info
    return pi


def center_align(
    initial_slice: AnnData,
    slices: list[AnnData],
    slice_weights=None,
    alpha: float = 0.1,
    n_components: int = 15,
    threshold: float = 0.001,
    max_iter: int = 10,
    exp_dissim_metric: str = "kl",
    norm: bool = False,
    random_seed: int | None = None,
    pi_inits: list[np.ndarray] | None = None,
    spots_weights=None,
    use_gpu: bool = True,
    fast: bool = False,
    pbar: Any = None,
) -> tuple[AnnData, list[np.ndarray]]:
    r"""
    Infers a "center" slice consisting of a low rank expression matrix :math:`X = WH` and a collection of
    :math:`\pi` of mappings from the spots of the center slice to the spots of each input slice.

    Given slices :math:`(X^{(1)}, D^{(1)}, g^{(1)}), \dots, (X^{(t)}, D^{(t)}, g^{(t)})` containing :math:`n_1, \dots, n_t`
    spots, respectively over the same :math:`p` genes, a spot distance matrix :math:`D \in \mathbb{R}^{n \times n}_{+}`,
    a distribution :math:`g` over :math:`n` spots, an expression cost function :math:`c`, a distribution
    :math:`\lambda \in \mathbb{R}^t_{+}` and parameters :math:`0 \leq \alpha \leq 1`, :math:`m \in \mathbb{N}`,
    find an expression matrix :math:`X = WH` where :math:`W \in \mathbb{R}^{p \times m}_{+}` and :math:`H \in \mathbb{R}^{m \times n}_{+}`,
    and mappings :math:`\Pi^{(q)} \in \Gamma(g, g^{(q)})` for each slice :math:`q = 1, \dots, t` that minimize the following objective:

    .. math::
        R(W, H, \Pi^{(1)}, \dots, \Pi^{(t)}) = \sum_q \lambda_q F(\Pi^{(q)}; WH, D, X^{(q)}, D^{(q)}, c, \alpha)

        = \sum_q \lambda_q \left[(1 - \alpha) \sum_{i,j} c(WH_{\cdot,i}, x^{(q)}_j) \pi^{(q)}_{ij} + \alpha \sum_{i,j,k,l} (d_{ik} - d^{(q)}_{jl})^2 \pi^{(q)}_{ij} \pi^{(q)}_{kl} \right].

    Where:

        - :math:`X^{q} = [x_{ij}] \in \mathbb{N}^{p \times n_t}` is a :math:`p` genes by :math:`n_t` spots transcript count matrix for :math:`q^{th}` slice,
        - :math:`D^{(q)}`, where :math:`d_ij = \parallel z_.i - z_.j \parallel` is the spatial distance between spot :math:`i` and :math:`j`, represents the spot pairwise distance matrix for :math:`q^{th}` slice,
        - :math:`c: \mathbb{R}^{p}_{+} \times \mathbb{R}^{p}_{+} \to \mathbb{R}_{+}`, is a function that measures a nonnegative cost between the expression profiles of two spots over all genes
        - :math:`\alpha` is a parameter balancing expression and spatial distance preservation,
        - :math:`W` and :math:`H` form the low-rank approximation of the center slice's expression matrix, and
        - :math:`\lambda_q` weighs each slice :math:`q` in the objective.

    Parameters
    ----------
    initial_slice : AnnData
        An AnnData object that represent a slice to be used as a reference data for alignment
    slices : List[AnnData]
        A list of AnnData objects that represent different slices to be aligned with the initial slice.
    slice_weights : List[float], optional
        Weights for each slice in the alignment process. If None, all slices are treated equally.
    alpha : float, default=0.1
        Regularization parameter balancing transcriptional dissimilarity and spatial distance among aligned spots.
        Setting \alpha = 0 uses only transcriptional information, while \alpha = 1 uses only spatial coordinates.
    n_components : int, default=15
        Number of components to use for the NMF.
    threshold : float, default=0.001
        Convergence threshold for the optimization process. The process stops when the change
        in loss is below this threshold.
    max_iter : int, default=10
        Maximum number of iterations for the optimization process.
    exp_dissim_metric : str, default="kl"
        The metric used to compute dissimilarity. Options include "euclidean" or "kl" for
        Kullback-Leibler divergence.
    norm : bool, default=False
        If True, normalizes spatial distances.
    random_seed : Optional[int], default=None
        Random seed for reproducibility.
    pi_inits : Optional[List[np.ndarray]], default=None
        Initial transport plans for each slice. If None, it will be computed.
    spots_weights : List[float], optional
        Weights for individual spots in each slices. If None, uniform distribution is used.
    use_gpu : bool, default=True
        Whether to use GPU for computations. If True but no GPU is available, will default to CPU.
    fast : bool, default=False
        Whether to use the fast (untested) torch nmf library
    pbar : Any, default=None
        Progress bar (tqdm or derived) for tracking the optimization process.
        Something that has an `update` method.
    Returns
    -------
    Tuple[AnnData, List[np.ndarray]]
        A tuple containing:
        - center_slice : AnnData
            The aligned AnnData object representing the center slice after optimization.
        - pis : List[np.ndarray]
            List of optimal transport distributions for each slice after alignment.

    Returns:
        - Inferred center slice with full and low dimensional representations (feature_matrix, coeff_matrix) of the gene expression matrix.
        - List of pairwise alignment mappings of the center slice (rows) to each input slice (columns).
    """
    if use_gpu and not torch.cuda.is_available():
        logger.info("GPU is not available, resorting to torch CPU.")
        use_gpu = False

    if slice_weights is None:
        slice_weights = len(slices) * [1 / len(slices)]

    if spots_weights is None:
        spots_weights = len(slices) * [None]

    # get common genes
    common_genes = initial_slice.var.index
    for s in slices:
        common_genes = common_genes.intersection(s.var.index)

    # subset common genes
    initial_slice = initial_slice[:, common_genes]
    for i in range(len(slices)):
        slices[i] = slices[i][:, common_genes]
    logger.info(
        "Filtered all slices for common genes. There are "
        + str(len(common_genes))
        + " common genes."
    )

    # Run initial NMF
    if exp_dissim_metric.lower() == "euclidean" or exp_dissim_metric.lower() == "euc":
        nmf_model = NMF(
            n_components=n_components,
            init="random",
            random_state=random_seed,
        )
    else:
        nmf_model = NMF(
            n_components=n_components,
            solver="mu",
            beta_loss="kullback-leibler",
            init="random",
            random_state=random_seed,
        )

    if pi_inits is None:
        pis = [None for i in range(len(slices))]
        feature_matrix = nmf_model.fit_transform(initial_slice.X)

    else:
        pis = pi_inits
        feature_matrix = nmf_model.fit_transform(
            initial_slice.shape[0]
            * sum(
                [
                    slice_weights[i] * np.dot(pis[i], to_dense_array(slices[i].X))
                    for i in range(len(slices))
                ]
            )
        )
    coeff_matrix = nmf_model.components_
    center_coordinates = initial_slice.obsm["spatial"]

    if not isinstance(center_coordinates, np.ndarray):
        logger.warning("A.obsm['spatial'] is not of type numpy array.")

    # Initialize center_slice
    center_slice = AnnData(np.dot(feature_matrix, coeff_matrix))
    center_slice.var.index = common_genes
    center_slice.obs.index = initial_slice.obs.index
    center_slice.obsm["spatial"] = center_coordinates

    # Minimize loss
    iteration_count = 0
    loss_init = 0
    loss_diff = 100
    while loss_diff > threshold and iteration_count < max_iter:
        logger.info("Iteration: " + str(iteration_count))
        pis, loss = center_ot(
            feature_matrix,
            coeff_matrix,
            slices,
            center_coordinates,
            common_genes,
            alpha,
            use_gpu,
            exp_dissim_metric=exp_dissim_metric,
            norm=norm,
            pi_inits=pis,
            spot_weights=spots_weights,
        )
        logger.info("center_ot done")
        feature_matrix, coeff_matrix = center_NMF(
            feature_matrix,
            slices,
            pis,
            slice_weights,
            n_components,
            random_seed,
            exp_dissim_metric=exp_dissim_metric,
            fast=fast,
        )
        loss_new = np.dot(loss, slice_weights)
        iteration_count += 1
        loss_diff = abs(loss_init - loss_new)
        logger.info(f"Objective {loss_new}")
        logger.info(f"Difference: {loss_diff}")
        loss_init = loss_new

        if pbar is not None:
            pbar.update(1)

    center_slice = initial_slice.copy()
    center_slice.X = np.dot(feature_matrix, coeff_matrix)
    center_slice.uns["paste_W"] = feature_matrix
    center_slice.uns["paste_H"] = coeff_matrix
    center_slice.uns["full_rank"] = (
        center_slice.shape[0]
        * sum(
            [
                slice_weights[i]
                * torch.matmul(pis[i], to_dense_array(slices[i].X).to(pis[i].device))
                for i in range(len(slices))
            ]
        )
        .cpu()
        .numpy()
    )
    center_slice.uns["obj"] = loss_init
    return center_slice, pis


# --------------------------- HELPER METHODS -----------------------------------


def center_ot(
    feature_matrix: np.ndarray,
    coeff_matrix: np.ndarray,
    slices: list[AnnData],
    center_coordinates: np.ndarray,
    common_genes: list[str],
    alpha: float,
    use_gpu: bool,
    exp_dissim_metric: str = "kl",
    norm: bool = False,
    pi_inits: list[np.ndarray] | None = None,
    spot_weights: list[float] | None = None,
    numItermax: int = 200,
) -> tuple[list[np.ndarray], np.ndarray]:
    r"""Computes the optimal mappings \Pi^{(1)}, \ldots, \Pi^{(t)} given W (specified features)
    and H (coefficient matrix) by solving the pairwise slice alignment problem between the
    center slice and each slices separately

    Parameters
    ----------
    feature_matrix : np.ndarray
        The matrix representing features extracted from the initial slice.
    coeff_matrix : np.ndarray
        The matrix representing the coefficients corresponding to the features.
    slices : List[AnnData]
        A list of AnnData objects representing the slices to be aligned with the center slice.
    center_coordinates : np.ndarray
        Spatial coordinates of the center slice.
    common_genes : Index
        Index of common genes shared among all slices for alignment.
    alpha : float
        Regularization parameter balancing transcriptional dissimilarity and spatial distance among aligned spots.
        Setting \alpha = 0 uses only transcriptional information, while \alpha = 1 uses only spatial coordinates.
    use_gpu : bool
        Whether to use GPU for computations. If True but no GPU is available, will default to CPU.
    exp_dissim_metric : str, default="kl"
        Metric used to compute the expression dissimilarity between slices. Options include "euclidean" and "kl".
    norm : bool, default=False
        If True, normalizes spatial distances.
    pi_inits : Optional[List[np.ndarray]], default=None
        Initial transport plans for each slice. If None, it will be computed.
    spot_weights : Optional[List[float]], default=None
        Weights for individual spots in each slice. If None, uniform distribution is used.
    numItermax : int, default=200
        Maximum number of iterations allowed for the optimization process.

    Returns
    -------
    Tuple[List[np.ndarray], np.ndarray]
        A tuple containing:
        - pis : List[np.ndarray]
            List of optimal transport plans for aligning each slice to the center slice.
        - losses : np.ndarray
            Array of loss values corresponding to each slice alignment.
    """

    center_slice = AnnData(np.dot(feature_matrix, coeff_matrix))
    center_slice.var.index = common_genes
    center_slice.obsm["spatial"] = center_coordinates

    if spot_weights is None:
        spot_weights = len(slices) * [None]

    pis = []
    losses = []
    logger.info("Solving Pairwise Slice Alignment Problem.")
    for i in range(len(slices)):
        logger.info(f"Slice {i}")
        pi, loss = pairwise_align(
            center_slice,
            slices[i],
            alpha=alpha,
            exp_dissim_metric=exp_dissim_metric,
            norm=norm,
            numItermax=numItermax,
            return_obj=True,
            pi_init=pi_inits[i],
            b_spots_weight=spot_weights[i],
            use_gpu=use_gpu,
        )
        pis.append(pi)
        losses.append(loss)
    return pis, np.array(losses)


def center_NMF(
    feature_matrix: np.ndarray,
    slices: list[AnnData],
    pis: list[torch.Tensor],
    slice_weights: list[float] | None,
    n_components: int,
    random_seed: float,
    exp_dissim_metric: str = "kl",
    fast: bool = False,
):
    r"""
    Finds two low-rank matrices \( W \) (feature matrix) and \( H \) (coefficient matrix) that approximate expression matrices of all
    slices by minimizing the following objective function:

    .. math::
            S(W, H) = \sum_q \lambda_q \sum_{i, j} c((WH)_i, x_j^{(q)}) \pi_{ij}^{(q)}

    Parameters
    ----------
    feature_matrix : np.ndarray
        The matrix representing the features extracted from the slices.
    slices : List[AnnData]
        A list of AnnData objects representing the slices involved in the mapping.
    pis : List[torch.Tensor]
        List of optimal transport plans for each slice, used to weight the features.
    slice_weights : List[float]
        Weights associated with each slice, indicating their importance in the NMF process.
    n_components : int
        The number of components to extract from the NMF.
    random_seed : int
        Random seed for reproducibility.
    exp_dissim_metric : str, default="kl"
        The metric used for measuring dissimilarity. Options include "euclidean" and "kl"
        for Kullback-Leibler divergence.
    fast : bool, default=False
        Whether to use the fast (untested) torch nmf library
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - new_feature_matrix : np.ndarray
            The updated matrix of features after applying NMF.
        - new_coeff_matrix : np.ndarray
            The updated matrix of coefficients resulting from the NMF decomposition.
    """
    logger.info("Solving Center Mapping NMF Problem.")
    n_features = feature_matrix.shape[0]
    weighted_features = n_features * sum(
        [
            slice_weights[i]
            * torch.matmul(pis[i], to_dense_array(slices[i].X).to(pis[i].device))
            for i in range(len(slices))
        ]
    )
    if exp_dissim_metric.lower() == "euclidean" or exp_dissim_metric.lower() == "euc":
        nmf_model = NMF(
            n_components=n_components,
            init="random",
            random_state=random_seed,
        )
    elif fast:
        nmf_model = TorchNMF(weighted_features.T.shape, rank=n_components).to(
            weighted_features.device
        )
    else:
        nmf_model = NMF(
            n_components=n_components,
            solver="mu",
            beta_loss="kullback-leibler",
            init="random",
            random_state=random_seed,
        )

    if fast:
        nmf_model.fit(weighted_features.T)
        new_feature_matrix = nmf_model.W.double().detach().cpu().numpy()
        new_coeff_matrix = nmf_model.H.T.detach().cpu().numpy()
    else:
        new_feature_matrix = nmf_model.fit_transform(weighted_features.cpu().numpy())
        new_coeff_matrix = nmf_model.components_
    return new_feature_matrix, new_coeff_matrix


def my_fused_gromov_wasserstein(
    exp_dissim_matrix: torch.Tensor,
    a_spatial_dist: torch.Tensor,
    b_spatial_dist: torch.Tensor,
    a_spots_weight: torch.Tensor,
    b_spots_weight: torch.Tensor,
    alpha: float | None = 0.5,
    overlap_fraction: float | None = None,
    pi_init: np.ndarray | None = None,
    loss_fun: str | None = "square_loss",
    armijo: bool | None = False,
    numItermax: int | None = 200,
    tol_rel: float | None = 1e-9,
    tol_abs: float | None = 1e-9,
    use_gpu: bool | None = True,
    numItermaxEmd: int | None = 100000,
    dummy: int | None = 1,
    **kwargs,
):
    """
    Computes a transport plan to align two weighted spatial distributions based on expression
    dissimilarity matrix and spatial distances, using the Gromov-Wasserstein framework.
    Also allows for partial alignment by specifying an overlap fraction.

    Parameters
    ----------
    exp_dissim_matrix : torch.Tensor
        Expression dissimilarity matrix between two slices.
    a_spatial_dist : torch.Tensor
        Spot distance matrix in the first slice.
    b_spatial_dist : torch.Tensor
        Spot distance matrix in the second slice.
    a_spots_weight : torch.Tensor
        Weight distribution for the spots in the first slice.
    b_spots_weight : torch.Tensor
        Weight distribution for the spots in the second slice.
    alpha : float, Optional
        Regularization parameter balancing transcriptional dissimilarity and spatial distance among aligned spots.
        Setting \alpha = 0 uses only transcriptional information, while \alpha = 1 uses only spatial coordinates.
    overlap_fraction : float, Option
        Fraction of overlap between the two slices, must be between 0 and 1. If None, full alignment is performed.
    pi_init : torch.Tensor, Optional
        Initial transport plan. If None, it will be computed.
    loss_fun : str, Optional
        Loss function to be used in optimization. Default is "square_loss".
    armijo : bool, Optional
        If True, uses Armijo rule for line search during optimization.
    numItermax : int, Optional
        Maximum number of iterations allowed for the optimization process.
    tol_rel : float, Optional
        Relative tolerance for convergence, by default 1e-9.
    tol_abs : float, Optional
        Absolute tolerance for convergence, by default 1e-9.
    use_gpu : bool, Optional
        Whether to use GPU for computations. If True but no GPU is available, will default to CPU.
    numItermaxEmd : int, Optional
        Maximum iterations for Earth Mover's Distance (EMD) solver.
    dummy : int, Optional
        Number of dummy points for partial overlap, by default 1.


    Returns
    -------
    Tuple[np.ndarray, Optional[dict]]
        - pi : np.ndarray
          Optimal transport plan that minimizes the fused gromov-wasserstein distance between
          two distributions
        - info : Optional[dict]
          A dictionary containing details of teh optimization process.

    For more info, see: https://pythonot.github.io/gen_modules/ot.gromov.html
    """
    a_spots_weight, b_spots_weight = ot.utils.list_to_array(
        a_spots_weight, b_spots_weight
    )
    nx = ot.backend.get_backend(
        a_spots_weight,
        b_spots_weight,
        a_spatial_dist,
        b_spatial_dist,
        exp_dissim_matrix,
    )

    if overlap_fraction is not None:
        if overlap_fraction < 0:
            raise ValueError(
                "Problem infeasible. Overlap fraction should be greater than 0."
            )
        if overlap_fraction > min(a_spots_weight.sum(), b_spots_weight.sum()):
            raise ValueError(
                "Problem infeasible. Overlap fraction should lower or equal to min(|p|_1, |q|_1)."
            )

        _info = {"err": []}
        count = 0
        _a_spots_weight = torch.cat(
            [
                a_spots_weight,
                torch.Tensor(
                    [(b_spots_weight.sum() - overlap_fraction) / dummy] * dummy
                ).to(a_spots_weight.device),
            ]
        )
        _b_spots_weight = torch.cat(
            [
                b_spots_weight,
                torch.Tensor(
                    [(b_spots_weight.sum() - overlap_fraction) / dummy] * dummy
                ).to(a_spots_weight.device),
            ]
        )

    if pi_init is not None:
        pi_init = (1 / nx.sum(pi_init)) * pi_init
        if use_gpu:
            pi_init = pi_init.cuda()

    def f_loss(pi):
        """Compute the Gromov-Wasserstein loss for a given transport plan."""
        combined_spatial_cost, a_gradient, b_gradient = ot.gromov.init_matrix(
            a_spatial_dist,
            b_spatial_dist,
            nx.sum(pi, axis=1).reshape(-1, 1).to(a_spatial_dist.dtype),
            nx.sum(pi, axis=0).reshape(1, -1).to(b_spatial_dist.dtype),
            loss_fun,
        )
        return ot.gromov.gwloss(combined_spatial_cost, a_gradient, b_gradient, pi)

    def f_gradient(pi):
        """Compute the gradient of the Gromov-Wasserstein loss for a given transport plan."""
        combined_spatial_cost, a_gradient, b_gradient = ot.gromov.init_matrix(
            a_spatial_dist,
            b_spatial_dist,
            nx.sum(pi, axis=1).reshape(-1, 1),
            nx.sum(pi, axis=0).reshape(1, -1),
            loss_fun,
        )
        return ot.gromov.gwggrad(combined_spatial_cost, a_gradient, b_gradient, pi)

    if loss_fun == "kl_loss":
        armijo = True  # there is no closed form line-search with KL

    def line_search(f_cost, pi, pi_diff, linearized_matrix, cost_pi, _, **kwargs):
        """Solve the linesearch in the fused wasserstein iterations"""
        if overlap_fraction:
            nonlocal count
            # keep track of error only on every 10th iteration
            if count % 10 == 0:
                _info["err"].append(torch.norm(pi_diff))
            count += 1

        if armijo:
            return ot.optim.line_search_armijo(
                f_cost, pi, pi_diff, linearized_matrix, cost_pi, nx=nx, **kwargs
            )
        if overlap_fraction:
            return line_search_partial(
                alpha,
                exp_dissim_matrix,
                pi,
                a_spatial_dist,
                b_spatial_dist,
                pi_diff,
                loss_fun=loss_fun,
                Mi=linearized_matrix,
                cost_pi=cost_pi,
            )
        return ot.gromov.solve_gromov_linesearch(
            G=pi,
            deltaG=pi_diff,
            cost_G=cost_pi,
            C1=a_spatial_dist,
            C2=b_spatial_dist,
            M=0.0,
            reg=2 * 1.0,
            nx=nx,
            **kwargs,
        )

    def lp_solver(
        a_spots_weight,
        b_spots_weight,
        exp_dissim_matrix,
    ):
        """Solves the Earth Movers distance problem and returns the OT matrix"""
        if overlap_fraction:
            _exp_dissim_matrix = torch.nn.functional.pad(
                exp_dissim_matrix, pad=(0, dummy, 0, dummy), mode="constant"
            )
            _exp_dissim_matrix[-dummy:, -dummy:] = torch.max(exp_dissim_matrix) * 1e2

            _pi, _innerlog = emd(
                _a_spots_weight, _b_spots_weight, _exp_dissim_matrix, 1000000, log=True
            )
            if _innerlog.get("warning"):
                raise ValueError(
                    "Error in EMD resolution: Increase the number of dummy points."
                )
            return _pi[: len(a_spots_weight), : len(b_spots_weight)], _innerlog
        return emd(
            a_spots_weight,
            b_spots_weight,
            exp_dissim_matrix,
            numItermaxEmd,
            log=True,
        )

    return_val = ot.optim.generic_conditional_gradient(
        a=a_spots_weight,
        b=b_spots_weight,
        M=(1 - alpha) * exp_dissim_matrix,
        f=f_loss,
        df=f_gradient,
        reg1=alpha,
        reg2=None,
        lp_solver=lp_solver,
        line_search=line_search,
        G0=pi_init,
        log=True,
        numItermax=numItermax,
        stopThr=tol_rel,
        stopThr2=tol_abs,
        **kwargs,
    )

    pi, info = return_val
    if overlap_fraction:
        info["partial_fgw_cost"] = info["loss"][-1]
        info["err"] = _info["err"]
    else:
        info["fgw_dist"] = info["loss"][-1]
        info["u"] = info["u"]
        info["v"] = info["v"]
    return pi, info


def line_search_partial(
    alpha: float,
    exp_dissim_matrix: torch.Tensor,
    pi: torch.Tensor,
    a_spatial_dist: torch.Tensor,
    b_spatial_dist: torch.Tensor,
    pi_diff: torch.Tensor,
    loss_fun: str = "square_loss",
    Mi=None,
    cost_pi=None,
):
    """
    Solve the linesearch in the fused wasserstein iterations for partially overlapping slices

    Parameters
    ----------
    alpha : float
        Regularization parameter balancing transcriptional dissimilarity and spatial distance among aligned spots.
        Setting \alpha = 0 uses only transcriptional information, while \alpha = 1 uses only spatial coordinates.
    exp_dissim_matrix : torch.Tensor
        Expression dissimilarity matrix between two slices.
    pi : torch.Tensor
        The transport map at a given iteration of the FW.
    a_spatial_dist : torch.Tensor
        Spot distance matrix in the first slice.
    b_spatial_dist : torch.Tensor
        Spot distance matrix in the first slice.
    pi_diff : torch.Tensor
        Difference between the optimal map found by linearization in the fused wasserstein algorithm and the value at a given iteration
    loss_fun : str, Optional
        Loss function to be used in optimization. Default is "square_loss".

    Returns
    -------
    minimal_cost : float
        The optimal step size of the fused wasserstein
    a : float
        The computed value for the first cost component.
    cost_G : float
        The final cost after the update of the transport plan.
    """

    def h2(a):
        return a * 2

    def f(pi, type="gradient"):
        """Compute the gradient of the Gromov-Wasserstein loss for a given transport plan."""
        combined_spatial_cost, a_gradient, b_gradient = ot.gromov.init_matrix(
            a_spatial_dist,
            b_spatial_dist,
            torch.sum(pi, axis=1).reshape(-1, 1),
            torch.sum(pi, axis=0).reshape(1, -1),
            loss_fun="square_loss",
        )
        if type == "gradient":
            return ot.gromov.gwggrad(combined_spatial_cost, a_gradient, b_gradient, pi)
        return ot.gromov.gwloss(combined_spatial_cost, a_gradient, b_gradient, pi)

    dot = torch.matmul(torch.matmul(a_spatial_dist, pi_diff), b_spatial_dist.T)
    a = alpha * torch.sum(dot * pi_diff)
    a_ = alpha * f(pi_diff, type='loss')

    dot_ = torch.matmul(torch.matmul(a_spatial_dist, pi_diff), h2(b_spatial_dist).T)
    a__ = -2 * alpha * torch.sum(dot_ * pi_diff)
    try:
        assert np.isclose(a_, a__)
    except AssertionError:
        print('what happened here')
    b = torch.sum(Mi * pi_diff) - alpha * (
        torch.sum(dot_ * -pi)
        + torch.sum(
            torch.matmul(torch.matmul(a_spatial_dist, pi), h2(b_spatial_dist).T)
            * pi_diff
        )
    )
    b_ = (1 - alpha) * torch.sum(exp_dissim_matrix * pi_diff) + 2 * alpha * torch.sum(
        f(pi_diff) * 0.5 * pi)
    assert np.isclose(b, b_)

    minimal_cost = ot.optim.solve_1d_linesearch_quad(a, b)
    minimal_cost_ = ot.optim.solve_1d_linesearch_quad(a_, b)
    pi = pi + minimal_cost * pi_diff
    cost_G = (1 - alpha) * torch.sum(exp_dissim_matrix * pi) + alpha * f(pi, type='loss')
    cost_G_ = cost_pi + a_ * (minimal_cost_**2) + b * minimal_cost_

    try:
        assert np.isclose(cost_G_, cost_G)
    except AssertionError:
        print('why???')

    return minimal_cost_, a, cost_G_
