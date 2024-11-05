"""
This module provides functions to compute an optimal transport plan that aligns multiple tissue slices
using result of an ST experiment that includes a p genes by n spots transcript count matrix and coordinate
matrix of the spots
"""

import logging

import numpy as np
import ot
import torch
from anndata import AnnData
from ot.lp import emd
from sklearn.decomposition import NMF

from paste3.helper import (
    compute_slice_weights,
    dissimilarity_metric,
    get_common_genes,
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
    maxIter=1000,
    optimizeTheta=True,
    eps=1e-4,
    do_histology: bool = False,
) -> tuple[np.ndarray, dict | None]:
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
    # Convert every numpy array into tensors

    if use_gpu and not torch.cuda.is_available():
        logger.info("GPU is not available, resorting to torch CPU.")
        use_gpu = False

    device = "cuda" if use_gpu else "cpu"

    slices, _ = get_common_genes([a_slice, b_slice])
    a_slice, b_slice = slices

    a_dist = torch.Tensor(a_slice.obsm["spatial"]).double()
    b_dist = torch.Tensor(b_slice.obsm["spatial"]).double()

    a_exp_dissim = to_dense_array(a_slice.X).double().to(device)
    b_exp_dissim = to_dense_array(b_slice.X).double().to(device)

    a_spatial_dist = torch.cdist(a_dist, a_dist).double().to(device)
    b_spatial_dist = torch.cdist(b_dist, b_dist).double().to(device)

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
    exp_dissim_matrix = torch.Tensor(exp_dissim_matrix).double().to(device)

    if do_histology:
        # Calculate RGB dissimilarity
        rgb_dissim_matrix = (
            torch.cdist(
                torch.Tensor(a_slice.obsm["rgb"]).double(),
                torch.Tensor(b_slice.obsm["rgb"]).double(),
            )
            .to(exp_dissim_matrix.dtype)
            .to(device)
        )

        # Scale M_exp and rgb_dissim_matrix, obtain M by taking half from each
        rgb_dissim_matrix /= rgb_dissim_matrix[rgb_dissim_matrix > 0].max()
        rgb_dissim_matrix *= exp_dissim_matrix.max()
        exp_dissim_matrix = 0.5 * exp_dissim_matrix + 0.5 * rgb_dissim_matrix

    if a_spots_weight is None:
        a_spots_weight = torch.ones((a_slice.shape[0],)) / a_slice.shape[0]
        a_spots_weight = a_spots_weight.double().to(device)
    else:
        a_spots_weight = torch.Tensor(a_spots_weight).double().to(device)

    if b_spots_weight is None:
        b_spots_weight = torch.ones((b_slice.shape[0],)) / b_slice.shape[0]
        b_spots_weight = b_spots_weight.double().to(device)
    else:
        b_spots_weight = torch.Tensor(b_spots_weight).double().to(device)

    if norm:
        a_spatial_dist /= torch.min(a_spatial_dist[a_spatial_dist > 0])
        b_spatial_dist /= torch.min(b_spatial_dist[b_spatial_dist > 0])
        if overlap_fraction:
            a_spatial_dist /= a_spatial_dist[a_spatial_dist > 0].max()
            a_spatial_dist *= exp_dissim_matrix.max()
            b_spatial_dist /= b_spatial_dist[b_spatial_dist > 0].max()
            b_spatial_dist *= exp_dissim_matrix.max()

    if pi_init is not None:
        pi_init = torch.Tensor(pi_init).double().to(device)
        pi_init = (1 / torch.sum(pi_init)) * pi_init

    return my_fused_gromov_wasserstein(
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
    )


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

    device = "cuda" if use_gpu else "cpu"

    if slice_weights is None:
        slice_weights = len(slices) * [1 / len(slices)]

    if spots_weights is None:
        spots_weights = len(slices) * [None]

    slices, common_genes = get_common_genes(slices)
    initial_slice = initial_slice[:, common_genes]

    feature_matrix, coeff_matrix = center_NMF(
        initial_slice.X,
        slices,
        pi_inits,
        slice_weights,
        n_components,
        random_seed,
        exp_dissim_metric=exp_dissim_metric,
        device=device,
    )

    if pi_inits is None:
        pis = [None for _ in slices]

    # Initialize center_slice
    center_slice = AnnData(np.dot(feature_matrix, coeff_matrix))
    center_slice.var.index = common_genes
    center_slice.obs.index = initial_slice.obs.index
    center_slice.obsm["spatial"] = initial_slice.obsm["spatial"]

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
            center_slice.obsm["spatial"],
            common_genes,
            alpha,
            use_gpu,
            exp_dissim_metric=exp_dissim_metric,
            norm=norm,
            pi_inits=pis,
            spot_weights=spots_weights,
        )
        feature_matrix, coeff_matrix = center_NMF(
            feature_matrix,
            slices,
            pis,
            slice_weights,
            n_components,
            random_seed,
            exp_dissim_metric=exp_dissim_metric,
            device=device,
        )
        loss_new = np.dot(loss, slice_weights)
        iteration_count += 1
        loss_diff = abs(loss_init - loss_new)
        logger.info(f"Objective {loss_new} | Difference: {loss_diff}")
        loss_init = loss_new
    center_slice = initial_slice.copy()
    center_slice.X = np.dot(feature_matrix, coeff_matrix)
    center_slice.uns["paste_W"] = feature_matrix
    center_slice.uns["paste_H"] = coeff_matrix
    center_slice.uns["full_rank"] = (
        center_slice.shape[0]
        * compute_slice_weights(slice_weights, pis, slices, device).cpu().numpy()
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
    """Computes the optimal mappings \Pi^{(1)}, \ldots, \Pi^{(t)} given W (specified features)
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
        pi, loss = pairwise_align(
            center_slice,
            slices[i],
            alpha=alpha,
            exp_dissim_metric=exp_dissim_metric,
            pi_init=pi_inits[i],
            b_spots_weight=spot_weights[i],
            norm=norm,
            numItermax=numItermax,
            use_gpu=use_gpu,
        )
        pis.append(pi)
        losses.append(loss["loss"][-1])
    return pis, np.array(losses)


def center_NMF(
    feature_matrix: np.ndarray,
    slices: list[AnnData],
    pis: list[torch.Tensor] | None,
    slice_weights: list[float] | None,
    n_components: int,
    random_seed: float,
    exp_dissim_metric: str = "kl",
    device="cpu",
):
    """
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

    exp_dissim_metric = exp_dissim_metric.lower()
    nmf_model = NMF(
        n_components=n_components,
        init="random",
        random_state=random_seed,
        solver="cd" if exp_dissim_metric[:3] == "euc" else "mu",
        beta_loss="frobenius" if exp_dissim_metric[:3] == "euc" else "kullback-leibler",
    )

    if pis is not None:
        pis = [torch.Tensor(pi).double().to(device) for pi in pis]
        feature_matrix = (
            feature_matrix.shape[0]
            * compute_slice_weights(slice_weights, pis, slices, device).cpu().numpy()
        )

    new_feature_matrix = nmf_model.fit_transform(feature_matrix)
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
    numItermaxEmd: int | None = 100000,
    dummy: int | None = 1,
    **kwargs,
) -> tuple[np.ndarray, dict]:
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

    def f_loss(pi):
        """Compute the Gromov-Wasserstein loss for a given transport plan."""
        combined_spatial_cost, a_gradient, b_gradient = ot.gromov.init_matrix(
            a_spatial_dist,
            b_spatial_dist,
            torch.sum(pi, axis=1).reshape(-1, 1).to(a_spatial_dist.dtype),
            torch.sum(pi, axis=0).reshape(1, -1).to(b_spatial_dist.dtype),
            loss_fun,
        )
        return ot.gromov.gwloss(combined_spatial_cost, a_gradient, b_gradient, pi)

    def f_gradient(pi):
        """Compute the gradient of the Gromov-Wasserstein loss for a given transport plan."""
        combined_spatial_cost, a_gradient, b_gradient = ot.gromov.init_matrix(
            a_spatial_dist,
            b_spatial_dist,
            torch.sum(pi, axis=1).reshape(-1, 1),
            torch.sum(pi, axis=0).reshape(1, -1),
            loss_fun,
        )
        return ot.gromov.gwggrad(combined_spatial_cost, a_gradient, b_gradient, pi)

    def line_search(f_cost, pi, pi_diff, linearized_matrix, cost_pi, **kwargs):
        """Solve the linesearch in the fused wasserstein iterations"""
        if overlap_fraction:
            nonlocal count
            # keep track of error only on every 10th iteration
            if count % 10 == 0:
                _info["err"].append(torch.norm(pi_diff))
            count += 1

        if loss_fun == "kl_loss" or armijo:
            return ot.optim.line_search_armijo(
                f_cost, pi, pi_diff, linearized_matrix, cost_pi, **kwargs
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
            )
        return solve_gromov_linesearch(
            pi,
            pi_diff,
            cost_pi,
            a_spatial_dist,
            b_spatial_dist,
            exp_dissim_matrix=0.0,
            alpha=1.0,
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

    pi, info = ot.optim.generic_conditional_gradient(
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
    if overlap_fraction:
        info["err"] = _info["err"]
    return pi, info


def solve_gromov_linesearch(
    pi: torch.Tensor,
    pi_diff: torch.Tensor,
    cost_pi: float,
    a_spatial_dist: torch.Tensor,
    b_spatial_dist: torch.Tensor,
    exp_dissim_matrix: float,
    alpha: float,
    alpha_min: float | None = None,
    alpha_max: float | None = None,
    nx: str | None = None,
):
    """
    Perform a line search to optimize the transport plan with respect to the Gromov-Wasserstein loss.

    Parameters
    ----------
    pi : torch.Tensor
        The transport map at a given iteration of the FW
    pi_diff : torch.Tensor
        Difference between the optimal map found by linearization in the fused wasserstein algorithm and the value at a given iteration
    cost_pi : float
        Value of the cost at `G`
    a_spatial_dist : torch.Tensor
        Spot distance matrix in the first slice.
    b_spatial_dist : torch.Tensor
        Spot distance matrix in the second slice.
    exp_dissim_matrix : torch.Tensor
         Expression dissimilarity matrix between two slices.
    alpha : float
        Regularization parameter balancing transcriptional dissimilarity and spatial distance among aligned spots.
        Setting \alpha = 0 uses only transcriptional information, while \alpha = 1 uses only spatial coordinates.
    alpha_min : float, Optional
        Minimum value for alpha
    alpha_max : float, Optional
        Maximum value for alpha
    nx : str, Optional
        If let to its default value None, a backend test will be conducted.

    Returns
    -------
    minimal_cost : float
        The optimal step size of the fused wasserstein
    fc : int
        Number of function call. (Not used in this case)
    cost_pi : float
        The final cost after the update of the transport plan.

    .. _references-solve-linesearch:
    References
    ----------
    .. [24] Vayer Titouan, Chapel Laetitia, Flamary Rémi, Tavenard Romain and Courty Nicolas
        "Optimal Transport for structured data with application on graphs"
        International Conference on Machine Learning (ICML). 2019.
    """
    if nx is None:
        pi, pi_diff, a_spatial_dist, b_spatial_dist = ot.utils.list_to_array(
            pi, pi_diff, a_spatial_dist, b_spatial_dist
        )

        if isinstance(exp_dissim_matrix, (int | float)):
            nx = ot.backend.get_backend(pi, pi_diff, a_spatial_dist, b_spatial_dist)
        else:
            nx = ot.backend.get_backend(
                pi, pi_diff, a_spatial_dist, b_spatial_dist, exp_dissim_matrix
            )

    dot = nx.dot(nx.dot(a_spatial_dist, pi_diff), b_spatial_dist.T)
    a = -2 * alpha * nx.sum(dot * pi_diff)
    b = nx.sum(exp_dissim_matrix * pi_diff) - 2 * alpha * (
        nx.sum(dot * pi)
        + nx.sum(nx.dot(nx.dot(a_spatial_dist, pi), b_spatial_dist.T) * pi_diff)
    )

    minimal_cost = ot.optim.solve_1d_linesearch_quad(a, b)
    if alpha_min is not None or alpha_max is not None:
        minimal_cost = np.clip(minimal_cost, alpha_min, alpha_max)

    # the new cost is deduced from the line search quadratic function
    cost_pi = cost_pi + a * (minimal_cost**2) + b * minimal_cost

    return minimal_cost, 1, cost_pi


def line_search_partial(
    alpha: float,
    exp_dissim_matrix: torch.Tensor,
    pi: torch.Tensor,
    a_spatial_dist: torch.Tensor,
    b_spatial_dist: torch.Tensor,
    pi_diff: torch.Tensor,
    loss_fun: str = "square_loss",
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
    combined_spatial_cost, a_gradient, b_gradient = ot.gromov.init_matrix(
        a_spatial_dist,
        b_spatial_dist,
        torch.sum(pi_diff, axis=1).reshape(-1, 1),
        torch.sum(pi_diff, axis=0).reshape(1, -1),
        loss_fun,
    )

    dot = torch.matmul(torch.matmul(a_spatial_dist, pi_diff), b_spatial_dist.T)
    a = alpha * torch.sum(dot * pi_diff)
    b = (1 - alpha) * torch.sum(exp_dissim_matrix * pi_diff) + 2 * alpha * torch.sum(
        ot.gromov.gwggrad(combined_spatial_cost, a_gradient, b_gradient, pi_diff)
        * 0.5
        * pi
    )
    minimal_cost = ot.optim.solve_1d_linesearch_quad(a, b)
    pi = pi + minimal_cost * pi_diff
    combined_spatial_cost, a_gradient, b_gradient = ot.gromov.init_matrix(
        a_spatial_dist,
        b_spatial_dist,
        torch.sum(pi, axis=1).reshape(-1, 1),
        torch.sum(pi, axis=0).reshape(1, -1),
        loss_fun,
    )
    cost_G = (1 - alpha) * torch.sum(exp_dissim_matrix * pi) + alpha * ot.gromov.gwloss(
        combined_spatial_cost, a_gradient, b_gradient, pi
    )
    return minimal_cost, a, cost_G
