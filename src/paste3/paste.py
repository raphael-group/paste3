from typing import List, Tuple, Optional
import torch
import numpy as np
from anndata import AnnData
import ot
from ot.lp import emd
from sklearn.decomposition import NMF
import logging
from paste3.helper import (
    intersect,
    to_dense_array,
    dissimilarity_metric,
)

logger = logging.getLogger(__name__)


def pairwise_align(
    a_slice: AnnData,
    b_slice: AnnData,
    overlap_fraction: float = None,
    exp_dissim_matrix=None,
    alpha: float = 0.1,
    exp_dissim_metric: str = "kl",
    pi_init=None,
    a_spots_weight=None,
    b_spots_weight=None,
    norm: bool = False,
    numItermax: int = 200,
    backend=ot.backend.TorchBackend(),
    use_gpu: bool = True,
    return_obj: bool = False,
    maxIter=1000,
    optimizeTheta=True,
    eps=1e-4,
    do_histology: bool = False,
    armijo=False,
    **kwargs,
) -> Tuple[np.ndarray, Optional[int]]:
    """
    Calculates and returns optimal alignment of two slices.

    Args:
        a_slice: Slice A to align.
        b_slice: Slice B to align.
        alpha:  Alignment tuning parameter. Note: 0 <= alpha <= 1.
        exp_dissim_metric: Expression dissimilarity measure: ``'kl'`` or ``'euclidean'``.
        pi_init (array-like, optional): Initial mapping to be used in FGW-OT, otherwise default is uniform mapping.
        a_spots_weight (array-like, optional): Distribution of sliceA spots, otherwise default is uniform.
        b_spots_weight (array-like, optional): Distribution of sliceB spots, otherwise default is uniform.
        numItermax: Max number of iterations during FGW-OT.
        norm: If ``True``, scales spatial distances such that neighboring spots are at distance 1. Otherwise, spatial distances remain unchanged.
        backend: Type of backend to run calculations. For list of backends available on system: ``ot.backend.get_backend_list()``.
        use_gpu: If ``True``, use gpu. Otherwise, use cpu. Currently we only have gpu support for Pytorch.
        return_obj: If ``True``, additionally returns objective function output of FGW-OT.

    Returns:
        - Alignment of spots.

        If ``return_obj = True``, additionally returns:

        - Objective function output of FGW-OT.
    """

    if use_gpu and not torch.cuda.is_available():
        logger.info("GPU is not available, resorting to torch CPU.")
        use_gpu = False

    # subset for common genes
    common_genes = intersect(a_slice.var.index, b_slice.var.index)
    a_slice = a_slice[:, common_genes]
    b_slice = b_slice[:, common_genes]

    # check if slices are valid
    for slice in [a_slice, b_slice]:
        if not len(slice):
            raise ValueError(f"Found empty `AnnData`:\n{a_slice}.")

    # Backend
    nx = backend

    # Calculate spatial distances
    a_coordinates = a_slice.obsm["spatial"].copy()
    a_coordinates = nx.from_numpy(a_coordinates)
    b_coordinates = b_slice.obsm["spatial"].copy()
    b_coordinates = nx.from_numpy(b_coordinates)

    a_spatial_dist = ot.dist(a_coordinates, a_coordinates, metric="euclidean")
    b_spatial_dist = ot.dist(b_coordinates, b_coordinates, metric="euclidean")

    if isinstance(nx, ot.backend.TorchBackend):
        a_spatial_dist = a_spatial_dist.double()
        b_spatial_dist = b_spatial_dist.double()
    if use_gpu:
        a_spatial_dist = a_spatial_dist.cuda()
        b_spatial_dist = b_spatial_dist.cuda()

    # Calculate expression dissimilarity
    a_exp_dissim = to_dense_array(a_slice.X)
    b_exp_dissim = to_dense_array(b_slice.X)

    if isinstance(nx, ot.backend.TorchBackend) and use_gpu:
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

    if isinstance(nx, ot.backend.TorchBackend):
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
        m=overlap_fraction,
        G0=pi_init,
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
    slices: List[AnnData],
    slice_weights=None,
    alpha: float = 0.1,
    n_components: int = 15,
    threshold: float = 0.001,
    max_iter: int = 10,
    exp_dissim_metric: str = "kl",
    norm: bool = False,
    random_seed: Optional[int] = None,
    pi_inits: Optional[List[np.ndarray]] = None,
    spots_weights=None,
    backend=ot.backend.TorchBackend(),
    use_gpu: bool = True,
) -> Tuple[AnnData, List[np.ndarray]]:
    """
    Computes center alignment of slices.

    Args:
        initial_slice: Slice to use as the initialization for center alignment; Make sure to include gene expression and spatial information.
        slices: List of slices to use in the center alignment.
        slice_weights (array-like, optional): List of probability weights assigned to each slice; If ``None``, use uniform weights.
        alpha:  Alignment tuning parameter. Note: 0 <= alpha <= 1.
        n_components: Number of components in NMF decomposition.
        threshold: Threshold for convergence of feature_matrix and coeff_matrix during NMF decomposition.
        max_iter: Maximum number of iterations for our center alignment algorithm.
        exp_dissim_metric: Expression dissimilarity measure: ``'kl'`` or ``'euclidean'``.
        norm:  If ``True``, scales spatial distances such that neighboring spots are at distance 1. Otherwise, spatial distances remain unchanged.
        random_seed: Set random seed for reproducibility.
        pi_inits: Initial list of mappings between 'A' and 'slices' to solver. Otherwise, default will automatically calculate mappings.
        spots_weights (List[array-like], optional): Distributions of spots for each slice. Otherwise, default is uniform.
        backend: Type of backend to run calculations. For list of backends available on system: ``ot.backend.get_backend_list()``.
        use_gpu: If ``True``, use gpu. Otherwise, use cpu. Currently we only have gpu support for Pytorch.

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
        common_genes = intersect(common_genes, s.var.index)

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
            backend,
            use_gpu,
            exp_dissim_metric=exp_dissim_metric,
            norm=norm,
            pi_inits=pis,
            spot_weights=spots_weights,
        )
        feature_matrix, coeff_matrix = center_NMF(
            feature_matrix,
            coeff_matrix,
            slices,
            pis,
            slice_weights,
            n_components,
            random_seed,
            dissimilarity=exp_dissim_metric,
        )
        loss_new = np.dot(loss, slice_weights)
        iteration_count += 1
        loss_diff = abs(loss_init - loss_new)
        logger.info(f"Objective {loss_new}")
        logger.info(f"Difference: {loss_diff}")
        loss_init = loss_new
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
    feature_matrix,
    coeff_matrix,
    slices,
    center_coordinates,
    common_genes,
    alpha,
    backend,
    use_gpu,
    exp_dissim_metric="kl",
    norm=False,
    pi_inits=None,
    spot_weights=None,
    numItermax=200,
):
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
            norm=norm,
            numItermax=numItermax,
            return_obj=True,
            pi_init=pi_inits[i],
            b_spots_weight=spot_weights[i],
            backend=backend,
            use_gpu=use_gpu,
        )
        pis.append(pi)
        losses.append(loss)
    return pis, np.array(losses)


def center_NMF(
    W,
    H,
    slices,
    pis,
    lmbda,
    n_components,
    random_seed,
    dissimilarity="kl",
):
    logger.info("Solving Center Mapping NMF Problem.")
    n = W.shape[0]
    B = n * sum(
        [
            lmbda[i]
            * torch.matmul(pis[i], to_dense_array(slices[i].X).to(pis[i].device))
            for i in range(len(slices))
        ]
    )
    if dissimilarity.lower() == "euclidean" or dissimilarity.lower() == "euc":
        model = NMF(
            n_components=n_components,
            init="random",
            random_state=random_seed,
        )
    else:
        model = NMF(
            n_components=n_components,
            solver="mu",
            beta_loss="kullback-leibler",
            init="random",
            random_state=random_seed,
        )
    W_new = model.fit_transform(B.cpu().numpy())
    H_new = model.components_
    return W_new, H_new


def my_fused_gromov_wasserstein(
    M,
    C1,
    C2,
    p,
    q,
    alpha=0.5,
    m=None,
    G0=None,
    loss_fun="square_loss",
    armijo=False,
    numItermax=200,
    tol_rel=1e-9,
    tol_abs=1e-9,
    use_gpu=True,
    numItermaxEmd=100000,
    **kwargs,
):
    """
    Adapted fused_gromov_wasserstein with the added capability of defining a G_init (inital mapping).
    Also added capability of utilizing different POT backends to speed up computation.

    For more info, see: https://pythonot.github.io/gen_modules/ot.gromov.html
    """
    p, q = ot.utils.list_to_array(p, q)
    nx = ot.backend.get_backend(p, q, C1, C2, M)

    if m is not None:
        if m < 0:
            raise ValueError(
                "Problem infeasible. Parameter m should be greater" " than 0."
            )
        elif m > min(p.sum(), q.sum()):
            raise ValueError(
                "Problem infeasible. Parameter m should lower or"
                " equal to min(|p|_1, |q|_1)."
            )

        _log = {"err": []}
        count = 0
        dummy = 1
        _p = torch.cat([p, torch.Tensor([(q.sum() - m) / dummy] * dummy).to(p.device)])
        _q = torch.cat([q, torch.Tensor([(q.sum() - m) / dummy] * dummy).to(p.device)])

    if G0 is not None:
        G0 = (1 / nx.sum(G0)) * G0
        if use_gpu:
            G0 = G0.cuda()

    def f(G):
        constC, hC1, hC2 = ot.gromov.init_matrix(
            C1,
            C2,
            nx.sum(G, axis=1).reshape(-1, 1).to(C1.dtype),
            nx.sum(G, axis=0).reshape(1, -1).to(C2.dtype),
            loss_fun,
        )
        return ot.gromov.gwloss(constC, hC1, hC2, G)

    def df(G):
        constC, hC1, hC2 = ot.gromov.init_matrix(
            C1,
            C2,
            nx.sum(G, axis=1).reshape(-1, 1),
            nx.sum(G, axis=0).reshape(1, -1),
            loss_fun,
        )
        return ot.gromov.gwggrad(constC, hC1, hC2, G)

    if loss_fun == "kl_loss":
        armijo = True  # there is no closed form line-search with KL

    def line_search(cost, G, deltaG, Mi, cost_G, **kwargs):
        if m:
            nonlocal count
            # keep track of error only on every 10th iteration
            if count % 10 == 0:
                _log["err"].append(torch.norm(deltaG))
            count += 1

        if armijo:
            return ot.optim.line_search_armijo(
                cost, G, deltaG, Mi, cost_G, nx=nx, **kwargs
            )
        else:
            if m:
                return line_search_partial(
                    alpha, M, G, C1, C2, deltaG, loss_fun=loss_fun
                )
            else:
                return solve_gromov_linesearch(
                    G, deltaG, cost_G, C1, C2, M=0.0, reg=1.0, nx=nx, **kwargs
                )

    def lp_solver(a, b, M, **kwargs):
        if m:
            _M = torch.nn.functional.pad(M, pad=(0, dummy, 0, dummy), mode="constant")
            _M[-dummy:, -dummy:] = torch.max(M) * 1e2

            Gc, innerlog_ = emd(_p, _q, _M, 1000000, log=True)
            if innerlog_.get("warning"):
                raise ValueError(
                    "Error in EMD resolution: Increase the number of dummy points."
                )
            return Gc[: len(a), : len(b)], innerlog_
        else:
            return emd(a, b, M, numItermaxEmd, log=True)

    return_val = ot.optim.generic_conditional_gradient(
        p,
        q,
        (1 - alpha) * M,
        f,
        df,
        alpha,
        None,
        lp_solver,
        line_search,
        G0,
        log=True,
        numItermax=numItermax,
        stopThr=tol_rel,
        stopThr2=tol_abs,
        **kwargs,
    )

    res, log = return_val
    if m:
        log["partial_fgw_cost"] = log["loss"][-1]
        log["err"] = _log["err"]
    else:
        log["fgw_dist"] = log["loss"][-1]
        log["u"] = log["u"]
        log["v"] = log["v"]
    return res, log


def solve_gromov_linesearch(
    G, deltaG, cost_G, C1, C2, M, reg, alpha_min=None, alpha_max=None, nx=None, **kwargs
):
    """
    Solve the linesearch in the FW iterations

    Parameters
    ----------

    G : array-like, shape(ns,nt)
        The transport map at a given iteration of the FW
    deltaG : array-like (ns,nt)
        Difference between the optimal map found by linearization in the FW algorithm and the value at a given iteration
    cost_G : float
        Value of the cost at `G`
    C1 : array-like (ns,ns), optional
        Structure matrix in the source domain.
    C2 : array-like (nt,nt), optional
        Structure matrix in the target domain.
    M : array-like (ns,nt)
        Cost matrix between the features.
    reg : float
        Regularization parameter.
    alpha_min : float, optional
        Minimum value for alpha
    alpha_max : float, optional
        Maximum value for alpha
    nx : backend, optional
        If let to its default value None, a backend test will be conducted.
    Returns
    -------
    alpha : float
        The optimal step size of the FW
    fc : int
        nb of function call. Useless here
    cost_G : float
        The value of the cost for the next iteration


    .. _references-solve-linesearch:
    References
    ----------
    .. [24] Vayer Titouan, Chapel Laetitia, Flamary Rémi, Tavenard Romain and Courty Nicolas
        "Optimal Transport for structured data with application on graphs"
        International Conference on Machine Learning (ICML). 2019.
    """
    if nx is None:
        G, deltaG, C1, C2 = ot.utils.list_to_array(G, deltaG, C1, C2)

        if isinstance(M, int) or isinstance(M, float):
            nx = ot.backend.get_backend(G, deltaG, C1, C2)
        else:
            nx = ot.backend.get_backend(G, deltaG, C1, C2, M)

    dot = nx.dot(nx.dot(C1, deltaG), C2.T)
    a = -2 * reg * nx.sum(dot * deltaG)
    b = nx.sum(M * deltaG) - 2 * reg * (
        nx.sum(dot * G) + nx.sum(nx.dot(nx.dot(C1, G), C2.T) * deltaG)
    )

    alpha = ot.optim.solve_1d_linesearch_quad(a, b)
    if alpha_min is not None or alpha_max is not None:
        alpha = np.clip(alpha, alpha_min, alpha_max)

    # the new cost is deduced from the line search quadratic function
    cost_G = cost_G + a * (alpha**2) + b * alpha

    return alpha, 1, cost_G


def line_search_partial(reg, M, G, C1, C2, deltaG, loss_fun="square_loss"):
    constC, hC1, hC2 = ot.gromov.init_matrix(
        C1,
        C2,
        torch.sum(deltaG, axis=1).reshape(-1, 1),
        torch.sum(deltaG, axis=0).reshape(1, -1),
        loss_fun,
    )

    dot = torch.matmul(torch.matmul(C1, deltaG), C2.T)
    a = reg * torch.sum(dot * deltaG)
    b = (1 - reg) * torch.sum(M * deltaG) + 2 * reg * torch.sum(
        ot.gromov.gwggrad(constC, hC1, hC2, deltaG) * 0.5 * G
    )
    alpha = ot.optim.solve_1d_linesearch_quad(a, b)
    G = G + alpha * deltaG
    constC, hC1, hC2 = ot.gromov.init_matrix(
        C1,
        C2,
        torch.sum(G, axis=1).reshape(-1, 1),
        torch.sum(G, axis=0).reshape(1, -1),
        loss_fun,
    )
    cost_G = (1 - reg) * torch.sum(M * G) + reg * ot.gromov.gwloss(constC, hC1, hC2, G)
    return alpha, a, cost_G
