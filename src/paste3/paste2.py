import numpy as np
import ot
from scipy.spatial import distance
from ot.lp import emd
from paste3.helper import (
    kl_divergence,
    intersect,
    to_dense_array,
    extract_data_matrix,
    glmpca_distance,
    dissimilarity_metric,
)


def gwloss_partial(C1, C2, T, loss_fun="square_loss"):
    g = gwgrad_partial(C1, C2, T, loss_fun) * 0.5
    return np.sum(g * T)


def wloss(M, T):
    return np.sum(M * T)


def fgwloss_partial(alpha, M, C1, C2, T, loss_fun="square_loss"):
    return (1 - alpha) * wloss(M, T) + alpha * gwloss_partial(C1, C2, T, loss_fun)


def print_fgwloss_partial(alpha, M, C1, C2, T, loss_fun="square_loss"):
    print("W term is: " + str((1 - alpha) * wloss(M, T)))
    print("GW term is: " + str(alpha * gwloss_partial(C1, C2, T, loss_fun)))


def gwgrad_partial(C1, C2, T, loss_fun="square_loss"):
    """Compute the GW gradient, as one term in the FGW gradient.

    Note: we can not use the trick in Peyre16 as the marginals may not sum to 1.

    Parameters
    ----------
    C1: array of shape (n_p,n_p)
        intra-source cost matrix

    C2: array of shape (n_q,n_q)
        intra-target cost matrix

    T : array of shape(n_p, n_q)
        Transport matrix

    loss_fun

    Returns
    -------
    numpy.array of shape (n_p, n_q)
        gradient
    """
    if loss_fun == "square_loss":

        def f1(a):
            return a**2

        def f2(b):
            return b**2

        def h1(a):
            return a

        def h2(b):
            return 2 * b
    elif loss_fun == "kl_loss":

        def f1(a):
            return a * np.log(a + 1e-15) - a

        def f2(b):
            return b

        def h1(a):
            return a

        def h2(b):
            return np.log(b + 1e-15)

    A = np.dot(f1(C1), np.dot(T, np.ones(C2.shape[0]).reshape(-1, 1)))

    B = np.dot(
        np.dot(np.ones(C1.shape[0]).reshape(1, -1), T), f2(C2).T
    )  # does f2(C2) here need transpose?

    constC = A + B
    C = -np.dot(h1(C1), T).dot(h2(C2).T)
    tens = constC + C
    return tens * 2


def fgwgrad_partial(alpha, M, C1, C2, T, loss_fun="square_loss"):
    return (1 - alpha) * M + alpha * gwgrad_partial(C1, C2, T, loss_fun)


def line_search_partial(reg, M, G, C1, C2, deltaG, loss_fun="square_loss"):
    dot = np.dot(np.dot(C1, deltaG), C2.T)
    a = reg * np.sum(dot * deltaG)
    b = (1 - reg) * np.sum(M * deltaG) + 2 * reg * np.sum(
        gwgrad_partial(C1, C2, deltaG, loss_fun) * 0.5 * G
    )
    alpha = ot.optim.solve_1d_linesearch_quad(a, b)
    G = G + alpha * deltaG
    cost_G = fgwloss_partial(reg, M, C1, C2, G, loss_fun)
    return alpha, a, cost_G


def partial_fused_gromov_wasserstein(
    M,
    C1,
    C2,
    p,
    q,
    alpha,
    m=None,
    G0=None,
    loss_fun="square_loss",
    armijo=False,
    log=False,
    verbose=False,
    numItermax=1000,
    tol=1e-7,
    stopThr=1e-9,
    stopThr2=1e-9,
    numItermaxEmd=100000,
):
    if m is None:
        raise ValueError("Parameter m is not provided.")
    elif m < 0:
        raise ValueError("Problem infeasible. Parameter m should be greater" " than 0.")
    elif m > np.min((np.sum(p), np.sum(q))):
        raise ValueError(
            "Problem infeasible. Parameter m should lower or"
            " equal to min(|p|_1, |q|_1)."
        )

    if log:
        _log = {"err": []}
    count = 0
    dummy = 1

    def f(G):
        p = np.sum(G, axis=1).reshape(-1, 1)
        q = np.sum(G, axis=0).reshape(1, -1)
        constC, hC1, hC2 = ot.gromov.init_matrix(C1, C2, p, q, loss_fun)
        return ot.gromov.gwloss(constC, hC1, hC2, G)

    def df(G):
        p = np.sum(G, axis=1).reshape(-1, 1)
        q = np.sum(G, axis=0).reshape(1, -1)
        constC, hC1, hC2 = ot.gromov.init_matrix(C1, C2, p, q, loss_fun)
        return ot.gromov.gwggrad(constC, hC1, hC2, G)

    def line_search(cost, G, deltaG, Mi, cost_G, **kwargs):
        nonlocal count
        if log:
            # keep track of error only on every 10th iteration
            if count % 10 == 0:
                _log["err"].append(np.linalg.norm(deltaG))
        count += 1

        if armijo:
            return ot.optim.line_search_armijo(cost, G, deltaG, Mi, cost_G)
        else:
            return line_search_partial(
                alpha, M, G, C1, C2, deltaG, loss_fun="square_loss"
            )

    def lp_solver(a, b, Mi, **kwargs):
        dummy = kwargs.get("dummy") if kwargs.get("dummy") else 1

        _a = np.append(a, [(np.sum(b) - m) / dummy] * dummy)
        _b = np.append(b, [(np.sum(a) - m) / dummy] * dummy)

        _emd = np.pad(Mi, [(0, dummy)] * 2, mode="constant")
        _emd[-dummy:, -dummy:] = np.max(Mi) * 1e2

        Gc, innerlog_ = emd(_a, _b, _emd, numItermaxEmd, log=True)
        if innerlog_.get("warning"):
            raise ValueError(
                "Error in EMD resolution: Increase the number of dummy points."
            )
        return Gc[: len(a), : len(b)], innerlog_

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
        numItermax,
        stopThr,
        stopThr2,
        verbose,
        log,
        nb_dummies=dummy,
    )

    if log:
        res, log = return_val
        log["partial_fgw_cost"] = log["loss"][-1]
        log["err"] = _log["err"]
        return res, log
    else:
        return return_val


def partial_pairwise_align(
    sliceA,
    sliceB,
    s,
    alpha=0.1,
    armijo=False,
    dissimilarity="glmpca",
    use_rep=None,
    G_init=None,
    a_distribution=None,
    b_distribution=None,
    norm=True,
    return_obj=False,
    verbose=True,
    maxIter=1000,
    eps=1e-4,
    optimizeTheta=True,
):
    """
    Calculates and returns optimal *partial* alignment of two slices.

    param: sliceA - AnnData object
    param: sliceB - AnnData object
    param: s - Amount of mass to transport; Overlap percentage between the two slices. Note: 0 ≤ s ≤ 1
    param: alpha - Alignment tuning parameter. Note: 0 ≤ alpha ≤ 1
    param: armijo - Whether or not to use armijo (approximate) line search during conditional gradient optimization of Partial-FGW. Default is to use exact line search.
    param: dissimilarity - Expression dissimilarity measure: 'kl' or 'euclidean' or 'glmpca'. Default is glmpca.
    param: use_rep - If none, uses slice.X to calculate dissimilarity between spots, otherwise uses the representation given by slice.obsm[use_rep]
    param: G_init - initial mapping to be used in Partial-FGW OT, otherwise default is uniform mapping
    param: a_distribution - distribution of sliceA spots (1-d numpy array), otherwise default is uniform
    param: b_distribution - distribution of sliceB spots (1-d numpy array), otherwise default is uniform
    param: norm - scales spatial distances such that maximum spatial distance is equal to maximum gene expression dissimilarity
    param: return_obj - returns objective function value if True, nothing if False
    param: verbose - whether to print glmpca progress
    param maxIter - maximum number of iterations for glmpca
    param eps - convergence threshold for glmpca
    param optimizeTheta - whether to optimize overdispersion in glmpca

    return: pi - partial alignment of spots
    return: log['fgw_dist'] - objective function output of FGW-OT
    """
    m = s
    print("PASTE2 starts...")

    # subset for common genes
    common_genes = intersect(sliceA.var.index, sliceB.var.index)
    sliceA = sliceA[:, common_genes]
    sliceB = sliceB[:, common_genes]

    # Calculate spatial distances
    D_A = distance.cdist(sliceA.obsm["spatial"], sliceA.obsm["spatial"])
    D_B = distance.cdist(sliceB.obsm["spatial"], sliceB.obsm["spatial"])

    # Calculate expression dissimilarity
    A_X, B_X = (
        to_dense_array(extract_data_matrix(sliceA, use_rep)),
        to_dense_array(extract_data_matrix(sliceB, use_rep)),
    )

    M = dissimilarity_metric(
        dissimilarity,
        sliceA,
        sliceB,
        A_X,
        B_X,
        latent_dim=50,
        filter=True,
        verbose=verbose,
        maxIter=maxIter,
        eps=eps,
        optimizeTheta=optimizeTheta,
    )

    # init distributions
    if a_distribution is None:
        a = np.ones((sliceA.shape[0],)) / sliceA.shape[0]
    else:
        a = a_distribution

    if b_distribution is None:
        b = np.ones((sliceB.shape[0],)) / sliceB.shape[0]
    else:
        b = b_distribution

    if norm:
        D_A /= D_A[D_A > 0].min().min()
        D_B /= D_B[D_B > 0].min().min()

        """
        Code for normalizing distance matrix
        """
        D_A /= D_A[D_A > 0].max()
        # D_A *= 10
        D_A *= M.max()
        D_B /= D_B[D_B > 0].max()
        # D_B *= 10
        D_B *= M.max()
        """
        Code for normalizing distance matrix ends
        """
    pi, log = partial_fused_gromov_wasserstein(
        M,
        D_A,
        D_B,
        a,
        b,
        alpha=alpha,
        m=m,
        G0=G_init,
        loss_fun="square_loss",
        armijo=armijo,
        log=True,
        verbose=verbose,
        numItermax=maxIter,
    )

    if return_obj:
        return pi, log["partial_fgw_cost"]
    return pi


def partial_pairwise_align_histology(
    sliceA,
    sliceB,
    alpha=0.1,
    s=None,
    armijo=False,
    dissimilarity="glmpca",
    use_rep=None,
    G_init=None,
    a_distribution=None,
    b_distribution=None,
    norm=True,
    return_obj=False,
    verbose=False,
    numItermax=1000,
    **kwargs,
):
    """
    Optimal partial alignment of two slices using both gene expression and histological image information.

    sliceA, sliceB must be AnnData objects that contain .obsm['rgb'], which stores the RGB value of each spot in the histology image.
    """
    m = s
    print("PASTE2 starts...")

    # subset for common genes
    common_genes = intersect(sliceA.var.index, sliceB.var.index)
    sliceA = sliceA[:, common_genes]
    sliceB = sliceB[:, common_genes]

    # Calculate spatial distances
    D_A = distance.cdist(sliceA.obsm["spatial"], sliceA.obsm["spatial"])
    D_B = distance.cdist(sliceB.obsm["spatial"], sliceB.obsm["spatial"])

    # Calculate expression dissimilarity
    A_X, B_X = (
        to_dense_array(extract_data_matrix(sliceA, use_rep)),
        to_dense_array(extract_data_matrix(sliceB, use_rep)),
    )
    if dissimilarity.lower() == "euclidean" or dissimilarity.lower() == "euc":
        M_exp = distance.cdist(A_X, B_X)
    elif dissimilarity.lower() == "kl":
        s_A = A_X + 0.01
        s_B = B_X + 0.01
        M_exp = kl_divergence(s_A, s_B)
    elif dissimilarity.lower() == "glmpca":
        M_exp = glmpca_distance(A_X, B_X, latent_dim=50, filter=True, verbose=verbose)
    else:
        print("ERROR")
        exit(1)

    # Calculate RGB dissimilarity
    M_rgb = distance.cdist(sliceA.obsm["rgb"], sliceB.obsm["rgb"])

    # Scale M_exp and M_rgb, obtain M by taking half from each
    M_rgb /= M_rgb[M_rgb > 0].max()
    M_rgb *= M_exp.max()
    M = 0.5 * M_exp + 0.5 * M_rgb

    # init distributions
    if a_distribution is None:
        a = np.ones((sliceA.shape[0],)) / sliceA.shape[0]
    else:
        a = a_distribution

    if b_distribution is None:
        b = np.ones((sliceB.shape[0],)) / sliceB.shape[0]
    else:
        b = b_distribution

    if norm:
        D_A /= D_A[D_A > 0].min().min()
        D_B /= D_B[D_B > 0].min().min()

        """
        Code for normalizing distance matrix
        """
        D_A /= D_A[D_A > 0].max()
        D_A *= M.max()
        D_B /= D_B[D_B > 0].max()
        D_B *= M.max()
        """
        Code for normalizing distance matrix ends
        """

    # Run OT
    pi, log = partial_fused_gromov_wasserstein(
        M,
        D_A,
        D_B,
        a,
        b,
        alpha=alpha,
        m=m,
        G0=G_init,
        loss_fun="square_loss",
        armijo=armijo,
        log=True,
        verbose=verbose,
        numItermax=numItermax,
    )

    if return_obj:
        return pi, log["partial_fgw_cost"]
    return pi


def partial_pairwise_align_given_cost_matrix(
    sliceA,
    sliceB,
    M,
    s,
    alpha=0.1,
    armijo=False,
    G_init=None,
    a_distribution=None,
    b_distribution=None,
    norm=True,
    return_obj=False,
    verbose=False,
    numItermax=1000,
    **kwargs,
):
    m = s

    # subset for common genes
    common_genes = intersect(sliceA.var.index, sliceB.var.index)
    sliceA = sliceA[:, common_genes]
    sliceB = sliceB[:, common_genes]
    # print('Filtered all slices for common genes. There are ' + str(len(common_genes)) + ' common genes.')

    # Calculate spatial distances
    D_A = distance.cdist(sliceA.obsm["spatial"], sliceA.obsm["spatial"])
    D_B = distance.cdist(sliceB.obsm["spatial"], sliceB.obsm["spatial"])

    # init distributions
    if a_distribution is None:
        a = np.ones((sliceA.shape[0],)) / sliceA.shape[0]
    else:
        a = a_distribution

    if b_distribution is None:
        b = np.ones((sliceB.shape[0],)) / sliceB.shape[0]
    else:
        b = b_distribution

    if norm:
        D_A /= D_A[D_A > 0].min().min()
        D_B /= D_B[D_B > 0].min().min()

        """
        Code for normalizing distance matrix
        """
        D_A /= D_A[D_A > 0].max()
        D_A *= M.max()
        D_B /= D_B[D_B > 0].max()
        D_B *= M.max()
        """
        Code for normalizing distance matrix ends
        """

    # Run Partial OT
    pi, log = partial_fused_gromov_wasserstein(
        M,
        D_A,
        D_B,
        a,
        b,
        alpha=alpha,
        m=m,
        G0=G_init,
        loss_fun="square_loss",
        armijo=armijo,
        log=True,
        verbose=verbose,
        numItermax=numItermax,
    )

    if return_obj:
        return pi, log["partial_fgw_cost"]
    return pi
