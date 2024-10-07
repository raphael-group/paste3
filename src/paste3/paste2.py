import numpy as np
import ot
from scipy.spatial import distance
from paste3.helper import (
    kl_divergence,
    intersect,
    to_dense_array,
    extract_data_matrix,
    glmpca_distance,
)
from paste3.paste import my_fused_gromov_wasserstein


def gwloss_partial(C1, C2, T, loss_fun="square_loss"):
    constC, hC1, hC2 = ot.gromov.init_matrix(
        C1,
        C2,
        np.sum(T, axis=1).reshape(-1, 1),
        np.sum(T, axis=0).reshape(1, -1),
        loss_fun,
    )
    return ot.gromov.gwloss(constC, hC1, hC2, T)


def gwgrad_partial(C1, C2, T, loss_fun="square_loss"):
    constC, hC1, hC2 = ot.gromov.init_matrix(
        C1,
        C2,
        np.sum(T, axis=1).reshape(-1, 1),
        np.sum(T, axis=0).reshape(1, -1),
        loss_fun,
    )
    return ot.gromov.gwggrad(constC, hC1, hC2, T)


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
    pi, log = my_fused_gromov_wasserstein(
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
    pi, log = my_fused_gromov_wasserstein(
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
