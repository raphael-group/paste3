"""
This module provides helper functions to compute an optimal transport plan that aligns multiple tissue slices
using result of an ST experiment that includes a p genes by n spots transcript count matrix and coordinate
matrix of the spots
"""

import logging

import anndata as ad
import numpy as np
import ot
import scanpy as sc
import scipy
import torch
from anndata import AnnData
from scipy.spatial import distance

from paste3.glmpca import glmpca

logger = logging.getLogger(__name__)


def kl_divergence(a_exp_dissim, b_exp_dissim):
    """
    Calculates the Kullback-Leibler divergence between two distributions.

    Parameters
    ----------
    a_exp_dissim : torch.Tensor
        A tensor representing the first probability distribution.

    b_exp_dissim : torch.Tensor
       A tensor representing the second probability distribution.

    Returns
    -------
    divergence : torch.Tensor
       A tensor containing the Kullback-Leibler divergence for each sample.
    """
    assert (
        a_exp_dissim.shape[1] == b_exp_dissim.shape[1]
    ), "X and Y do not have the same number of features."

    a_exp_dissim = a_exp_dissim / a_exp_dissim.sum(axis=1, keepdims=True)
    b_exp_dissim = b_exp_dissim / b_exp_dissim.sum(axis=1, keepdims=True)
    a_log_exp_dissim = a_exp_dissim.log()
    b_log_exp_dissim = b_exp_dissim.log()
    a_weighted_dissim_sum = torch.sum(a_exp_dissim * a_log_exp_dissim, axis=1)[
        torch.newaxis, :
    ]
    return a_weighted_dissim_sum.T - torch.matmul(a_exp_dissim, b_log_exp_dissim.T)


def generalized_kl_divergence(a_exp_dissim, b_exp_dissim):
    """
    Computes the generalized Kullback-Leibler (KL) divergence between two distributions

    Parameters
    ----------
    a_exp_dissim : torch.Tensor
        A tensor representing first probability distribution.

    b_exp_dissim : torch.Tensor
        A tensor representing the second probability distribution.

    Returns
    -------
    divergence : torch.Tensor
        A tensor containing the generalized Kullback-Leibler divergence for each sample.
    """
    assert (
        a_exp_dissim.shape[1] == b_exp_dissim.shape[1]
    ), "X and Y do not have the same number of features."

    a_log_exp_dissim = a_exp_dissim.log()
    b_log_exp_dissim = b_exp_dissim.log()
    a_weighted_dissim_sum = torch.sum(a_exp_dissim * a_log_exp_dissim, axis=1)[
        torch.newaxis, :
    ]
    divergence = a_weighted_dissim_sum.T - torch.matmul(
        a_exp_dissim, b_log_exp_dissim.T
    )
    sum_a_exp_dissim = torch.sum(a_exp_dissim, axis=1)
    sum_b_exp_dissim = torch.sum(b_exp_dissim, axis=1)
    return (divergence.T - sum_a_exp_dissim).T + sum_b_exp_dissim.T


def glmpca_distance(
    a_exp_dissim,
    b_exp_dissim,
    latent_dim=50,
    filter=True,
    maxIter=1000,
    eps=1e-4,
    optimizeTheta=True,
):
    """
    Computes the distance between two distributions after reducing dimensionality using GLM-PCA.

    Parameters
    ----------
    a_exp_dissim : torch.Tensor
        A tensor representing first probability distribution.

    b_exp_dissim : torch.Tensor
        A tensor representing the second probability distribution.

    latent_dim : int, Optional
        Number of latent dimensions for GLM-PCA reduction.

    filter : bool, Optional
        Whether to filter features based on top gene counts before GLM-PCA.

    maxIter : int, Optional
        Maximum number of iterations for GLM-PCA.

    eps : float, Optional
        Convergence threshold for GLM-PCA.

    optimizeTheta : bool, Optional
        If True, optimizes theta during GLM-PCA.

    Returns
    -------
    np.ndarray
        Distances between the two distributions after dimensionality reduction.
    """
    assert (
        a_exp_dissim.shape[1] == b_exp_dissim.shape[1]
    ), "X and Y do not have the same number of features."

    joint_dissim_matrix = torch.vstack((a_exp_dissim, b_exp_dissim))
    if filter:
        gene_umi_counts = torch.sum(joint_dissim_matrix, axis=0).cpu().numpy()
        top_indices = np.sort((-gene_umi_counts).argsort(kind="stable")[:2000])
        joint_dissim_matrix = joint_dissim_matrix[:, top_indices]

    logging.info("Starting GLM-PCA...")
    res = glmpca(
        joint_dissim_matrix.T.cpu().numpy(),  # TODO: Use Tensors
        latent_dim,
        penalty=1,
        ctl={"maxIter": maxIter, "eps": eps, "optimizeTheta": optimizeTheta},
    )
    reduced_joint_dissim_matrix = res["factors"]
    logging.info("GLM-PCA finished.")

    a_exp_dissim = reduced_joint_dissim_matrix[: a_exp_dissim.shape[0], :]
    b_exp_dissim = reduced_joint_dissim_matrix[a_exp_dissim.shape[0] :, :]
    return distance.cdist(a_exp_dissim, b_exp_dissim)


def pca_distance(a_slice, b_slice, n_top_genes, latent_dim):
    """
    Computes pairwise distances between two distributions slices after dimensionality
    reduction using PCA.

    Parameters
    ----------
    a_slice : AnnData
        AnnData object representing the first slice.

    b_slice : AnnData
        AnnData object representing the second slice.

    n_top_genes : int
        Number of highly variable genes to select for PCA.

    latent_dim : int
        Number of principal components to retain in the PCA reduction.

    Returns
    -------
    distances : np.ndarray
        Distances between the two distributions after dimensionality reduction
    """
    joint_adata = ad.concat([a_slice, b_slice])
    sc.pp.normalize_total(joint_adata, inplace=True)
    sc.pp.log1p(joint_adata)
    sc.pp.highly_variable_genes(
        joint_adata, flavor="seurat", n_top_genes=n_top_genes, inplace=True, subset=True
    )
    sc.pp.pca(joint_adata, latent_dim)
    joint_data_matrix = joint_adata.obsm["X_pca"]
    return distance.cdist(
        joint_data_matrix[: a_slice.shape[0], :],
        joint_data_matrix[a_slice.shape[0] :, :],
    )


def high_umi_gene_distance(a_exp_dissim, b_exp_dissim, n):
    """
    Computes the Kullback-Leibler (KL) divergence between two distribution
    using genes with highest UMI counts.

    Parameters
    ----------
    a_exp_dissim : torch.Tensor
        A tensor representing the first probability distribution.

    b_exp_dissim : torch.Tensor
        A tensor representing the second probability distribution.

    n : int
        Number of genes with the highest UMI counts to select for computing
        the KL divergence.

    Returns
    -------
    torch.Tensor
        KL divergence matrix between two distributions.

    """
    assert (
        a_exp_dissim.shape[1] == b_exp_dissim.shape[1]
    ), "X and Y do not have the same number of features."

    joint_dissim_matrix = torch.vstack((a_exp_dissim, b_exp_dissim))
    gene_umi_counts = torch.sum(joint_dissim_matrix, axis=0).cpu().numpy()
    top_indices = np.sort((-gene_umi_counts).argsort(kind="stable")[:n])
    a_exp_dissim = a_exp_dissim[:, top_indices]
    b_exp_dissim = b_exp_dissim[:, top_indices]
    a_exp_dissim += torch.tile(
        0.01 * (torch.sum(a_exp_dissim, axis=1) / a_exp_dissim.shape[1]),
        (a_exp_dissim.shape[1], 1),
    ).T
    b_exp_dissim += torch.tile(
        0.01 * (torch.sum(b_exp_dissim, axis=1) / b_exp_dissim.shape[1]),
        (b_exp_dissim.shape[1], 1),
    ).T
    return kl_divergence(a_exp_dissim, b_exp_dissim)


def norm_and_center_coordinates(spatial_dist):
    """
    Normalizes and centers spatial coordinates by subtracting the mean and
    scaling by the minimum pairwise distance

    Parameters
    ----------
    spatial_dist : np.ndarray
        Spot distance matrix of a slice.

    Returns
    -------
    np.ndarray
        Normalized and centered spatial coordinates.
    """
    return (spatial_dist - spatial_dist.mean(axis=0)) / min(
        scipy.spatial.distance.pdist(spatial_dist)
    )


def to_dense_array(X):
    """Converts a sparse matrix into a dense one"""
    np_array = np.array(X.todense()) if isinstance(X, scipy.sparse.csr.spmatrix) else X
    return torch.Tensor(np_array).double()


def filter_for_common_genes(slices: list[AnnData]) -> None:
    """
    Filters a list of AnnData objects to retain only the common genes across
    all slices.

    Parameters
    ----------
    slices: List[AnnData]
        A list of AnnData objects that represent different slices.
    """
    assert len(slices) > 0, "Cannot have empty list."

    common_genes = slices[0].var.index
    for s in slices:
        common_genes = common_genes.intersection(s.var.index)
    for i in range(len(slices)):
        slices[i] = slices[i][:, common_genes]
    logging.info(
        "Filtered all slices for common genes. There are "
        + str(len(common_genes))
        + " common genes."
    )


def match_spots_using_spatial_heuristic(
    a_spatial_dist, b_spatial_dist, use_ot: bool = True
) -> np.ndarray:
    """
    Matches spatial coordinates between two datasets using either optimal
    transport or bipartite matching based on spatial proximity.

    Parameters
    ----------
    a_spatial_dist : np.ndarray
        Spot distance matrix in the first slice.

    b_spatial_dist : np.ndarray
        Spot distance matrix in the second slice.

    use_ot : bool, Optional
        If True, matches spots using optimal transport (OT); if False,
        uses minimum-weight full bipartite matching.

    Returns
    -------
    np.ndarray
        A transport matrix (`pi`) representing matching weights between
        the points in `a_spatial_dist` and `b_spatial_dist`.

    """

    len_a, len_b = len(a_spatial_dist), len(b_spatial_dist)
    a_spatial_dist, b_spatial_dist = (
        norm_and_center_coordinates(a_spatial_dist),
        norm_and_center_coordinates(b_spatial_dist),
    )
    inter_slice_spot_dist = scipy.spatial.distance_matrix(
        a_spatial_dist, b_spatial_dist
    )
    if use_ot:
        pi = ot.emd(
            np.ones(len_a) / len_a, np.ones(len_b) / len_b, inter_slice_spot_dist
        )
    else:
        row_ind, col_ind = scipy.sparse.csgraph.min_weight_full_bipartite_matching(
            scipy.sparse.csr_matrix(inter_slice_spot_dist)
        )
        pi = np.zeros((len_a, len_b))
        pi[row_ind, col_ind] = 1 / max(len_a, len_b)
        if len_a < len_b:
            pi[:, [(j not in col_ind) for j in range(len_b)]] = 1 / (len_a * len_b)
        elif len_b < len_a:
            pi[[(i not in row_ind) for i in range(len_a)], :] = 1 / (len_a * len_b)
    return pi


def kl_divergence_backend(a_exp_dissim, b_exp_dissim):
    """
    Calculates the Kullback-Leibler divergence between two distributions.

    Parameters
    ----------
    a_exp_dissim : torch.Tensor
        A tensor representing the first probability distribution.

    b_exp_dissim : torch.Tensor
        A tensor representing the second probability distribution.

    Returns
    -------
    divergence : np.ndarray
        A tensor containing the Kullback-Leibler divergence for each sample.
    """
    assert (
        a_exp_dissim.shape[1] == b_exp_dissim.shape[1]
    ), "X and Y do not have the same number of features."

    nx = ot.backend.get_backend(a_exp_dissim, b_exp_dissim)

    a_exp_dissim = a_exp_dissim / nx.sum(a_exp_dissim, axis=1, keepdims=True)
    b_exp_dissim = b_exp_dissim / nx.sum(b_exp_dissim, axis=1, keepdims=True)
    a_log_exp_dissim = nx.log(a_exp_dissim)
    b_log_exp_dissim = nx.log(b_exp_dissim)
    a_weighted_dissim_sum = nx.einsum("ij,ij->i", a_exp_dissim, a_log_exp_dissim)
    a_weighted_dissim_sum = nx.reshape(
        a_weighted_dissim_sum, (1, a_weighted_dissim_sum.shape[0])
    )
    divergence = a_weighted_dissim_sum.T - nx.dot(a_exp_dissim, b_log_exp_dissim.T)
    return nx.to_numpy(divergence)


def dissimilarity_metric(which, a_slice, b_slice, a_exp_dissim, b_exp_dissim, **kwargs):
    """
    Computes a dissimilarity matrix between two distribution using a specified
    metric.

    Parameters
    ----------
    which : str
        The dissimilarity metric to use. Options are:
        - "euc" or "euclidean" for Euclidean distance.
        - "gkl" for generalized KL divergence.
        - "kl" for KL divergence.
        - "selection_kl" for KL divergence with top 2000 high-UMI genes.
        - "pca" for PCA-based distance.
        - "glmpca" for GLM-PCA-based distance.

    a_slice : AnnData
        AnnData object containing data for the first slice.

    b_slice : AnnData
        AnnData object containing data for the second slice.

    a_exp_dissim : torch.Tensor
        A tensor representing the first probability distribution.

    b_exp_dissim : torch.Tensor
        A tensor representing the second probability distribution.

    Returns
    -------
    torch.Tensor
        A tensor representing pairwise dissimilarities between two distributions
        according to the specified metric.
    """
    match which:
        case "euc" | "euclidean":
            return torch.cdist(a_exp_dissim, b_exp_dissim)
        case "gkl":
            a_exp_dissim = a_exp_dissim + 0.01
            b_exp_dissim = b_exp_dissim + 0.01
            exp_dissim_matrix = generalized_kl_divergence(a_exp_dissim, b_exp_dissim)
            exp_dissim_matrix /= exp_dissim_matrix[exp_dissim_matrix > 0].max()
            exp_dissim_matrix *= 10
            return exp_dissim_matrix
        case "kl":
            a_exp_dissim = a_exp_dissim + 0.01
            b_exp_dissim = b_exp_dissim + 0.01
            return kl_divergence(a_exp_dissim, b_exp_dissim)
        case "selection_kl":
            return high_umi_gene_distance(a_exp_dissim, b_exp_dissim, 2000)
        case "pca":
            # TODO: Modify this function to work with Tensors
            return torch.Tensor(pca_distance(a_slice, b_slice, 2000, 20)).double()
        case "glmpca":
            # TODO: Modify this function to work with Tensors
            return torch.Tensor(
                glmpca_distance(a_exp_dissim, b_exp_dissim, **kwargs)
            ).double()
        case _:
            msg = f"Error: Invalid dissimilarity metric {which}"
            raise RuntimeError(msg)
