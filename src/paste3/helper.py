from paste3.glmpca import glmpca
import anndata as ad
import scanpy as sc
from scipy.spatial import distance
from typing import List
from anndata import AnnData
import numpy as np
import torch
import scipy
import ot
import logging

logger = logging.getLogger(__name__)


def kl_divergence(a_exp_dissim, b_exp_dissim):
    """
    Returns pairwise KL divergence (over all pairs of samples) of two matrices X and Y.

    param: X - np array with dim (n_samples by n_features)
    param: Y - np array with dim (m_samples by n_features)

    return: D - np array with dim (n_samples by m_samples). Pairwise KL divergence matrix.
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
    divergence = a_weighted_dissim_sum.T - torch.matmul(
        a_exp_dissim, b_log_exp_dissim.T
    )
    return divergence


def generalized_kl_divergence(a_exp_dissim, b_exp_dissim):
    """
    Returns pairwise generalized KL divergence (over all pairs of samples) of two matrices X and Y.

    param: X - np array with dim (n_samples by n_features)
    param: Y - np array with dim (m_samples by n_features)

    return: D - np array with dim (n_samples by m_samples). Pairwise generalized KL divergence matrix.
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
    divergence = (divergence.T - sum_a_exp_dissim).T + sum_b_exp_dissim.T
    return divergence


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
    param: X - np array with dim (n_samples by n_features)
    param: Y - np array with dim (m_samples by n_features)
    param: latent_dim - number of latent dimensions in glm-pca
    param: filter - whether to first select genes with highest UMI counts
    param maxIter - maximum number of iterations for glmpca
    param eps - convergence threshold for glmpca
    param optimizeTheta - whether to optimize overdispersion in glmpca
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
    n: number of highest umi count genes to keep
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


def intersect(a_list, b_list):
    """
    param: lst1 - list
    param: lst2 - list

    return: list of common elements
    """
    return [val for val in a_list if val in set(b_list)]


def norm_and_center_coordinates(spatial_dist):
    """
    param: X - numpy array

    return:
    """
    return (spatial_dist - spatial_dist.mean(axis=0)) / min(
        scipy.spatial.distance.pdist(spatial_dist)
    )


## Covert a sparse matrix into a dense matrix
def to_dense_array(X):
    np_array = np.array(X.todense()) if isinstance(X, scipy.sparse.csr.spmatrix) else X
    return torch.Tensor(np_array).double()


def filter_for_common_genes(slices: List[AnnData]) -> None:
    """
    Filters for the intersection of genes between all slices.

    Args:
        slices: List of slices.
    """
    assert len(slices) > 0, "Cannot have empty list."

    common_genes = slices[0].var.index
    for s in slices:
        common_genes = intersect(common_genes, s.var.index)
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
    Calculates and returns a mapping of spots using a spatial heuristic.

    Args:
        a_spatial_dist (array-like, optional): Coordinates for spots X.
        b_spatial_dist (array-like, optional): Coordinates for spots Y.
        use_ot: If ``True``, use optimal transport ``ot.emd()`` to calculate mapping. Otherwise, use Scipy's ``min_weight_full_bipartite_matching()`` algorithm.

    Returns:
        Mapping of spots using a spatial heuristic.
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
    Returns pairwise KL divergence (over all pairs of samples) of two matrices X and Y.

    Takes advantage of POT backend to speed up computation.

    Args:
        a_exp_dissim: np array with dim (n_samples by n_features)
        b_exp_dissim: np array with dim (m_samples by n_features)

    Returns:
        D: np array with dim (n_samples by m_samples). Pairwise KL divergence matrix.
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
            exp_dissim_matrix = kl_divergence(a_exp_dissim, b_exp_dissim)
            return exp_dissim_matrix
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
            raise RuntimeError(f"Error: Invalid dissimilarity metric {which}")
