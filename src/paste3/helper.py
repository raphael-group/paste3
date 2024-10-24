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


def kl_divergence(X, Y):
    """
    Returns pairwise KL divergence (over all pairs of samples) of two matrices X and Y.

    param: X - np array with dim (n_samples by n_features)
    param: Y - np array with dim (m_samples by n_features)

    return: D - np array with dim (n_samples by m_samples). Pairwise KL divergence matrix.
    """
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."

    X = X / X.sum(axis=1, keepdims=True)
    Y = Y / Y.sum(axis=1, keepdims=True)
    log_X = X.log()
    log_Y = Y.log()
    X_log_X = torch.sum(X * log_X, axis=1)[torch.newaxis, :]
    D = X_log_X.T - torch.matmul(X, log_Y.T)
    return D


def generalized_kl_divergence(X, Y):
    """
    Returns pairwise generalized KL divergence (over all pairs of samples) of two matrices X and Y.

    param: X - np array with dim (n_samples by n_features)
    param: Y - np array with dim (m_samples by n_features)

    return: D - np array with dim (n_samples by m_samples). Pairwise generalized KL divergence matrix.
    """
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."

    log_X = X.log()
    log_Y = Y.log()
    X_log_X = torch.sum(X * log_X, axis=1)[torch.newaxis, :]
    D = X_log_X.T - torch.matmul(X, log_Y.T)
    sum_X = torch.sum(X, axis=1)
    sum_Y = torch.sum(Y, axis=1)
    D = (D.T - sum_X).T + sum_Y.T
    return D


def glmpca_distance(
    X,
    Y,
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
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."

    joint_matrix = torch.vstack((X, Y))
    if filter:
        gene_umi_counts = torch.sum(joint_matrix, axis=0).cpu().numpy()
        top_indices = np.sort((-gene_umi_counts).argsort(kind="stable")[:2000])
        joint_matrix = joint_matrix[:, top_indices]

    logging.info("Starting GLM-PCA...")
    res = glmpca(
        joint_matrix.T.cpu().numpy(),  # TODO: Use Tensors
        latent_dim,
        penalty=1,
        ctl={"maxIter": maxIter, "eps": eps, "optimizeTheta": optimizeTheta},
    )
    reduced_joint_matrix = res["factors"]
    logging.info("GLM-PCA finished.")

    X = reduced_joint_matrix[: X.shape[0], :]
    Y = reduced_joint_matrix[X.shape[0] :, :]
    return distance.cdist(X, Y)


def pca_distance(sliceA, sliceB, n, latent_dim):
    joint_adata = ad.concat([sliceA, sliceB])
    sc.pp.normalize_total(joint_adata, inplace=True)
    sc.pp.log1p(joint_adata)
    sc.pp.highly_variable_genes(
        joint_adata, flavor="seurat", n_top_genes=n, inplace=True, subset=True
    )
    sc.pp.pca(joint_adata, latent_dim)
    joint_datamatrix = joint_adata.obsm["X_pca"]
    X = joint_datamatrix[: sliceA.shape[0], :]
    Y = joint_datamatrix[sliceA.shape[0] :, :]
    return distance.cdist(X, Y)


def high_umi_gene_distance(X, Y, n):
    """
    n: number of highest umi count genes to keep
    """
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."

    joint_matrix = torch.vstack((X, Y))
    gene_umi_counts = torch.sum(joint_matrix, axis=0).cpu().numpy()
    top_indices = np.sort((-gene_umi_counts).argsort(kind="stable")[:n])
    X = X[:, top_indices]
    Y = Y[:, top_indices]
    X += torch.tile(0.01 * (torch.sum(X, axis=1) / X.shape[1]), (X.shape[1], 1)).T
    Y += torch.tile(0.01 * (torch.sum(Y, axis=1) / Y.shape[1]), (Y.shape[1], 1)).T
    return kl_divergence(X, Y)


def intersect(lst1, lst2):
    """
    param: lst1 - list
    param: lst2 - list

    return: list of common elements
    """

    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3


def norm_and_center_coordinates(X):
    """
    param: X - numpy array

    return:
    """
    return (X - X.mean(axis=0)) / min(scipy.spatial.distance.pdist(X))


## Covert a sparse matrix into a dense matrix
def to_dense_array(X):
    np_array = np.array(X.todense()) if isinstance(X, scipy.sparse.csr.spmatrix) else X
    return torch.Tensor(np_array).double()


def extract_data_matrix(adata, rep=None):
    return adata.X if rep is None else adata.obsm[rep]


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


def match_spots_using_spatial_heuristic(X, Y, use_ot: bool = True) -> np.ndarray:
    """
    Calculates and returns a mapping of spots using a spatial heuristic.

    Args:
        X (array-like, optional): Coordinates for spots X.
        Y (array-like, optional): Coordinates for spots Y.
        use_ot: If ``True``, use optimal transport ``ot.emd()`` to calculate mapping. Otherwise, use Scipy's ``min_weight_full_bipartite_matching()`` algorithm.

    Returns:
        Mapping of spots using a spatial heuristic.
    """
    # X, Y = X.todense(), Y.todense()
    n1, n2 = len(X), len(Y)
    X, Y = norm_and_center_coordinates(X), norm_and_center_coordinates(Y)
    dist = scipy.spatial.distance_matrix(X, Y)
    if use_ot:
        pi = ot.emd(np.ones(n1) / n1, np.ones(n2) / n2, dist)
    else:
        row_ind, col_ind = scipy.sparse.csgraph.min_weight_full_bipartite_matching(
            scipy.sparse.csr_matrix(dist)
        )
        pi = np.zeros((n1, n2))
        pi[row_ind, col_ind] = 1 / max(n1, n2)
        if n1 < n2:
            pi[:, [(j not in col_ind) for j in range(n2)]] = 1 / (n1 * n2)
        elif n2 < n1:
            pi[[(i not in row_ind) for i in range(n1)], :] = 1 / (n1 * n2)
    return pi


def kl_divergence_backend(X, Y):
    """
    Returns pairwise KL divergence (over all pairs of samples) of two matrices X and Y.

    Takes advantage of POT backend to speed up computation.

    Args:
        X: np array with dim (n_samples by n_features)
        Y: np array with dim (m_samples by n_features)

    Returns:
        D: np array with dim (n_samples by m_samples). Pairwise KL divergence matrix.
    """
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."

    nx = ot.backend.get_backend(X, Y)

    X = X / nx.sum(X, axis=1, keepdims=True)
    Y = Y / nx.sum(Y, axis=1, keepdims=True)
    log_X = nx.log(X)
    log_Y = nx.log(Y)
    X_log_X = nx.einsum("ij,ij->i", X, log_X)
    X_log_X = nx.reshape(X_log_X, (1, X_log_X.shape[0]))
    D = X_log_X.T - nx.dot(X, log_Y.T)
    return nx.to_numpy(D)


def dissimilarity_metric(which, sliceA, sliceB, A, B, **kwargs):
    match which:
        case "euc" | "euclidean":
            return torch.cdist(A, B)
        case "gkl":
            s_A = A + 0.01
            s_B = B + 0.01
            M = generalized_kl_divergence(s_A, s_B)
            M /= M[M > 0].max()
            M *= 10
            return M
        case "kl":
            s_A = A + 0.01
            s_B = B + 0.01
            M = kl_divergence(s_A, s_B)
            return M
        case "selection_kl":
            return high_umi_gene_distance(A, B, 2000)
        case "pca":
            # TODO: Modify this function to work with Tensors
            return torch.Tensor(pca_distance(sliceA, sliceB, 2000, 20)).double()
        case "glmpca":
            # TODO: Modify this function to work with Tensors
            return torch.Tensor(glmpca_distance(A, B, **kwargs)).double()
        case _:
            raise RuntimeError(f"Error: Invalid dissimilarity metric {which}")
