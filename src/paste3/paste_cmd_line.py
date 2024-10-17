import ot.backend
import numpy as np
import argparse
from pathlib import Path

import pandas as pd

from paste3.io import process_files
import logging
from paste3.paste import pairwise_align, center_align
from paste3.visualization import stack_slices_pairwise, stack_slices_center

logger = logging.getLogger(__name__)


def main(
    mode,
    gene_fpath,
    spatial_fpath=None,
    output_directory="",
    alpha=0.1,
    cost="kl",
    n_components=15,
    lmbda=None,
    initial_slice=1,
    threshold=0.001,
    coordinates=False,
    weight_fpath=None,
    overlap_fraction=None,
    start=None,
    seed=None,
    cost_matrix=None,
    max_iter=10,
    norm=False,
    numItermax=200,
    use_gpu=False,
    return_obj=False,
    optimizeTheta=True,
    eps=1e-4,
    is_histology=False,
    armijo=False,
):
    slices = process_files(gene_fpath, spatial_fpath, weight_fpath)
    n_slices = len(slices)

    if not (mode == "pairwise" or mode == "center"):
        raise (ValueError("Please select either pairwise or center alignment mode."))

    if alpha < 0 or alpha > 1:
        raise (ValueError("Alpha specified outside of 0-1 range."))

    if initial_slice < 1 or initial_slice > n_slices:
        raise (ValueError("Initial specified outside of 0 - n range"))

    if overlap_fraction:
        if overlap_fraction < 0 or overlap_fraction > 1:
            raise (ValueError("Overlap fraction specified outside of 0-1 range."))

    if lmbda is None:
        lmbda = n_slices * [1 / n_slices]
    elif len(lmbda) != n_slices:
        raise (ValueError("Length of lambda doesn't equal number of files"))
    else:
        if not all(i >= 0 for i in lmbda):
            raise (ValueError("lambda includes negative weights"))
        else:
            print("Normalizing lambda weights into probability vector.")
            lmbda = [float(i) / sum(lmbda) for i in lmbda]

    if cost_matrix:
        cost_matrix = np.genfromtxt(cost_matrix, delimiter=",", dtype="float64")

    if start is None:
        pis_init = [None] * (n_slices - 1) if mode == "pairwise" else None
    elif mode == "pairwise" and not (len(start) == n_slices - 1):
        raise ValueError(
            f"Number of slices {n_slices} is not equal to number of start pi files {len(start)}"
        )
    else:
        pis_init = [np.genfromtxt(pi, delimiter=",") for pi in start]

    # make output directory if it doesn't exist
    output_directory = Path(output_directory)
    Path.mkdir(output_directory, exist_ok=True)

    if mode == "pairwise":
        logger.info("Computing Pairwise Alignment ")
        pis = []
        for i in range(n_slices - 1):
            pi = pairwise_align(
                sliceA=slices[i],
                sliceB=slices[i + 1],
                s=overlap_fraction,
                M=cost_matrix,
                alpha=alpha,
                dissimilarity=cost,
                use_rep=None,
                G_init=pis_init[i],
                a_distribution=slices[i].obsm["weights"],
                b_distribution=slices[i + 1].obsm["weights"],
                norm=norm,
                numItermax=numItermax,
                backend=ot.backend.NumpyBackend(),
                use_gpu=use_gpu,
                return_obj=return_obj,
                maxIter=max_iter,
                optimizeTheta=optimizeTheta,
                eps=eps,
                is_histology=is_histology,
                armijo=armijo,
            )
            pis.append(pi)
            pd.DataFrame(
                pi, index=slices[i].obs.index, columns=slices[i + 1].obs.index
            ).to_csv(output_directory / f"slice_{i+1}_{i+2}_pairwise.csv")

        if coordinates:
            new_slices = stack_slices_pairwise(
                slices, pis, is_partial=overlap_fraction is not None
            )

    elif mode == "center":
        logger.info("Computing Center Alignment")
        initial_slice = slices[initial_slice - 1].copy()

        center_slice, pis = center_align(
            A=initial_slice,
            slices=slices,
            lmbda=lmbda,
            alpha=alpha,
            n_components=n_components,
            threshold=threshold,
            max_iter=max_iter,
            dissimilarity=cost,
            norm=norm,
            random_seed=seed,
            pis_init=pis_init,
            distributions=[slice.obsm["weights"] for slice in slices],
            backend=ot.backend.NumpyBackend(),
            use_gpu=use_gpu,
        )

        center_slice.write(output_directory / "center_slice.h5ad")
        for i in range(len(pis) - 1):
            pd.DataFrame(
                pis[i], index=center_slice.obs.index, columns=slices[i].obs.index
            ).to_csv(output_directory / f"slice_{i}_{i+1}_pairwise.csv")

        if coordinates:
            new_slices = stack_slices_center(center_slice, slices, pis)

    if coordinates:
        if mode == "center":
            center, new_slices = new_slices
            center.write(output_directory / "new_center.h5ad")

        for i, slice in enumerate(new_slices, start=1):
            slice.write(output_directory / f"new_slices_{i}.h5ad")


def add_args(parser):
    parser.add_argument(
        "mode",
        type=str,
        default="pairwise",
        help="Alignment type (Pairwise or Center)",
    )
    parser.add_argument(
        "--gene_fpath",
        type=str,
        nargs="+",
        help="Path to gene expression files (.csv/.h5ad)",
    )
    parser.add_argument(
        "--spatial_fpath",
        type=str,
        nargs="*",
        help="Path to spatial data files (.csv).",
    )
    parser.add_argument(
        "--weight_fpath",
        type=str,
        nargs="*",
        help="Path to the files containing weights of spots in each slice.",
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        default="",
        help="Path to the directory to save output files",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Alpha parameter for alignment (range from [0,1])",
    )
    parser.add_argument(
        "--cost",
        type=str,
        default="kl",
        help="Expression dissimilarity cost, either 'kl', 'euc', 'gkl', 'selection_kl', 'pca' or 'glmpca'",
    )
    parser.add_argument(
        "--cost_matrix",
        type=str,
        required=False,
        help="File path to expression dissimilarity cost matrix if available",
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=15,
        help="Number of components for NMF step in center alignment",
    )
    parser.add_argument(
        "--lmbda",
        type=float,
        nargs="+",
        help="Lambda param in center alignment (weight vector of length n)",
    )
    parser.add_argument(
        "--initial_slice",
        type=int,
        default=1,
        help="Specify which slice is the initial slice for center alignment (1 to n)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.001,
        help="Convergence threshold for center alignment.",
    )
    parser.add_argument(
        "--coordinates", action="store_true", help="Compute and save new coordinates"
    )
    parser.add_argument(
        "--overlap_fraction",
        type=float,
        default=None,
        help="Overlap fraction between two slices. (0-1)",
    )
    parser.add_argument(
        "--start",
        type=str,
        nargs="+",
        help="Path to files containing initial starting alignmnets. If not given the OT starts the search with uniform alignments. The format of the files is the same as the alignments files output by PASTE",
    )
    parser.add_argument("--norm", action="store_true", help="Normalize Data")
    parser.add_argument("--max_iter", type=int, help="Maximum number of iterations")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU")
    parser.add_argument(
        "--return_obj",
        action="store_true",
        help="Additionally returns objective function output of FGW-OT",
    )
    parser.add_argument(
        "--is_histology", action="store_true", help="If true, uses histological images "
    )
    parser.add_argument(
        "--armijo",
        action="store_true",
        help="If true, runs armijo line search function",
    )
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser).parse_args()
    main(
        mode=args.mode,
        gene_fpath=args.gene_fpath,
        spatial_fpath=args.spatial_fpath,
        output_directory=args.output_directory,
        alpha=args.alpha,
        cost=args.cost,
        n_components=args.n_components,
        lmbda=args.lmbda,
        initial_slice=args.initial_slice,
        threshold=args.threshold,
        coordinates=args.coordinates,
        weight_fpath=args.weight_fpath,
        overlap_fraction=args.overlap_fraction,
        start=args.start,
        seed=args.seed,
        cost_matrix=args.cost_matrix,
        norm=args.norm,
        numItermax=args.max_iter,
        use_gpu=args.use_gpu,
        return_obj=args.return_obj,
        is_histology=args.is_histology,
        armijo=args.armijo,
    )
