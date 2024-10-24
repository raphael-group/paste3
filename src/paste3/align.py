import ot.backend
import numpy as np
from pathlib import Path

import pandas as pd

from paste3.io import process_files
import logging
from paste3.paste import pairwise_align, center_align
from paste3.visualization import stack_slices_pairwise, stack_slices_center

logger = logging.getLogger(__name__)


def align(
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
    use_gpu=True,
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
        pi_inits = [None] * (n_slices - 1) if mode == "pairwise" else None
    elif mode == "pairwise" and not (len(start) == n_slices - 1):
        raise ValueError(
            f"Number of slices {n_slices} is not equal to number of start pi files {len(start)}"
        )
    else:
        pi_inits = [np.genfromtxt(pi, delimiter=",") for pi in start]

    # make output directory if it doesn't exist
    output_directory = Path(output_directory)
    Path.mkdir(output_directory, exist_ok=True)

    if mode == "pairwise":
        logger.info("Computing Pairwise Alignment ")
        pis = []
        for i in range(n_slices - 1):
            pi = pairwise_align(
                a_slice=slices[i],
                b_slice=slices[i + 1],
                overlap_fraction=overlap_fraction,
                exp_dissim_matrix=cost_matrix,
                alpha=alpha,
                exp_dissim_metric=cost,
                pi_init=pi_inits[i],
                a_spots_weight=slices[i].obsm["weights"],
                b_spots_weight=slices[i + 1].obsm["weights"],
                norm=norm,
                numItermax=numItermax,
                backend=ot.backend.TorchBackend(),
                use_gpu=use_gpu,
                return_obj=return_obj,
                maxIter=max_iter,
                optimizeTheta=optimizeTheta,
                eps=eps,
                do_histology=is_histology,
                armijo=armijo,
            )
            pis.append(pi)
            pd.DataFrame(
                pi.cpu().numpy(),
                index=slices[i].obs.index,
                columns=slices[i + 1].obs.index,
            ).to_csv(output_directory / f"slice_{i+1}_{i+2}_pairwise.csv")

        if coordinates:
            new_slices = stack_slices_pairwise(
                slices, pis, is_partial=overlap_fraction is not None
            )

    elif mode == "center":
        logger.info("Computing Center Alignment")
        initial_slice = slices[initial_slice - 1].copy()

        center_slice, pis = center_align(
            initial_slice=initial_slice,
            slices=slices,
            slice_weights=lmbda,
            alpha=alpha,
            n_components=n_components,
            threshold=threshold,
            max_iter=max_iter,
            exp_dissim_metric=cost,
            norm=norm,
            random_seed=seed,
            pi_inits=pi_inits,
            spots_weights=[slice.obsm["weights"] for slice in slices],
            backend=ot.backend.TorchBackend(),
            use_gpu=use_gpu,
        )

        center_slice.write(output_directory / "center_slice.h5ad")
        for i in range(len(pis) - 1):
            pd.DataFrame(
                pis[i].cpu().numpy(),
                index=center_slice.obs.index,
                columns=slices[i].obs.index,
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
        "mode", type=str, help="Alignment type: 'pairwise' or 'center'."
    )
    parser.add_argument(
        "--g_fpath", type=str, nargs="+", help="Paths to gene exp files (.csv/ .h5ad)."
    )
    parser.add_argument(
        "--s_fpath", type=str, nargs="*", help="Paths to spatial data files (.csv)."
    )
    parser.add_argument(
        "--w_fpath", type=str, nargs="*", help="Paths to spot weight files (.csv)."
    )
    parser.add_argument(
        "--output_dir", default="./output", help="Directory to save output files."
    )
    parser.add_argument(
        "--alpha", type=float, default=0.1, help="Alpha param for alignment (0 to 1)."
    )
    parser.add_argument(
        "--cost",
        choices=["kl", "euc", "gkl", "selection_kl", "pca", "glmpca"],
        default="kl",
        help="Expression dissimilarity cost",
    )

    parser.add_argument(
        "--cost_mat", type=str, help="Paths to exp dissimilarity cost matrix."
    )
    parser.add_argument(
        "--n_comp", type=int, default=15, help="Components for NMF in center alignment."
    )
    parser.add_argument(
        "--lmbda", type=float, nargs="+", help="Weight vector for each slice."
    )
    parser.add_argument(
        "--init_slice", type=int, default=1, help="First slice for alignment (1 to n)."
    )
    parser.add_argument(
        "--thresh",
        type=float,
        default=1e-3,
        help="Convergence threshold for alignment.",
    )

    parser.add_argument(
        "--coor", action="store_true", help="Compute and save new coordinates."
    )
    parser.add_argument(
        "--ovlp_frac", type=float, default=None, help="Overlap fraction (0-1)."
    )
    parser.add_argument(
        "--start", type=str, nargs="+", help="Paths to initial alignment files."
    )
    parser.add_argument(
        "--norm", action="store_true", help="Normalize expression data if True."
    )
    parser.add_argument("--max_iter", type=int, help="Maximum number of iterations.")
    parser.add_argument(
        "--gpu", action="store_true", help="Use GPU for processing if True."
    )
    parser.add_argument("--r_info", action="store_true", help="Returns log if True.")
    parser.add_argument(
        "--hist", action="store_true", help="Use histological images if True."
    )
    parser.add_argument(
        "--armijo", action="store_true", help="Run Armijo line search if True."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility."
    )
    return parser


def main(args):
    align(
        mode=args.mode,
        gene_fpath=args.g_fpath,
        spatial_fpath=args.s_fpath,
        output_directory=args.output_dir,
        alpha=args.alpha,
        cost=args.cost,
        n_components=args.n_comp,
        lmbda=args.lmbda,
        initial_slice=args.init_slice,
        threshold=args.thresh,
        coordinates=args.coor,
        weight_fpath=args.w_fpath,
        overlap_fraction=args.ovlp_frac,
        start=args.start,
        seed=args.seed,
        cost_matrix=args.cost_mat,
        norm=args.norm,
        numItermax=args.max_iter,
        use_gpu=args.gpu,
        return_obj=args.r_info,
        is_histology=args.hist,
        armijo=args.armijo,
    )
