import logging
import os.path
from functools import cached_property
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from sklearn.cluster import KMeans

from paste3.paste import center_align, pairwise_align
from paste3.visualization import stack_slices_center, stack_slices_pairwise

logger = logging.getLogger(__name__)


class Slice:
    """
    A single slice of spatial data.
    """

    def __init__(
        self,
        filepath: Path | None = None,
        adata: AnnData | None = None,
        name: str | None = None,
    ):
        """
        Initialize a slice of spatial data.

        Parameters
        ----------
        filepath : Path, optional
            Path to an h5ad file containing spatial data.
        adata : AnnData, optional
            Anndata object containing spatial data.
            If specified, takes precedence over `filepath`.
        name : str, optional
            Name of the slice.
            If not specified, the name is inferred from the file path or the adata object.
        """
        self.filepath = filepath
        self._adata = adata
        if name is None:
            if self.filepath is not None:
                self.name = Path(self.filepath).stem
            else:
                self.name = "Slice with adata: " + str(self.adata).split("\n")[0]
        else:
            self.name = name

        """
        Is the 'obs' array of `adata` indexed by strings of the form "XxY",
        where X/Y are Visium array locations?
        This format has been observed in legacy data.
        """
        self.has_coordinate_indices = all(
            "x" in index for index in self.adata.obs.index.values
        )

    def __str__(self):
        return self.name

    def _repr_mimebundle_(self, include=None, exclude=None):  # noqa: ARG002
        try:
            import squidpy
        except ImportError:
            return {}
        else:
            squidpy.pl.spatial_scatter(
                self.adata,
                frameon=False,
                shape=None,
                color="original_clusters",
                title=str(self),
            )

            # squidpy takes care of the rendering so we return an empty dict
            return {}

    @cached_property
    def adata(self):
        """
        Anndata object containing spatial data.
        """
        return self._adata or sc.read_h5ad(str(self.filepath))

    @cached_property
    def obs(self):
        """
        Anndata object containing observation metadata.
        The index of this dataframe is updated to be a MultiIndex
        with Visium array coordinates as indices if the observation
        metadata was originally indexed by strings of the form "XxY"
        """
        if self.has_coordinate_indices:
            logger.debug("Updating obs indices for easy access")
            obs = self.adata.obs.copy()
            obs.index = pd.MultiIndex.from_arrays(
                zip(*[map(int, i.split("x")) for i in obs.index], strict=False)
            )
            return obs
        return self.adata.obs

    def get_obs_values(self, which: str, coordinates: Any | None = None):
        """
        Get values from the observation metadata for specific coordinates.

        Parameters
        ----------
        which : str
            Column name to extract values from.
        coordinates : Any, optional
            List of Visium array coordinates to extract values for.
            These should be in the form of a list of tuples (X, Y),
            or whatever the format of the index of the observation metadata is.
            If not specified, values for all coordinates are returned.
        """
        assert which in self.obs.columns, f"Unknown column: {which}"
        if coordinates is None:
            coordinates = self.obs.index.values
        return self.obs.loc[coordinates][which].tolist()

    def set_obs_values(self, which: str, values: Any):
        """
        Set values in the observation metadata for specific coordinates.

        Parameters
        ----------
        which : str
            Column name to set values for.
        values : Any
            List of values to set for the specified column.
        """
        self.obs[which] = values

    def cluster(
        self,
        n_clusters: int,
        uns_key: str,
        random_state: int = 0,
        save_as: str | None = None,
    ) -> list[Any]:
        """
        Cluster observations based on a specified uns (unstructured) key
        in the underlying AnnData object of the Slice.
        The uns key is expected to contain a matrix of weights with shape
        (n_obs, n_features).

        Parameters
        ----------
        n_clusters : int
            Number of clusters to form.
        uns_key : str, optional
            Key in the uns array of the AnnData object to use for clustering.
        random_state : int, optional
            Random seed for reproducibility. Default 0.
        save_as : str, optional
            Name of the observation metadata column to save the cluster labels to.
            If not specified, the labels are not saved.

        Returns
        -------
        labels : np.ndarray
            Cluster labels for each observation.
        """
        a = self.adata.uns[uns_key].copy()
        a = (a.T / a.sum(axis=1)).T
        a = a + 1
        a = np.log(a)
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(a)
        labels = kmeans.labels_

        if save_as is not None:
            self.obs[save_as] = labels

        return labels


class AlignmentDataset:
    """
    A dataset of spatial slices that can be aligned together.
    """

    def __init__(
        self,
        file_paths: list[Path] | None = None,
        glob_pattern: str | None = None,
        slices: list[Slice] | None = None,
        max_slices: int | None = None,
        name: str | None = None,
    ):
        """
        Initialize a dataset of spatial slices.

        Parameters
        ----------
        file_paths : list of Path, optional
            List of paths to h5ad files containing spatial data.
        glob_pattern : str, optional
            Glob pattern to match files containing spatial data.
            If specified, takes precedence over `file_paths`.
        slices : list of Slice, optional
            List of Slice objects containing spatial data.
            If specified, takes precedence over `file_paths` and `glob_pattern`.
        max_slices : int, optional
            Maximum number of slices to load.
            If not specified, all slices are loaded.
        name : str, optional
            Name of the dataset.
            If not specified, the name is inferred from the common prefix of slice names.
        """
        if slices is not None:
            self.slices = slices[:max_slices]
        elif glob_pattern is not None:
            self.slices = [
                Slice(filepath)
                for filepath in sorted(glob(glob_pattern))[:max_slices]  # noqa: PTH207
            ]
        else:
            self.slices = [Slice(filepath) for filepath in file_paths[:max_slices]]

        if name is not None:
            self.name = name
        else:
            # Take common prefix of slice names, but remove the word "slice"
            # and any trailing underscores
            name = os.path.commonprefix([str(slice_) for slice_ in self])
            name = name.replace("slice", "").rstrip("_")
            self.name = name

    def __str__(self):
        return self.name

    def __iter__(self):
        return iter(self.slices)

    def __len__(self):
        return len(self.slices)

    def _repr_mimebundle_(self, include=None, exclude=None):
        for slice in self.slices:
            slice._repr_mimebundle_(include=include, exclude=exclude)

        # each slice takes care of the rendering so we return an empty dict
        return {}

    @property
    def slices_adata(self) -> list[AnnData]:
        """
        List of AnnData objects containing spatial data.
        """
        return [slice_.adata for slice_ in self.slices]

    def get_obs_values(self, which: str, coordinates: Any | None = None):
        """
        Get values from the observation metadata for specific coordinates.

        Parameters
        ----------
        which : str
            Column name to extract values from.
        coordinates : Any, optional
            List of Visium array coordinates to extract values for.
            These should be in the form of a list of tuples (X, Y),
            or whatever the format of the index of the observation metadata is.
            If not specified, values for all coordinates are returned.
        """
        return [slice_.get_obs_values(which, coordinates) for slice_ in self.slices]

    def align(
        self,
        center_align: bool = False,
        pis: np.ndarray | None = None,
        overlap_fraction: float | list[float] | None = None,
        max_iters: int = 1000,
    ):
        """
        Align slices in the dataset.

        Parameters
        ----------
        center_align : bool, optional
            Whether to center-align the slices. Default False.
            If False, pairwise-align the slices.
        pis : np.ndarray, optional
            Pairwise similarity between slices. Only used in pairwise-align
            mode. If not specified, the similarity is calculated.
        overlap_fraction : float or list of float, optional
            Fraction of overlap between slices. Only used, and required
            in pairwise-align mode.
        max_iters : int, optional
            Maximum number of iterations for alignment. Default 1000.
            Only used in pairwise-align mode.
        """
        if center_align:
            if overlap_fraction is not None:
                logger.warning(
                    "Ignoring overlap_fraction argument (unsupported in center_align mode)"
                )
            if pis is not None:
                logger.warning(
                    "Ignoring pis argument (unsupported in center_align mode)"
                )
            return self.center_align()
        assert overlap_fraction is not None, "overlap_fraction must be specified"
        return self.pairwise_align(
            overlap_fraction=overlap_fraction, pis=pis, max_iters=max_iters
        )

    def find_pis(
        self, overlap_fraction: float | list[float] | None = None, max_iters: int = 1000
    ):
        # If multiple overlap_fraction values are specified
        # ensure that they are |slices| - 1 in length
        try:
            iter(overlap_fraction)
        except TypeError:
            overlap_fraction = [overlap_fraction] * (len(self) - 1)
        assert (
            len(overlap_fraction) == len(self) - 1
        ), "Either specify a single overlap_fraction or one for each pair of slices"

        pis = []
        for i in range(len(self) - 1):
            logger.info(f"Finding Pi for slices {i} and {i+1}")
            pis.append(
                pairwise_align(
                    self.slices[i].adata,
                    self.slices[i + 1].adata,
                    overlap_fraction=overlap_fraction[i],
                    numItermax=max_iters,
                    maxIter=max_iters,
                )
            )
        return pis

    def pairwise_align(
        self,
        overlap_fraction: float | list[float] | None = None,
        pis: list[np.ndarray] | None = None,
        max_iters: int = 1000,
    ) -> tuple["AlignmentDataset", list[np.ndarray], list[np.ndarray]]:
        """
        Pairwise align slices in the dataset.

        Parameters
        ----------
        overlap_fraction : float or list of float or None, optional
            Fraction of overlap between each adjacent pair of slices.
            If a single value between 0 and 1 is specified, it is used for all pairs.
            If None, then a full alignment is performed.
        pis : list of np.ndarray, optional
            Pairwise similarity between slices.
            If not specified, the similarity is calculated.
        max_iters : int, optional
            Maximum number of iterations for alignment. Default 1000.

        Returns
        -------
        aligned_dataset : AlignmentDataset
            Aligned dataset.
        rotation_angles : list of np.ndarray
            Rotation angles for each slice.
        translations : list of np.ndarray
            Mutual translations for each pair of adjacent slices.
        """
        if pis is None:
            pis = self.find_pis(overlap_fraction=overlap_fraction, max_iters=max_iters)
        new_slices, rotation_angles, translations = stack_slices_pairwise(
            self.slices_adata, pis, return_params=True
        )
        aligned_dataset = AlignmentDataset(
            slices=[
                Slice(adata=new_slice, name=old_slice.name + "_pairwise_aligned")
                for old_slice, new_slice in zip(self.slices, new_slices, strict=False)
            ],
            name=self.name + "_pairwise_aligned",
        )

        return aligned_dataset, rotation_angles, translations

    def find_center_slice(
        self,
        initial_slice: Slice | None = None,
        slice_weights: list[float] | None = None,
        alpha: float = 0.1,
        n_components: int = 15,
        threshold: float = 0.001,
        max_iter: int = 10,
        exp_dissim_metric: str = "kl",
        norm: bool = False,
        random_seed: int | None = None,
        block: bool = True,
    ) -> tuple[Slice, list[np.ndarray]]:
        r"""
        Find the center slice of the dataset.

        Parameters
        ----------
        initial_slice : Slice, optional
            Initial slice to be used as a reference data for alignment.
            If not specified, the first slice in the dataset is used.
        slice_weights : list of float, optional
            Weights for each slice. If not specified, all slices are equally weighted.
        alpha : float, optional, default 0.1
            Regularization parameter balancing transcriptional dissimilarity and spatial distance among aligned spots.
            Setting \alpha = 0 uses only transcriptional information, while \alpha = 1 uses only spatial coordinates.
        n_components : int, optional, default 15
            Number of components to use in the NMF decomposition.
        threshold : float, optional, default 0.001
            Convergence threshold for the NMF algorithm.
        max_iter : int, optional, default 10
            Maximum number of iterations for the NMF algorithm.
        exp_dissim_metric : str, optional, default 'kl'
            The metric used to compute dissimilarity. Options include "euclidean" or "kl" for
            Kullback-Leibler divergence.
        norm : bool, default=False
            If True, normalize spatial distances.
        random_seed : Optional[int], default=None
            Random seed for reproducibility.
        block : bool, optional, default True
            Whether to block till the center slice is found.
            Set False to return a generator.

        Returns
        -------
        Tuple[Slice, List[np.ndarray]]
            A tuple containing:
            - center_slice : Slice
                Center slice of the dataset.
            - pis : List[np.ndarray]
                List of optimal transport distributions for each slice
                with the center slice.
        """
        logger.info("Finding center slice")
        if initial_slice is None:
            initial_slice = self.slices[0]

        gen = center_align(
            initial_slice=initial_slice.adata,
            slices=self.slices_adata,
            slice_weights=slice_weights,
            alpha=alpha,
            n_components=n_components,
            threshold=threshold,
            max_iter=max_iter,
            exp_dissim_metric=exp_dissim_metric,
            norm=norm,
            random_seed=random_seed,
            fast=True,
        )

        if block:
            try:
                while True:
                    next(gen)
            except StopIteration as e:
                center_slice, pis = e.value
                return Slice(adata=center_slice, name=self.name + "_center_slice"), pis
        else:
            return iter(gen)

    def center_align(
        self,
        initial_slice: Slice | None = None,
        center_slice: Slice | None = None,
        pis: list[np.ndarray] | None = None,
    ):
        logger.info("Center aligning")
        if center_slice is None:
            if pis is not None:
                logger.warning(
                    "Ignoring pis argument since center_slice is not provided"
                )
            center_slice, pis = self.find_center_slice(
                initial_slice=initial_slice, block=True
            )

        logger.info("Stacking slices around center slice")
        new_center, new_slices, rotation_angles, translations = stack_slices_center(
            center_slice=center_slice.adata,
            slices=self.slices_adata,
            pis=pis,
            output_params=True,
        )
        aligned_dataset = AlignmentDataset(
            slices=[
                Slice(adata=new_slice, name=old_slice.name + "_center_aligned")
                for old_slice, new_slice in zip(self.slices, new_slices, strict=False)
            ],
            name=self.name + "_center_aligned",
        )

        return aligned_dataset, rotation_angles, translations

    def all_points(self, translation: np.ndarray | None = None) -> np.ndarray:
        layers = []
        for i, slice in enumerate(self.slices):
            adata = slice.adata
            points = adata.obsm["spatial"]
            if translation is not None:
                points = points + translation
            layer_data = np.pad(
                points, pad_width=((0, 0), (1, 0)), mode="constant", constant_values=i
            )
            layers.append(layer_data)

        return np.concatenate(layers)
