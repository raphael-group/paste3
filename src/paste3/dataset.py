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
    def __init__(self, filepath: Path | None = None, adata: AnnData | None = None):
        self.filepath = filepath
        self._adata = adata

        # Is the 'obs' array of `adata` indexed by strings of the form "XxY",
        # where X/Y are Visium array locations?
        # This format has been observed in legacy data.
        self.has_coordinate_indices = all(
            "x" in index for index in self.adata.obs.index.values
        )

        self.has_spatial_data = "spatial" in self.adata.obsm

    def __str__(self):
        if self.filepath is not None:
            return Path(self.filepath).stem
        return "Slice with adata: " + str(self.adata).split("\n")[0]

    @cached_property
    def adata(self):
        return self._adata or sc.read_h5ad(str(self.filepath))

    @cached_property
    def obs(self):
        if self.has_coordinate_indices:
            logger.debug("Updating obs indices for easy access")
            obs = self.adata.obs.copy()
            obs.index = pd.MultiIndex.from_arrays(
                zip(*[map(int, i.split("x")) for i in obs.index], strict=False)
            )
            return obs
        return self.adata.obs

    def get_obs_values(self, which, coordinates=None):
        assert which in self.obs.columns, f"Unknown column: {which}"
        if coordinates is None:
            coordinates = self.obs.index.values
        return self.obs.loc[coordinates][which].tolist()

    def set_obs_values(self, which, values):
        self.obs[which] = values

    def cluster(
        self,
        n_clusters: int,
        uns_key: str = "paste_W",
        random_state: int = 5,
        save_as: str | None = None,
    ):
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
    def __init__(
        self,
        file_paths: list[Path] | None = None,
        glob_pattern: str | None = None,
        slices: list[Slice] | None = None,
        max_slices: int | None = None,
        name: str | None = None,
    ):
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

    @property
    def slices_adata(self) -> list[AnnData]:
        return [slice_.adata for slice_ in self.slices]

    def get_obs_values(self, which, coordinates=None):
        return [slice_.get_obs_values(which, coordinates) for slice_ in self.slices]

    def align(
        self,
        center_align: bool = False,
        pis: np.ndarray | None = None,
        overlap_fraction: float | list[float] | None = None,
        max_iters: int = 1000,
    ):
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

    def find_pis(self, overlap_fraction: float | list[float], max_iters: int = 1000):
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
        overlap_fraction: float | list[float],
        pis: list[np.ndarray] | None = None,
        max_iters: int = 1000,
    ):
        if pis is None:
            pis = self.find_pis(overlap_fraction=overlap_fraction, max_iters=max_iters)
        new_slices, rotation_angles, translations = stack_slices_pairwise(
            self.slices_adata, pis, return_params=True
        )
        aligned_dataset = AlignmentDataset(
            slices=[Slice(adata=s) for s in new_slices],
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
        pbar: Any = None,
    ) -> tuple[Slice, list[np.ndarray]]:
        logger.info("Finding center slice")
        if initial_slice is None:
            initial_slice = self.slices[0]

        center_slice, pis = center_align(
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
            pbar=pbar,
            fast=True,
        )
        return Slice(adata=center_slice), pis

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
            center_slice, pis = self.find_center_slice(initial_slice=initial_slice)

        logger.info("Stacking slices around center slice")
        new_center, new_slices, rotation_angles, translations = stack_slices_center(
            center_slice=center_slice.adata,
            slices=self.slices_adata,
            pis=pis,
            output_params=True,
        )
        aligned_dataset = AlignmentDataset(
            slices=[Slice(adata=s) for s in new_slices],
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
