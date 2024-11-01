import logging
from pathlib import Path

import numpy as np
import scanpy as sc
from anndata import AnnData

from paste3.helper import match_spots_using_spatial_heuristic
from paste3.paste import center_align, pairwise_align
from paste3.visualization import stack_slices_center, stack_slices_pairwise

logger = logging.getLogger(__name__)


class Slice:
    def __init__(self, filepath: Path | None = None, adata: AnnData | None = None):
        if adata is None:
            self.adata = sc.read_h5ad(filepath)
        else:
            self.adata = adata

    def __str__(self):
        return f"Slice {self.adata}"


class AlignmentDataset:
    @staticmethod
    def from_csvs(gene_expression_csvs: list[Path], coordinate_csvs: list[Path]):
        pass

    def __init__(
        self,
        data_dir: Path | None = None,
        slices: list[Slice] | None = None,
        max_slices: int | None = None,
    ):
        if slices is not None:
            self.slices = slices[:max_slices]
        else:
            self.slices = [
                Slice(filepath)
                for filepath in sorted(Path(data_dir).glob("*.h5ad"))[:max_slices]
            ]

    def __str__(self):
        return f"Data with {len(self.slices)} slices"

    def __iter__(self):
        return iter(self.slices)

    def __len__(self):
        return len(self.slices)

    @property
    def slices_adata(self) -> list[AnnData]:
        return [slice_.adata for slice_ in self.slices]

    def align(
        self,
        center_align: bool = False,
        center_slice: Slice | None = None,
        pis: np.ndarray | None = None,
        overlap_fraction: float | None = None,
        max_iters: int = 1000,
    ):
        if center_align:
            if overlap_fraction is not None:
                logger.warning(
                    "Ignoring overlap_fraction argument (unsupported in center_align mode)"
                )
            return self.center_align(center_slice, pis)
        assert overlap_fraction is not None, "overlap_fraction must be specified"
        return self.pairwise_align(
            overlap_fraction=overlap_fraction, pis=pis, max_iters=max_iters
        )

    def find_pis(self, overlap_fraction: float, max_iters: int = 1000):
        pis = []
        for i in range(len(self) - 1):
            logger.info(f"Finding Pi for slices {i} and {i+1}")
            pis.append(
                pairwise_align(
                    self.slices[i].adata,
                    self.slices[i + 1].adata,
                    overlap_fraction=overlap_fraction,
                    numItermax=max_iters,
                    maxIter=max_iters,
                )
            )
        return pis

    def pairwise_align(
        self,
        overlap_fraction: float,
        pis: list[np.ndarray] | None = None,
        max_iters: int = 1000,
    ):
        if pis is None:
            pis = self.find_pis(overlap_fraction=overlap_fraction, max_iters=max_iters)
        new_slices = stack_slices_pairwise(self.slices_adata, pis)
        return AlignmentDataset(slices=[Slice(adata=s) for s in new_slices])

    def find_center_slice(
        self, reference_slice: Slice | None = None, pis: np.ndarray | None = None
    ) -> tuple[Slice, list[np.ndarray]]:
        if reference_slice is None:
            reference_slice = self.slices[0]
        center_slice, pis = center_align(
            reference_slice.adata, self.slices_adata, pi_inits=pis
        )
        return Slice(adata=center_slice), pis

    def find_pis_init(self) -> list[np.ndarray]:
        reference_slice = self.slices[0]
        return [
            match_spots_using_spatial_heuristic(reference_slice.adata.X, slice_.adata.X)
            for slice_ in self.slices
        ]

    def center_align(
        self,
        reference_slice: Slice | None = None,
        pis: list[np.ndarray] | None = None,
    ):
        if reference_slice is None:
            reference_slice, pis = self.find_center_slice(pis=pis)
        else:
            pis = self.find_pis_init()

        _, new_slices = stack_slices_center(
            center_slice=reference_slice.adata, slices=self.slices_adata, pis=pis
        )
        return AlignmentDataset(slices=[Slice(adata=s) for s in new_slices])


if __name__ == "__main__":
    dataset = AlignmentDataset("data/", max_slices=3)
    aligned_dataset = dataset.align(
        center_align=False, overlap_fraction=0.7, max_iters=2
    )
