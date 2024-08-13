from pathlib import Path
import numpy as np
import ot.backend
import pandas as pd
from pandas.testing import assert_frame_equal
from paste.helper import (
    intersect,
    kl_divergence_backend,
    to_dense_array,
    extract_data_matrix,
)

test_dir = Path(__file__).parent
input_dir = test_dir / "data/input"
output_dir = test_dir / "data/output"


def test_intersect(slices):
    common_genes = intersect(slices[1].var.index, slices[2].var.index)
    assert np.all(
        np.equal(
            common_genes, np.genfromtxt(output_dir / "common_genes.csv", dtype=str)
        )
    )


def test_kl_divergence_backend(slices):
    nx = ot.backend.NumpyBackend()

    common_genes = intersect(slices[1].var.index, slices[2].var.index)
    sliceA = slices[1][:, common_genes]
    sliceB = slices[2][:, common_genes]

    slice1_X = nx.from_numpy(to_dense_array(extract_data_matrix(sliceA, None)))
    slice2_X = nx.from_numpy(to_dense_array(extract_data_matrix(sliceB, None)))

    kl_divergence_matrix = kl_divergence_backend(slice1_X + 0.01, slice2_X + 0.01)
    assert_frame_equal(
        pd.DataFrame(kl_divergence_matrix, columns=[str(i) for i in range(264)]),
        pd.read_csv(output_dir / "kl_divergence_matrix.csv"),
        check_names=False,
        check_dtype=False,
        rtol=1e-04,
    )
