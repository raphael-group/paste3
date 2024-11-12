from pathlib import Path

import numpy as np
import ot.backend
import pytest
import scanpy as sc
import torch

from paste3.napari.data.ondemand import get_file

test_dir = Path(__file__).parent
input_dir = test_dir / "data/input"


@pytest.fixture(scope="session")
def slices():
    slices = []
    for i in range(1, 5):
        # File path of slices and respective coordinates
        s_fpath = Path(f"{input_dir}/slice{i}.csv")
        c_fpath = Path(f"{input_dir}/slice{i}_coor.csv")

        # Create ann data object of each slice and add other properties
        _slice = sc.read_csv(s_fpath, dtype="float64")
        _slice.obsm["spatial"] = np.genfromtxt(c_fpath, delimiter=",")
        _slice.obsm["weights"] = np.ones((_slice.shape[0],)) / _slice.shape[0]
        slices.append(_slice)

    return slices


@pytest.fixture(scope="session")
def spot_distance_matrix(slices):
    nx = ot.backend.TorchBackend()

    spot_distances = []
    for slice in slices:
        spot_distances.append(
            ot.dist(
                nx.from_numpy(slice.obsm["spatial"]),
                nx.from_numpy(slice.obsm["spatial"]),
                metric="euclidean",
            )
        )

    return spot_distances


@pytest.fixture(scope="session")
def intersecting_slices(slices):
    # Make a copy of the list
    slices = list(slices)

    common_genes = slices[0].var.index
    for slice in slices[1:]:
        common_genes = common_genes.intersection(slice.var.index)

    for i in range(len(slices)):
        slices[i] = slices[i][:, common_genes]

    return slices


@pytest.fixture(scope="session")
def slices2():
    slices = []
    for i in range(3, 7):
        fpath = Path(f"{input_dir}/15167{i}.h5ad")

        _slice = sc.read_h5ad(fpath)
        slices.append(_slice)

    return slices


@pytest.fixture(scope="session")
def sample_data_files():
    files = [
        "paste3_sample_patient_2_slice_0.h5ad",
        "paste3_sample_patient_2_slice_1.h5ad",
        "paste3_sample_patient_2_slice_2.h5ad",
    ]
    return [Path(get_file(f)) for f in files]


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
