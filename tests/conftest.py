from pathlib import Path

import numpy as np
import scanpy as sc
import glob
import pytest

base_dir = Path(__file__).parent.parent
data_folder = base_dir / "sample_data"


@pytest.fixture(scope="session")
def slices():
    # Returns ann data object relating each slices
    slice_files = glob.glob(f"{data_folder}/slice[0-9].csv")
    coord_files = glob.glob(f"{data_folder}/slice[0-9]_coor.csv")

    slices = []
    for slice, coord in zip(slice_files, coord_files):
        _slice = sc.read_csv(Path(slice))
        _slice.obsm["spatial"] = np.genfromtxt(coord, delimiter=",")
        _slice.obsm["weights"] = np.ones((_slice.shape[0],)) / _slice.shape[0]
        slices.append(_slice)

    return slices
