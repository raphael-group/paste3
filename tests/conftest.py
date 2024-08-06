from pathlib import Path
import numpy as np
import scanpy as sc
import pytest

base_dir = Path(__file__).parent.parent
data_dir = base_dir / "sample_data"


@pytest.fixture(scope="session")
def slices():
    slices = []
    for i in range(1, 5):
        # File path of slices and respective coordinates
        s_fpath = Path(f"{data_dir}/slice{i}.csv")
        c_fpath = Path(f"{data_dir}/slice{i}_coor.csv")

        # Create ann data object of each slice and add other properties
        _slice = sc.read_csv(s_fpath)
        _slice.obsm["spatial"] = np.genfromtxt(c_fpath, delimiter=",")
        _slice.obsm["weights"] = np.ones((_slice.shape[0],)) / _slice.shape[0]
        slices.append(_slice)

    return slices
