from pathlib import Path

import anndata
import numpy as np

from paste3.dataset import Slice

test_dir = Path(__file__).parent
input_dir = test_dir / "data/input"


def test_slice_adata(slices):
    # The `Slice` class is a convenient way to work with a single slice.
    # The `slices` fixture returns a list of AnnData objects
    slice = Slice(adata=slices[0])
    # The AnnData object for a slice is available as the `adata` attribute
    assert isinstance(slice.adata, anndata.AnnData)
    assert np.all(slice.adata.var_names == slices[0].var_names)
    assert np.all(slice.adata.obs_names == slices[0].obs_names)


def test_slice_adata_name_str(slices):
    # We can give names to slices
    slice = Slice(adata=slices[0], name="my_slice")
    assert str(slice) == "my_slice"


def test_slice_adata_noname_str(slices):
    # If we don't give names to slices, we have a sensible default
    slice = Slice(adata=slices[0])
    assert (
        str(slice)
        == "Slice with adata: AnnData object with n_obs × n_vars = 254 × 7998"  # noqa: RUF001
    )


def test_slice_filepath_str(sample_data_files):
    # If we don't give names to slices, we have a sensible default
    slice = Slice(filepath=sample_data_files[0])
    assert str(slice) == "paste3_sample_patient_2_slice_0"


def test_slice_filepath_get_obs(sample_data_files):
    # Retrieve observation values
    slice = Slice(filepath=sample_data_files[0])
    cluster_indices = slice.get_obs_values("original_clusters")
    assert isinstance(cluster_indices, list)  # list of strings

    cluster_indices = np.array(cluster_indices)
    assert len(cluster_indices) == 666  # 666 n_obs
    assert len(np.unique(cluster_indices)) == 12  # 12 unique clusters


def test_slice_filepath_set_obs(sample_data_files):
    # Set and retrieve observation values
    slice = Slice(filepath=sample_data_files[0])
    obs_values = np.random.random(666)
    slice.set_obs_values("foo", obs_values)
    assert np.allclose(slice.get_obs_values("foo"), obs_values)
