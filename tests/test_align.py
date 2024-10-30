import pandas as pd
import anndata as ad
from pandas.testing import assert_frame_equal
import scanpy as sc
from pathlib import Path
from paste3.io import get_shape, process_files
from paste3.align import align
import sys
import subprocess as sp
import paste3
import logging

logger = logging.getLogger(__name__)


test_dir = Path(__file__).parent
input_dir = test_dir / "data/input"
output_dir = test_dir / "data/output"


def test_cmd_line_center(tmp_path):
    logger.info(f"Running command in {tmp_path}")
    result = align(
        "center",
        [f"{input_dir}/slice{i}.csv" for i in range(1, 4)],
        [f"{input_dir}/slice{i}_coor.csv" for i in range(1, 4)],
        f"{tmp_path}",
        0.1,
        "kl",
        15,
        None,
        1,
        0.001,
        False,
        None,
        None,
        None,
        0,
        None,
    )

    assert result is None
    result = sc.read(tmp_path / "center_slice.h5ad")

    assert_frame_equal(
        pd.DataFrame(
            result.uns["paste_W"],
            index=result.obs.index,
            columns=[str(i) for i in range(15)],
        ),
        pd.read_csv(output_dir / "W_center", index_col=0),
        check_names=False,
        check_index_type=False,
        rtol=1e-05,
        atol=1e-08,
    )
    assert_frame_equal(
        pd.DataFrame(result.uns["paste_H"], columns=result.var.index),
        pd.read_csv(output_dir / "H_center", index_col=0),
        rtol=1e-05,
        atol=1e-08,
    )

    for i, pi in enumerate(range(2)):
        assert_frame_equal(
            pd.read_csv(tmp_path / f"slice_{i}_{i+1}_pairwise.csv"),
            pd.read_csv(
                output_dir / f"slice_center_slice{i + 1}_pairwise.csv",
            ),
        )


def test_cmd_line_pairwise_csv(tmp_path):
    logger.info(f"Running command in {tmp_path}")
    result = align(
        "pairwise",
        [
            f"{input_dir}/slice1.csv",
            f"{input_dir}/slice2.csv",
            f"{input_dir}/slice3.csv",
        ],
        [
            f"{input_dir}/slice1_coor.csv",
            f"{input_dir}/slice2_coor.csv",
            f"{input_dir}/slice3_coor.csv",
        ],
        f"{tmp_path}",
        0.1,
        "kl",
        15,
        None,
        1,
        0.001,
        False,
        None,
        None,
        None,
        0,
        None,
        max_iter=1000,
    )

    assert result is None
    assert_frame_equal(
        pd.read_csv(tmp_path / "slice_1_2_pairwise.csv"),
        pd.read_csv(
            output_dir / "slices_1_2_pairwise.csv",
        ),
    )


def test_process_files_csv():
    """Ensure process files works with csv inputs."""
    gene_fpath = []
    spatial_fpath = []
    for i in range(1, 5):
        gene_fpath.append(Path(f"{input_dir}/slice{i}.csv"))
        spatial_fpath.append(Path(f"{input_dir}/slice{i}_coor.csv"))

    ad_objs = process_files(
        gene_fpath,
        spatial_fpath,
    )
    for obj in ad_objs:
        assert isinstance(obj, ad.AnnData)


def test_process_files_ann_data():
    """Ensure process files works with Ann Data inputs."""
    gene_fpath = []
    for i in range(3, 7):
        gene_fpath.append(Path(f"{input_dir}/15167{i}.h5ad"))

    ad_objs = process_files(gene_fpath, s_fpath=None)
    for obj in ad_objs:
        assert isinstance(obj, ad.AnnData)


def test_get_shape():
    s_fpath = Path(f"{input_dir}/slice1.csv")
    c_fpath = Path(f"{input_dir}/slice1_coor.csv")

    assert get_shape(s_fpath) == (254, 7999)
    assert get_shape(c_fpath) == (254, 2)


def test_version():
    result = sp.run(
        [sys.executable, "-m", "paste3", "--version"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert result.stdout.strip() == paste3.__version__
