import pandas as pd
from pandas.testing import assert_frame_equal
from pathlib import Path
import subprocess

test_dir = Path(__file__).parent
input_dir = test_dir / "data/input"
output_dir = test_dir / "data/output"


def test_cmd_line_center(tmp_path):
    print(f"Running command in {tmp_path}")
    result = subprocess.run(
        [
            "python",
            f"{test_dir.parent}/src/paste/paste-cmd-line.py",
            "-m",
            "center",
            "--seed",
            "0",
            "-f",
            f"{input_dir}/slice1.csv",
            f"{input_dir}/slice1_coor.csv",
            f"{input_dir}/slice2.csv",
            f"{input_dir}/slice2_coor.csv",
            f"{input_dir}/slice3.csv",
            f"{input_dir}/slice3_coor.csv",
            "-d",
            f"{tmp_path}",
        ]
    )
    assert result.returncode == 0
    assert_frame_equal(
        pd.read_csv(tmp_path / "paste_output/W_center"),
        pd.read_csv(output_dir / "W_center"),
        check_names=False,
        rtol=1e-05,
        atol=1e-08,
    )
    assert_frame_equal(
        pd.read_csv(tmp_path / "paste_output/H_center"),
        pd.read_csv(output_dir / "H_center"),
        rtol=1e-05,
        atol=1e-08,
    )

    for i, pi in enumerate(range(3)):
        assert_frame_equal(
            pd.read_csv(
                tmp_path / f"paste_output/slice_center_slice{i + 1}_pairwise.csv"
            ),
            pd.read_csv(output_dir / f"slice_center_slice{i + 1}_pairwise.csv"),
        )


def test_cmd_line_pairwise(tmp_path):
    print(f"Running command in {tmp_path}")
    result = subprocess.run(
        [
            "python",
            f"{test_dir.parent}/src/paste/paste-cmd-line.py",
            "-m",
            "pairwise",
            "-f",
            f"{input_dir}/slice1.csv",
            f"{input_dir}/slice1_coor.csv",
            f"{input_dir}/slice2.csv",
            f"{input_dir}/slice2_coor.csv",
            f"{input_dir}/slice3.csv",
            f"{input_dir}/slice3_coor.csv",
            "-d",
            f"{tmp_path}",
        ]
    )
    assert result.returncode == 0
    assert_frame_equal(
        pd.read_csv(tmp_path / f"paste_output/slice1_slice2_pairwise.csv"),
        pd.read_csv(output_dir / "slices_1_2_pairwise.csv"),
    )
