import pandas as pd
from pandas.testing import assert_frame_equal
from pathlib import Path
from collections import namedtuple
from paste3.paste_cmd_line import main as paste_cmd_line

test_dir = Path(__file__).parent
input_dir = test_dir / "data/input"
output_dir = test_dir / "data/output"

args = namedtuple(
    "args",
    [
        "filename",
        "mode",
        "direc",
        "alpha",
        "cost",
        "n_components",
        "lmbda",
        "initial_slice",
        "threshold",
        "coordinates",
        "weights",
        "start",
        "seed",
    ],
)


def test_cmd_line_center(tmp_path):
    print(f"Running command in {tmp_path}")
    result = paste_cmd_line(
        args(
            [
                f"{input_dir}/slice1.csv",
                f"{input_dir}/slice1_coor.csv",
                f"{input_dir}/slice2.csv",
                f"{input_dir}/slice2_coor.csv",
                f"{input_dir}/slice3.csv",
                f"{input_dir}/slice3_coor.csv",
            ],
            "center",
            f"{tmp_path}",
            0.1,
            "kl",
            15,
            [],
            1,
            0.001,
            False,
            [],
            [],
            0,
        )
    )

    assert result is None
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
    result = paste_cmd_line(
        args(
            [
                f"{input_dir}/slice1.csv",
                f"{input_dir}/slice1_coor.csv",
                f"{input_dir}/slice2.csv",
                f"{input_dir}/slice2_coor.csv",
                f"{input_dir}/slice3.csv",
                f"{input_dir}/slice3_coor.csv",
            ],
            "pairwise",
            f"{tmp_path}",
            0.1,
            "kl",
            15,
            [],
            1,
            0.001,
            False,
            [],
            [],
            0,
        )
    )

    assert result is None
    assert_frame_equal(
        pd.read_csv(tmp_path / "paste_output/slice1_slice2_pairwise.csv"),
        pd.read_csv(output_dir / "slices_1_2_pairwise.csv"),
    )
