import hashlib
from pathlib import Path
import pandas as pd
import tempfile

from src.paste import pairwise_align, center_align

test_dir = Path(__file__).parent
input_dir = test_dir / "data/input"
output_dir = test_dir / "data/output"


def assert_checksum_equals(generated_file, oracle):
    assert (
        hashlib.md5(
            "".join(open(generated_file, "r").readlines()).encode("utf8")
        ).hexdigest()
        == hashlib.md5(
            "".join(open(oracle, "r").readlines()).encode("utf8")
        ).hexdigest()
    )


def test_pairwise_alignment(slices):
    temp_dir = Path(tempfile.mkdtemp())
    outcome = pairwise_align(
        slices[0],
        slices[1],
        alpha=0.1,
        dissimilarity="kl",
        a_distribution=slices[0].obsm["weights"],
        b_distribution=slices[1].obsm["weights"],
        G_init=None,
    )
    outcome_df = pd.DataFrame(
        outcome, index=slices[0].obs.index, columns=slices[1].obs.index
    )
    outcome_df.to_csv(temp_dir / "slices_1_2_pairwise.csv")
    assert_checksum_equals(
        temp_dir / "slices_1_2_pairwise.csv", output_dir / "slices_1_2_pairwise.csv"
    )


# TODO: possibly some randomness to the code that needs to be dealt with
def test_center_alignment(slices):
    temp_dir = Path(tempfile.mkdtemp())

    n_slices = len(slices)
    center_slice, pairwise_info = center_align(
        slices[0],
        slices,
        lmbda=n_slices * [1.0 / n_slices],
        alpha=0.1,
        n_components=15,
        threshold=0.001,
        dissimilarity="kl",
        distributions=[slices[i].obsm["weights"] for i in range(len(slices))],
    )
    pd.DataFrame(center_slice.uns["paste_W"], index=center_slice.obs.index).to_csv(
        temp_dir / "W_center.csv"
    )
    pd.DataFrame(center_slice.uns["paste_H"], columns=center_slice.var.index).to_csv(
        temp_dir / "H_center.csv"
    )

    # assert_checksum_equals(temp_dir / "W_center.csv", output_dir / "W_center.csv")
    #
    # assert_checksum_equals(temp_dir / "H_center.csv", output_dir / "H_center.csv")

    for i, pi in enumerate(pairwise_info):
        pd.DataFrame(
            pi, index=center_slice.obs.index, columns=slices[i].obs.index
        ).to_csv(temp_dir / f"center_slice{i+1}_pairwise.csv")
        # assert_checksum_equals(
        #     temp_dir / f"center_slice{i}_pairwise.csv",
        #     output_dir / f"center_slice{i}_pairwise.csv",
        # )
