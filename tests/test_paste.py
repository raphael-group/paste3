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
    temp_dir = tempfile.mkdtemp()
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
    outcome_df.to_csv(f"{temp_dir}/slices_1_2_pairwise_csv")
    assert_checksum_equals(
        f"{temp_dir}/slices_1_2_pairwise_csv", f"{output_dir}/slices_1_2_pairwise.csv"
    )

def test_center_alignment(slices):
    n_slices = len(slices)
    center_align(
        slices[0],
        slices,
        lmbda=n_slices * [1.0 / n_slices],
        alpha=0.1,
        n_components=15,
        threshold=0.001,
        dissimilarity="kl",
        distributions=[slices[i].obsm["weights"] for i in range(len(slices))],
    )
