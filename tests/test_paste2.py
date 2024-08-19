from pathlib import Path
import pandas as pd
from paste2.PASTE2 import partial_pairwise_align

test_dir = Path(__file__).parent
input_dir = test_dir / "data/input"
output_dir = test_dir / "data/output"

from pandas.testing import assert_frame_equal


# TODO: this takes 3 years to pass, see whats going on here
def test_partial_pairwise_align(slices2):
    pi_BC = partial_pairwise_align(slices2[0], slices2[1], s=0.7)

    assert_frame_equal(
        pd.DataFrame(pi_BC, index=None),
        pd.read_csv(output_dir / "partial_pairwise_align.csv"),
        rtol=1e-05,
    )
