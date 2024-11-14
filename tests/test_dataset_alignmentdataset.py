from pathlib import Path

import numpy as np
import torch

from paste3.dataset import AlignmentDataset, Slice

test_dir = Path(__file__).parent
input_dir = test_dir / "data/input"


def test_dataset(sample_data_files):
    dataset = AlignmentDataset(file_paths=sample_data_files)
    # common prefix of slice file paths
    assert str(dataset) == "paste3_sample_patient_2"
    assert len(dataset) == 3  # 3 slices
    for slice in dataset:
        assert isinstance(slice, Slice)


def test_dataset_pairwise_align(sample_data_files):
    dataset = AlignmentDataset(file_paths=sample_data_files)
    aligned_dataset, rotation_angles, translations = dataset.pairwise_align(
        overlap_fraction=0.7
    )

    assert len(aligned_dataset) == 3  # same number of slices, but aligned

    assert np.allclose(rotation_angles[0].detach().cpu().numpy(), 0.07333706179204347)
    assert np.allclose(rotation_angles[1].detach().cpu().numpy(), -0.05378653217862863)

    assert np.allclose(
        torch.stack(translations[0]).detach().cpu().numpy(),
        [[18.48282774, 22.53792994], [19.00173271, 14.72333356]],
    )
    assert np.allclose(
        torch.stack(translations[1]).detach().cpu().numpy(),
        [[5.39930207, 4.79523229], [19.06410298, 20.61153107]],
    )


def test_dataset_center_align(sample_data_files):
    dataset = AlignmentDataset(file_paths=sample_data_files)
    center_slice, pis = dataset.find_center_slice(max_iter=1)
    assert isinstance(center_slice, Slice)
    aligned_dataset = dataset.center_align(center_slice=center_slice, pis=pis)
    assert len(aligned_dataset) == 3  # same number of slices, but aligned
