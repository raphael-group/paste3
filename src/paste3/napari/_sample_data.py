from pathlib import Path

import seaborn as sns

from paste3.experimental import AlignmentDataset
from paste3.napari.data.ondemand import get_file


def make_sample_data():
    remote_files = [
        "paste3_sample_patient_2_slice_0.h5ad",
        "paste3_sample_patient_2_slice_1.h5ad",
        "paste3_sample_patient_2_slice_2.h5ad",
    ]
    local_files = [get_file(file) for file in remote_files]  # paths to local files
    dataset = AlignmentDataset(file_paths=[Path(file) for file in local_files])

    data = []  # list of 3-tuples (data, kwargs, layer_type)

    all_clusters = []
    for slice in dataset:
        points = slice.adata.obsm["spatial"]
        clusters = slice.get_obs_values("original_clusters")
        all_clusters.extend(clusters)

        _data = (
            points,
            {
                "features": {"cluster": clusters},
                "face_color": "cluster",
                "face_color_cycle": sns.color_palette("Paired", 20),
                "size": 1,
                "metadata": {"slice": slice},
                "name": f"{slice}",
            },
            "points",
        )
        data.append(_data)

    # add volume
    data.append(
        (
            dataset.all_points(),
            {
                "features": {"cluster": all_clusters},
                "face_color": "cluster",
                "face_color_cycle": sns.color_palette("Paired", 20),
                "ndim": 3,
                "size": 1,
                "scale": [5, 1, 1],
                "name": f"{dataset}",
            },
            "points",
        )
    )

    return data
