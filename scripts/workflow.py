import logging

from paste3.experimental import AlignmentDataset

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    dataset = AlignmentDataset(
        glob_pattern="/home/vineetb/paste3/paste_reproducibility/data/SCC/cached-results/H5ADs/patient_2*"
    )

    all_points_orig = dataset.all_points()

    cluster_indices = set()
    for slice in dataset.slices:
        clusters = set(slice.get_obs_values("original_clusters"))
        cluster_indices |= clusters
    n_clusters = len(cluster_indices)

    # ------- Center Align ------- #
    center_slice, pis = dataset.find_center_slice()
    aligned_dataset, *_ = dataset.center_align(center_slice=center_slice, pis=pis)
    all_points = aligned_dataset.all_points()

    center_slice.cluster(n_clusters, save_as="new_clusters")

    new_clusters = center_slice.get_obs_values("new_clusters")
    # ------- Center Align ------- #

    # ------- Pairwise Align ------- #
    aligned_dataset, *_ = dataset.pairwise_align(overlap_fraction=0.7)
    all_points = aligned_dataset.all_points()
    # ------- Pairwise Align ------- #
