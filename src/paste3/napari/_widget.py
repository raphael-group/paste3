import napari
import seaborn as sns
from magicgui.widgets import Container, create_widget
from napari.utils.progress import progress

from paste3.experimental import AlignmentDataset

face_color_cycle = sns.color_palette("Paired", 20)


class AlignContainer(Container):
    def valid_slice_layers(self):
        return [
            layer
            for layer in self._viewer.layers
            if isinstance(layer, napari.layers.Points) and "slice" in layer.metadata
        ]

    def show_dataset(
        self,
        dataset: AlignmentDataset,
        show_slices: bool = True,
        show_volume: bool = True,
        first_layer_translation: list[float] | None = None,
    ):
        all_clusters = []
        for slice in dataset:
            points = slice.adata.obsm["spatial"]
            clusters = slice.get_obs_values(self._spot_color_key_dropdown.value)
            all_clusters.extend(clusters)

            if show_slices:
                self._viewer.add_points(
                    points,
                    features={"cluster": clusters},
                    face_color="cluster",
                    face_color_cycle=face_color_cycle,
                    size=1,
                    metadata={"slice": slice},
                    name=f"{slice}",
                )

        if show_volume:
            self._viewer.add_points(
                dataset.all_points(translation=first_layer_translation),
                features={"cluster": all_clusters},
                face_color="cluster",
                face_color_cycle=face_color_cycle,
                ndim=3,
                size=1,
                scale=[5, 1, 1],
                name=f"{dataset}",
            )


class CenterAlignContainer(AlignContainer):
    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        alignment_dataset: AlignmentDataset | None = None,
    ):
        super().__init__()
        self._viewer = viewer
        if alignment_dataset is None:
            slices = [layer.metadata["slice"] for layer in self.valid_slice_layers()]
            self.dataset = AlignmentDataset(slices=slices)
        else:
            self.dataset = alignment_dataset

        self._reference_slice_dropdown = create_widget(
            label="Reference Slice",
            annotation=str,
            widget_type="ComboBox",
            options={"choices": [str(slice) for slice in self.dataset]},
        )

        keys = list(self.dataset.slices[0].adata.obs.keys())
        spot_color_key = None
        for key in keys:
            if "cluster" in key:
                spot_color_key = key
                break
        if spot_color_key is None and len(keys) > 0:
            spot_color_key = keys[0]

        self._spot_color_key_dropdown = create_widget(
            label="Spot Color Key",
            annotation=str,
            widget_type="ComboBox",
            # Show first slices's obs keys, assume all slices have the same.
            options={"choices": keys},
            value=spot_color_key,
        )

        self._slice_weights_textbox = create_widget(
            label="Slice Weights", annotation=str
        )

        self._alpha_slider = create_widget(
            label="Alpha", annotation=float, widget_type="FloatSlider", value=0.1
        )
        self._alpha_slider.min = 0
        self._alpha_slider.max = 1

        self._n_components_textbox = create_widget(
            label="No. of components", annotation=int, value=15
        )

        self._threshold_textbox = create_widget(
            label="Threshold", annotation=float, value=0.001
        )

        self._max_iterations_textbox = create_widget(
            label="Max Iterations", annotation=int, value=10
        )

        self._exp_dis_metric_dropdown = create_widget(
            label="Dissimilarity",
            annotation=str,
            widget_type="ComboBox",
            options={"choices": ["kl", "euclidean"]},
            value="kl",
        )

        self._norm_checkbox = create_widget(
            label="Normalize", annotation=bool, widget_type="CheckBox", value=False
        )

        self._random_seed_textbox = create_widget(
            label="Random Seed", annotation=int, value=0
        )

        self._run_button = create_widget(
            label="Run", annotation=None, widget_type="PushButton"
        )

        self._run_button.changed.connect(self._run)

        self.extend(
            [
                self._reference_slice_dropdown,
                self._spot_color_key_dropdown,
                self._slice_weights_textbox,
                self._alpha_slider,
                self._n_components_textbox,
                self._threshold_textbox,
                self._max_iterations_textbox,
                self._exp_dis_metric_dropdown,
                self._norm_checkbox,
                self._random_seed_textbox,
                self._run_button,
            ]
        )

    def _run(self):
        cluster_indices = set()
        for slice in self.dataset:
            clusters = set(slice.get_obs_values(self._spot_color_key_dropdown.value))
            cluster_indices |= clusters
        n_clusters = len(cluster_indices)

        reference_slice = self._viewer.layers[
            self._reference_slice_dropdown.value
        ].metadata["slice"]

        try:
            slice_weights = [
                float(w) for w in self._slice_weights_textbox.value.split(",")
            ]
        except ValueError:
            slice_weights = None

        with progress(total=self._max_iterations_textbox.value) as pbar:
            center_slice, pis = self.dataset.find_center_slice(
                initial_slice=reference_slice,
                slice_weights=slice_weights,
                alpha=self._alpha_slider.value,
                n_components=self._n_components_textbox.value,
                threshold=self._threshold_textbox.value,
                max_iter=self._max_iterations_textbox.value,
                exp_dissim_metric=self._exp_dis_metric_dropdown.value,
                norm=self._norm_checkbox.value,
                random_seed=self._random_seed_textbox.value,
                pbar=pbar,
            )

        aligned_dataset, _, translations = self.dataset.center_align(
            center_slice=center_slice, pis=pis
        )
        # We'll translate all points w.r.t translations in the first layer
        # so that the first layer in original volume and the aligned volume are
        # coincident
        first_layer_translation = translations[0][0].detach().cpu().numpy()

        self.show_dataset(
            aligned_dataset,
            show_slices=False,
            show_volume=True,
            first_layer_translation=first_layer_translation,
        )

        # Show center slice
        center_slice_points = center_slice.adata.obsm["spatial"]
        center_slice_clusters = center_slice.cluster(n_clusters)
        self._viewer.add_points(
            center_slice_points,
            ndim=2,
            size=1,
            features={"cluster": center_slice_clusters},
            face_color="cluster",
            face_color_cycle=face_color_cycle,
            name="paste3_center_slice",
        )


class PairwiseAlignContainer(AlignContainer):
    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        alignment_dataset: AlignmentDataset | None = None,
    ):
        super().__init__()
        self._viewer = viewer
        if alignment_dataset is None:
            slices = [layer.metadata["slice"] for layer in self.valid_slice_layers()]
            self.dataset = AlignmentDataset(slices=slices)
        else:
            self.dataset = alignment_dataset

        keys = list(self.dataset.slices[0].adata.obs.keys())
        spot_color_key = None
        for key in keys:
            if "cluster" in key:
                spot_color_key = key
                break
        if spot_color_key is None and len(keys) > 0:
            spot_color_key = keys[0]

        self._spot_color_key_dropdown = create_widget(
            label="Spot Color Key",
            annotation=str,
            widget_type="ComboBox",
            # Show first slices's obs keys, assume all slices have the same.
            options={"choices": keys},
            value=spot_color_key,
        )

        self._overlap_slider = create_widget(
            label="Overlap", annotation=float, widget_type="FloatSlider", value=0.7
        )
        self._overlap_slider.min = 0
        self._overlap_slider.max = 1

        self._max_iterations_textbox = create_widget(
            label="Max Iterations", annotation=int, value=10
        )

        self._run_button = create_widget(
            label="Run", annotation=None, widget_type="PushButton"
        )

        self._run_button.changed.connect(self._run)

        self.extend(
            [
                self._spot_color_key_dropdown,
                self._overlap_slider,
                self._max_iterations_textbox,
                self._run_button,
            ]
        )

    def _run(self):
        aligned_dataset, _, translations = self.dataset.pairwise_align(
            overlap_fraction=self._overlap_slider.value,
            max_iters=self._max_iterations_textbox.value,
        )
        # We'll translate all points w.r.t translations in the first layer
        # so that the first layer in original volume and the aligned volume are
        # coincident
        first_layer_translation = translations[0][0].detach().cpu().numpy()

        self.show_dataset(
            aligned_dataset,
            show_slices=False,
            show_volume=True,
            first_layer_translation=first_layer_translation,
        )


def init_widget(alignment_dataset: AlignmentDataset):
    viewer = napari.current_viewer()

    center_align_container = CenterAlignContainer(viewer, alignment_dataset)
    viewer.window.add_dock_widget(center_align_container, name="Paste3 Center Align")

    pairwise_align_container = PairwiseAlignContainer(viewer, alignment_dataset)
    viewer.window.add_dock_widget(
        pairwise_align_container, name="Paste3 Pairwise Align"
    )

    # We could also have done
    # pairwise_align_container.show_dataset(alignment_dataset)
    # but we only need to add the dataset to the viewer once
    center_align_container.show_dataset(alignment_dataset)
