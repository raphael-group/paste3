import napari
import seaborn as sns
from magicgui.widgets import Container, create_widget
from napari.utils.progress import progress

from paste3.experimental import AlignmentDataset

face_color_cycle = sns.color_palette("Paired", 20)


class CenterAlignContainer(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer

        self._reference_slice_dropdown = create_widget(
            label="Reference Slice",
            annotation=str,
            widget_type="ComboBox",
            options={"choices": [layer.name for layer in self.valid_slice_layers()]},
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

        # connect your own callbacks
        self._run_button.changed.connect(self._run)

        # append into/extend the container with your widgets
        self.extend(
            [
                self._reference_slice_dropdown,
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

    def valid_slice_layers(self):
        return [
            layer
            for layer in self._viewer.layers
            if isinstance(layer, napari.layers.Points) and "slice" in layer.metadata
        ]

    def _run(self):
        slices = [layer.metadata["slice"] for layer in self.valid_slice_layers()]
        dataset = AlignmentDataset(slices=slices)

        cluster_indices = set()
        for slice in dataset.slices:
            clusters = set(slice.get_obs_values("original_clusters"))
            cluster_indices |= clusters
        n_clusters = len(cluster_indices)

        # Align !
        # We could do
        # dataset.align(center_align=True, overlap_fraction=self._alpha_slider.value, max_iters=self._max_iterations_textbox.value)
        # but we need to do it in parts so we can get `center_slice`
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
            center_slice, pis = dataset.find_center_slice(
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

        aligned_dataset = dataset.center_align(center_slice=center_slice, pis=pis)

        self._viewer.add_points(
            aligned_dataset.all_points(),
            ndim=3,
            size=1,
            scale=(3, 1, 1),
            name="paste3_center_aligned_volume",
        )

        center_slice_points = center_slice.adata.obsm["spatial"]
        center_slice_clusters = center_slice.cluster(n_clusters)
        self._viewer.add_points(
            center_slice_points,
            ndim=2,
            size=1,
            features={"cluster": center_slice_clusters},
            face_color="cluster",
            face_color_cycle=face_color_cycle,
            name="paste3_aligned_center_slice",
        )


class PairwiseAlignContainer(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer

        self._overlap_slider = create_widget(
            label="Overlap", annotation=float, widget_type="FloatSlider", value=0.1
        )
        self._overlap_slider.min = 0
        self._overlap_slider.max = 1

        self._max_iterations_textbox = create_widget(
            label="Max Iterations", annotation=int, value=10
        )

        self._run_button = create_widget(
            label="Run", annotation=None, widget_type="PushButton"
        )

        # connect your own callbacks
        self._run_button.changed.connect(self._run)

        # append into/extend the container with your widgets
        self.extend(
            [
                self._overlap_slider,
                self._max_iterations_textbox,
                self._run_button,
            ]
        )

    def valid_slice_layers(self):
        return [
            layer
            for layer in self._viewer.layers
            if isinstance(layer, napari.layers.Points) and "slice" in layer.metadata
        ]

    def _run(self):
        slices = [layer.metadata["slice"] for layer in self.valid_slice_layers()]
        dataset = AlignmentDataset(slices=slices)

        aligned_dataset = dataset.pairwise_align(
            overlap_fraction=self._overlap_slider.value,
            max_iters=self._max_iterations_textbox.value,
        )

        self._viewer.add_points(
            aligned_dataset.all_points(),
            ndim=3,
            size=1,
            scale=(3, 1, 1),
            name="paste3_pairwise_aligned_volume",
        )
