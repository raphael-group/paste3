from paste3.dataset import AlignmentDataset
from paste3.helper import wait
from paste3.napari._reader import napari_get_reader
from paste3.napari._sample_data import make_sample_data
from paste3.napari._widget import CenterAlignContainer, PairwiseAlignContainer


def test_reader(sample_data_files, make_napari_viewer_proxy):
    # Get a fake viewer from napari fixtures
    viewer = make_napari_viewer_proxy()
    # Initialize a reader, emulating `open files as stack..` menu option
    reader = napari_get_reader(sample_data_files)
    # Read the files
    reader(sample_data_files)

    layers = viewer.layers

    # We should have 4 Point layers, one for each slice
    # and one for the volume
    assert len(layers) == 4

    assert layers[0].name == "paste3_sample_patient_2_slice_0"
    assert layers[1].name == "paste3_sample_patient_2_slice_1"
    assert layers[2].name == "paste3_sample_patient_2_slice_2"
    assert layers[3].name == "paste3_sample_patient_2"


def test_sample_data():
    layer_data = make_sample_data("paste3_sample_patient_2")
    # We should have 4 Point layers, one for each slice
    # and one for the volume
    assert len(layer_data) == 4


def test_center_align_widget(sample_data_files, make_napari_viewer_proxy):
    dataset = AlignmentDataset(file_paths=sample_data_files)
    viewer = make_napari_viewer_proxy()

    widget = CenterAlignContainer(viewer, dataset)
    widget.show_dataset()

    # emulate UI interaction
    widget._reference_slice_dropdown.value = "paste3_sample_patient_2_slice_0"
    widget._max_iterations_textbox.value = "1"

    wait(widget._run())  # wait for completion
    layers = viewer.layers

    # We should have 6 Point layers
    #   one for each slice (3),
    #   one for the original volume (1)
    #   one for the aligned volume (1)
    #   one for the center slice (1)
    assert len(layers) == 6

    assert layers[0].name == "paste3_sample_patient_2_slice_0"
    assert layers[1].name == "paste3_sample_patient_2_slice_1"
    assert layers[2].name == "paste3_sample_patient_2_slice_2"
    assert layers[3].name == "paste3_sample_patient_2"
    assert layers[4].name == "paste3_sample_patient_2_center_aligned"
    assert layers[5].name == "paste3_center_slice"


def test_pairwise_align_widget(sample_data_files, make_napari_viewer_proxy):
    dataset = AlignmentDataset(file_paths=sample_data_files)
    viewer = make_napari_viewer_proxy()

    widget = PairwiseAlignContainer(viewer, dataset)
    widget.show_dataset()

    # emulate UI interaction
    widget._max_iterations_textbox.value = "1"

    widget.run()

    layers = viewer.layers

    # We should have 5 Point layers
    #   one for each slice (3),
    #   one for the original volume (1)
    #   one for the aligned volume (1)
    assert len(layers) == 5

    assert layers[0].name == "paste3_sample_patient_2_slice_0"
    assert layers[1].name == "paste3_sample_patient_2_slice_1"
    assert layers[2].name == "paste3_sample_patient_2_slice_2"
    assert layers[3].name == "paste3_sample_patient_2"
    assert layers[4].name == "paste3_sample_patient_2_pairwise_aligned"
