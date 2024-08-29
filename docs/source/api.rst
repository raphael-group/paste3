API
===

Import Paste as::
    from paste3.glmpca import ortho, mat_binom_dev, glmpca_init, est_nb_theta, glmpca
    from paste3.model_selection import create_graph, generate_graph_from_labels, edge_inconsistency_score, calculate_convex_hull_edge_inconsistency, select_overlap_function, plot_edge_curve, select_overlap_fraction_plotting
    from paste3.visualization import plot_slice, stack_slices_pairwise, stack_slices_center
    from paste3.projection import partial_stack_slices_pairwise, partial_procrustes_analysis
    from paste3.paste import pairwise_align, center_align
    from paste3.helper import filter_for_common_genes, match_spots_using_spatial_heuristic
    from paste3.paste2 import partial_pairwise_align_given_cost_matrix, partial_pairwise_align_histology, partial_pairwise_align, partial_fused_gromov_wasserstein, gwgrad_partial

.. automodule:: paste3

PASTE Alignment
~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api

    paste.pairwise_align
    paste.center_align

PASTE2 Alignment
~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api

    paste2.partial_pairwise_align_given_cost_matrix
    paste2.partial_pairwise_align_histology
    paste2.partial_pairwise_align
    paste2.partial_fused_gromov_wasserstein
    paste2.gwgrad_partial

Visualization
~~~~~~~~~~~~~

.. autosummary::
   :toctree: api

    visualization.stack_slices_pairwise
    visualization.stack_slices_center
    visualization.plot_slice

Projection
~~~~~~~~~~~~~

.. autosummary::
   :toctree: api

    projection.partial_stack_slices_pairwise
    projection.partial_procrustes_analysis

Model Selection
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api

    model_selection.create_graph
    model_selection.generate_graph_from_labels
    model_selection.edge_inconsistency_score
    model_selection.calculate_convex_hull_edge_inconsistency
    model_selection.plot_edge_curve
    model_selection.select_overlap_fraction_plotting


GLMPCA
~~~~~~~

.. autosummary::
   :toctree: api

    glmpca.ortho
    glmpca.mat_binom_dev
    glmpca.glmpca_init
    glmpca.est_nb_theta
    glmpca.glmpca


Miscellaneous
~~~~~~~~~~~~~

.. autosummary::
   :toctree: api

   helper.filter_for_common_genes
   helper.match_spots_using_spatial_heuristic

