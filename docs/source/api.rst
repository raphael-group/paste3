API
===

.. automodule:: paste3

Alignment
~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api

    paste.pairwise_align
    paste.center_align
    paste.center_ot
    paste.center_NMF
    paste.my_fused_gromov_wasserstein
    paste.line_search_partial

Visualization
~~~~~~~~~~~~~

.. autosummary::
   :toctree: api

    visualization.stack_slices_pairwise
    visualization.stack_slices_center
    visualization.plot_slice
    visualization.generalized_procrustes_analysis

Model Selection
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api

    model_selection.generate_graph
    model_selection.convex_hull_edge_inconsistency
    model_selection.select_overlap_fraction


Miscellaneous
~~~~~~~~~~~~~

.. autosummary::
   :toctree: api

   helper.kl_divergence
   helper.glmpca_distance
   helper.pca_distance
   helper.high_umi_gene_distance
   helper.norm_and_center_coordinates
   helper.get_common_genes
   helper.match_spots_using_spatial_heuristic
   helper.dissimilarity_metric
