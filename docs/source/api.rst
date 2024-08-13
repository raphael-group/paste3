API
===

Import Paste as::

   from paste.visualization import plot_slice, stack_slices_pairwise, stack_slices_center
   from paste.PASTE import pairwise_align, center_align
   from paste.helper import filter_for_common_genes, match_spots_using_spatial_heuristic, match_spots_using_spatial_heuristic, apply_trsf

.. automodule:: paste

Alignment
~~~~~~~~~

.. autosummary::
   :toctree: api

    PASTE.pairwise_align
    PASTE.center_align

Visualization
~~~~~~~~~~~~~

.. autosummary::
   :toctree: api

    visualization.stack_slices_pairwise
    visualization.stack_slices_center
    visualization.plot_slice

Miscellaneous
~~~~~~~~~~~~~

.. autosummary::
   :toctree: api

   helper.filter_for_common_genes
   helper.match_spots_using_spatial_heuristic
   helper.apply_trsf


