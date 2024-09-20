Welcome to PASTE3 documentation!
=================================

**PASTE3 (WIP)** package that provides combined functionality of PASTE and PASTE2.
**PASTE** is a computational method that leverages both gene expression similarity and spatial distances between spots to align and integrate spatial transcriptomics data, and
**PASTE2**, the extension of PASTE, is a method for partial alignment and 3D reconstruction of spatial transcriptomics slices when they do not fully overlap in space.
In particular, PASTE3 combines PASTE and PASTE2 to provide five main functionalities:

1. Pairwise Alignment: align spots across pairwise slices.
2. Center Alignment: integrate multiple slices into one center slice.
3. Partial Pairwise Alignment: given a pair of slices and their overlap percentage, find a partial alignment matrix.
4. Select Overlap: decide the overlap percentage between two slices
5. Partial Stack Slices Pairwise: given a sequence of consecutive slices and the partial alignments between them,
project all slices onto the same 2D coordinate system. 3D reconstruction can be done by assigning a z-value to each slice.

.. image:: _static/images/paste_overview.png
    :alt: PASTE Overview Figure
    :width: 800px
    :align: center
|

.. image:: _static/images/paste2.png
    :alt: PASTE2 Overview Figure
    :width: 800px
    :align: center
|

Manuscript
----------

You can view PASTE `preprint <https://www.biorxiv.org/content/10.1101/2021.03.16.435604v1>`_ on **bioRxiv**.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   paste3/installation.md
   api
   tutorial
