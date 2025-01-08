[![Tests](https://github.com/raphael-group/paste3/actions/workflows/test_pinned_deps.yml/badge.svg)](https://github.com/raphael-group/paste3/actions/workflows/test_pinned_deps.yml)
[![Coverage Status](https://coveralls.io/repos/github/raphael-group/paste3/badge.svg?branch=main)](https://coveralls.io/github/raphael-group/paste3?branch=main)
[![Docs](https://github.com/raphael-group/paste3/actions/workflows/docs.yml/badge.svg)](https://raphael-group.github.io/paste3/)

<video src="https://github.com/user-attachments/assets/977c05c0-4c45-4d21-9302-dfe23800937e"/>


# Paste 3

**Paste 3** (Paste + Paste 2) is a Python package and NAPARI plugin that
provides advanced alignment methods of Spatial Transcriptonomics (ST) data
as detailed in the following publications:

### 1. *PASTE*
**Zeira, R., Land, M., Strzalkowski, A., et al.**
*Alignment and integration of spatial transcriptomics data.*
**Nat Methods**, 19, 567–575 (2022).

[Read the publication](https://doi.org/10.1038/s41592-022-01459-6)  
[Original PASTE code](https://github.com/raphael-group/paste)

---

### 2. *PASTE2*
**Liu X, Zeira R, Raphael BJ.**
*Partial alignment of multislice spatially resolved transcriptomics data.*
**Genome Res.** 2023 Jul; 33(7):1124-1132.
[Read the publication](https://doi.org/10.1101/gr.277670.123)  
[Original PASTE2 code](https://github.com/raphael-group/paste2)

The motivation behind PASTE3 is to provide a NAPARI plugin
for practitioners to experiment with both PASTE and PASTE2 at an operational
level, as well as provide a common codebase for future development of ST
alignment algorithms. (`Paste-N`..)

PASTE3 is built on `pytorch` and can leverage a GPU for performance if
available, though it is able to run just fine in the absence of a GPU, on all
major platforms.

Auto-generated documentation for the PASTE3 package is available [here](https://raphael-group.github.io/paste3/).

Additional examples and the code to reproduce the original PASTE paper's analyses are available [here](https://github.com/raphael-group/paste_reproducibility). Preprocessed datasets used in the paper can be found on [zenodo](https://doi.org/10.5281/zenodo.6334774).

## Overview

![PASTE Overview](https://github.com/raphael-group/paste/blob/main/docs/source/_static/images/paste_overview.png)

The PASTE series of algorithms provide computational methods that leverage both
gene expression  similarity and spatial distances between spots to align and
integrate spatial transcriptomics data. In particular, there are two modes of
operation:
1. `Pairwise-Alignment`: align spots between successive pairs of slices.
2. `Center-Alignment`: infer a `center slice` (low sparsity, low variance) and
align all slices with respect to this center slice.


### Installation

The easiest way is to install PASTE3 is using `pip`:

`pip install git+https://github.com/raphael-group/paste3.git`

Developers who wish to work with `paste3` in Python will likely want to review
the detailed [installation](https://raphael-group.github.io/paste3/installation)
page.


### Getting Started

If you intend to use PASTE3 as a `napari` plugin, install `paste3` in a python
environment that has `napari` installed, or install `napari` after having
installed `paste3` as above.

`pip install napari`

Open one of the sample datasets we provide (`File->Open Sample->Paste3->SCC Patient..`)
and then select one of the two modes of PASTE3 operations
(`Plugins->Paste3->Center Align` or `Plugins->Paste3->Pairwise Align`).

Your own datasets can be used if they're in the .h5ad format, with each file denoting a single
slice. With the default parameters, alignment should take a couple of minutes, though
you have the option of changing these to suit your needs.

<video src="https://github.com/user-attachments/assets/a527aa12-190e-46ed-a843-4cc10f8146ce"/>

If you intend to use PASTE3 programmatically in your Python code, follow along
the [Getting Started](https://raphael-group.github.io/paste3/notebooks/paste_tutorial.html)
tutorial.
