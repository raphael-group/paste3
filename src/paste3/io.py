"""This module provide functions to load and process data from ST experiments."""

import scanpy as sc
import numpy as np

from pathlib import Path
from collections import defaultdict
import logging


logger = logging.getLogger(__name__)


def process_files(g_fpath, s_fpath, w_fpath=None):
    """
    Processes gene expression files and associated spatial and weight files,
    returning a list of AnnData objects with the relevant data loaded into
    appropriate attributes.

    This function supports two file types: CSV and HDF5 (H5AD). For CSV files,
    it expects corresponding spatial and weight files to be provided.

    Parameters
    ----------
    g_fpath : List[str]
        List of file paths to the gene expression files (CSV or H5AD).

    s_fpath : List[str]
        List of file paths to the corresponding spatial files (CSV).

    w_fpath : List[str], Optional
        List of file paths to the corresponding weight files (CSV). If not
        provided, weights are initialized uniformly.

    Returns
    -------
    List[AnnData]
        A list of AnnData objects, each containing gene expression data,
        spatial information, and weights.
    """

    ext = Path(g_fpath[0]).suffix

    if ext == ".csv":
        if not (len(s_fpath) == len(g_fpath)):
            ValueError("Length of spatial files doesn't equal number of gene files")
        _slices = defaultdict()
        for file in g_fpath:
            # The header of this file is alphanumeric, so this file has to be imported as a string
            _slices[get_shape(file)[0]] = sc.read_csv(file)

        for file in s_fpath:
            try:
                _slice = _slices[get_shape(file)[0]]
            except KeyError:
                raise ValueError("Incomplete information for a slice")
            else:
                _slice.obsm["spatial"] = np.genfromtxt(
                    file, delimiter=",", dtype="float64"
                )

        if w_fpath:
            if not (len(w_fpath) == len(g_fpath)):
                ValueError("Length of weight files doesn't equal number of gene files")
            for file in w_fpath:
                _slice = _slices[get_shape(file)[0]]
                _slice.obsm["weights"] = np.genfromtxt(
                    file, delimiter=",", dtype="float64"
                )
        else:
            for k, v in _slices.items():
                v.obsm["weights"] = np.ones((v.shape[0],)) / v.shape[0]

        slices = list(_slices.values())
    elif ext == ".h5ad":
        slices = [sc.read_h5ad(file) for file in g_fpath]

    else:
        raise ValueError("Incorrect file type provided ")

    return slices


def get_shape(file_path):
    """Determines the shape of data inside of a csv file without opening it."""

    def is_numeric(value):
        """Determine if the passed value is numeric."""
        try:
            float(value)
            return True
        except ValueError:
            return False

    with open(file_path, "r") as file:
        first_line = file.readline().strip()
        num_columns = len(first_line.split(","))

        num_rows = sum(1 for _ in file)

        # Determine if the first row is a header
        if all(is_numeric(val) for val in first_line.split(",")):
            num_rows += 1

    return num_rows, num_columns
