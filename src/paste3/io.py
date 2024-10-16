import scanpy as sc
import numpy as np

from pathlib import Path
from collections import defaultdict
import logging


logger = logging.getLogger(__name__)


def process_files(g_fpath, s_fpath, w_fpath=None):
    """Returns a list of AnnData objects."""

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
    """Determines the shapes of the csv without opening the files"""

    def is_numeric(value):
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
