from pathlib import Path
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from paste2.glmpca import (
    ortho,
    mat_binom_dev,
    remove_intercept,
    glmpca_init,
    est_nb_theta,
    glmpca,
)

test_dir = Path(__file__).parent
input_dir = test_dir / "data/input"
output_dir = test_dir / "data/output"


def test_ortho():
    U = np.genfromtxt(input_dir/'cell_factors.csv', delimiter=',',skip_header=1)
    V = np.genfromtxt(input_dir/'loadings_onto_genes.csv', delimiter=',', skip_header=1)
    A = np.genfromtxt(input_dir/'coeffX.csv', delimiter=',', ndmin=1)
    Z = np.genfromtxt(input_dir/'gene_specific_covariates.csv', delimiter=',')
    G = None

    res = ortho(U, V, A.T, X=1, G=G, Z=Z)
    print('breakpoints')

def test_mat_binom_dev():
    print("breakpoint")


def test_remove_intercept():
    X = np.genfromtxt(input_dir/'cell_specific_covariates.csv', delimiter=',', defaultfmt='[%.18e]')
    output = remove_intercept(X)
    print("breakpoint")


def test_glmpca_init():
    Y = np.genfromtxt(input_dir/'Y.csv', delimiter=',', skip_header=2)

    gnt = glmpca_init(Y, 'poi', None, 100)
    print("breakpoint")


def test_est_nb_theta():
    print("breakpoint")


def test_glmpca():
    joint_matrix_T = np.genfromtxt(input_dir / "joint_matrix.csv", delimiter=",")

    res = glmpca(joint_matrix_T, L=50, penalty=1)
    assert_frame_equal(
        pd.DataFrame(res["coefX"], columns=[str(i) for i in range(res["coefX"].shape[1])]),
        pd.read_csv(input_dir / "glmpca_coefX.csv"),
    )
    assert_frame_equal(
        pd.DataFrame(res["dev"], columns=[str(i) for i in range(res["dev"].shape[1])]),
        pd.read_csv(input_dir / "glmpca_dev.csv"),
    )
    assert_frame_equal(
        pd.DataFrame(res["factors"], columns=[str(i) for i in range(res["factors"].shape[1])]),
        pd.read_csv(input_dir / "glmpca_factors.csv"),
    )
    assert_frame_equal(
        pd.DataFrame(res["loadings"], columns=[str(i) for i in range(res["loadings"].shape[1])]),
        pd.read_csv(input_dir / "glmpca_loadings.csv"),
    )
    assert_frame_equal(
        pd.DataFrame(res["coefZ"], columns=[str(i) for i in range(res["coefZ"].shape[1])]),
        pd.read_csv(input_dir / "glmpca_coefZ.csv"),
    )