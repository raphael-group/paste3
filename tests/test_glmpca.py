from pathlib import Path
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from paste3.glmpca import (
    ortho,
    mat_binom_dev,
    remove_intercept,
    glmpca_init,
    est_nb_theta,
    glmpca,
)
import pytest

test_dir = Path(__file__).parent
input_dir = test_dir / "data/input"
output_dir = test_dir / "data/output"


def test_ortho():
    U = np.genfromtxt(input_dir / "cell_factors.csv", delimiter=",", skip_header=1)
    V = np.genfromtxt(
        input_dir / "loadings_onto_genes.csv", delimiter=",", skip_header=1
    )
    A = np.genfromtxt(input_dir / "coeffX.csv", delimiter=",", ndmin=2)
    Z = np.genfromtxt(input_dir / "gene_specific_covariates.csv", delimiter=",")
    G = None

    outcome = ortho(U, V, A, X=1, G=G, Z=Z)

    assert_frame_equal(
        pd.DataFrame(outcome["factors"], columns=[str(i) for i in range(50)]),
        pd.read_csv(output_dir / "ortho_factors.csv"),
    )
    assert_frame_equal(
        pd.DataFrame(outcome["loadings"], columns=[str(i) for i in range(50)]),
        pd.read_csv(output_dir / "ortho_loadings.csv"),
    )
    assert_frame_equal(
        pd.DataFrame(outcome["coefX"], columns=["0"]),
        pd.read_csv(output_dir / "ortho_coefX.csv"),
    )
    assert outcome["coefZ"] is None


def test_mat_binom_dev():
    X = np.array([[1, 2, 3], [4, 5, 6]])
    P = np.array([[0.5, 0.4, 0.1], [0.2, 0.3, 0.5]])
    n = np.array([1, 2, 3])
    outcome = mat_binom_dev(X, P, n)

    assert outcome == 80.67099373045231


def test_remove_intercept():
    X = np.array([[1, 2, 3], [4, 5, 6]])
    outcome = remove_intercept(X)
    expected_outcome = [[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]]

    for i, j in zip(outcome, expected_outcome):
        assert np.all(np.isclose(i, j))


@pytest.mark.parametrize("fam", ("poi", "nb", "mult", "bern"))
def test_glmpca_init(fam):
    Y = np.genfromtxt(input_dir / "Y.csv", delimiter=",", skip_header=2)

    glmpca_obj = glmpca_init(Y, fam, None, 100)

    assert np.allclose(
        glmpca_obj["intercepts"],
        np.load(output_dir / "glmpca_intercepts.npz")[fam],
    )


def test_est_nb_theta():
    y = np.array([1, 2, 3])
    mu = np.array([1.5, 2.5, 3.5])
    th = 0.5

    output = est_nb_theta(y, mu, th)
    expected_output = 1.8467997201907858

    assert output == expected_output


@pytest.mark.parametrize("fam", ("poi", "nb", "mult", "bern"))
def test_glmpca(fam):
    joint_matrix_T = np.genfromtxt(input_dir / "joint_matrix.csv", delimiter=",")
    if fam == "bern":
        joint_matrix_T /= np.linalg.norm(joint_matrix_T, axis=1, keepdims=True)
    res = glmpca(
        joint_matrix_T,
        L=50,
        penalty=1,
        verbose=True,
        fam=fam,
        ctl={"maxIter": 10, "eps": 1e-4, "optimizeTheta": True},
    )

    saved_result = np.load(output_dir / "glmpca_result.npz")
    assert np.allclose(res["coefX"], saved_result[f"coefX_{fam}"])
    assert np.allclose(res["loadings"], saved_result[f"loadings_{fam}"])
    assert np.allclose(res["factors"], saved_result[f"factors_{fam}"])
    assert np.allclose(res["dev"], saved_result[f"dev_{fam}"])
    assert res["coefZ"] is None


def test_glmpca_covariates():
    X = np.array(range(505))
    Z = np.array(range(2001))
    joint_matrix_T = np.genfromtxt(input_dir / "joint_matrix.csv", delimiter=",")
    res = glmpca(
        joint_matrix_T,
        L=50,
        penalty=1,
        verbose=True,
        fam="poi",
        X=X[:, None],
        Z=Z[:, None],
        ctl={"maxIter": 10, "eps": 1e-4, "optimizeTheta": True},
    )

    saved_result = np.load(output_dir / "covariates.npz")
    assert np.allclose(res["coefX"], saved_result["coefX"])
    assert np.allclose(res["loadings"], saved_result["loadings"])
    assert np.allclose(res["factors"], saved_result["factors"])
    assert np.allclose(res["dev"], saved_result["dev"])
