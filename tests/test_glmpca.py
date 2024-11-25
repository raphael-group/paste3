from pathlib import Path

import numpy as np
import pytest

from paste3.glmpca import (
    est_nb_theta,
    glmpca,
    glmpca_init,
    mat_binom_dev,
    ortho,
    remove_intercept,
)

test_dir = Path(__file__).parent / "data"


def test_ortho():
    data = np.load(test_dir / "test_ortho.npz")
    outcome = ortho(data["U"], data["V"], data["A"], X=1, G=None, Z=data["Z"])

    assert np.allclose(outcome["factors"], data["factors"])
    assert np.allclose(outcome["loadings"], data["loadings"])
    assert np.allclose(outcome["coefX"], data["coefX"])
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

    for i, j in zip(outcome, expected_outcome, strict=False):
        assert np.all(np.isclose(i, j))


@pytest.mark.parametrize("fam", ["poi", "nb", "mult", "bern"])
def test_glmpca_init(fam):
    data = np.load(test_dir / "glmpca_intercepts.npz")

    glmpca_obj = glmpca_init(data["input"], fam, None, 100)
    assert np.allclose(glmpca_obj["intercepts"], data[fam])


def test_est_nb_theta():
    y = np.array([1, 2, 3])
    mu = np.array([1.5, 2.5, 3.5])
    th = 0.5

    output = est_nb_theta(y, mu, th)
    expected_output = 1.8467997201907858

    assert output == expected_output


@pytest.mark.parametrize("fam", ["poi", "nb", "mult", "bern"])
def test_glmpca(fam):
    data = np.load(test_dir / "glmpca_result.npz")
    input = data["input"]
    if fam == "bern":
        input /= np.linalg.norm(input, axis=1, keepdims=True)
    res = glmpca(
        input,
        L=50,
        penalty=1,
        fam=fam,
        ctl={"maxIter": 10, "eps": 1e-4, "optimizeTheta": True},
    )

    assert np.allclose(res["coefX"], data[f"coefX_{fam}"])
    assert np.allclose(res["loadings"], data[f"loadings_{fam}"])
    assert np.allclose(res["factors"], data[f"factors_{fam}"])
    assert np.allclose(res["dev"], data[f"dev_{fam}"])
    assert res["coefZ"] is None


def test_glmpca_covariates():
    data = np.load(test_dir / "covariates.npz")

    X = np.array(range(505))
    Z = np.array(range(2001))

    input = data["input"]
    res = glmpca(
        input,
        L=50,
        penalty=1,
        fam="poi",
        X=X[:, None],
        Z=Z[:, None],
        ctl={"maxIter": 10, "eps": 1e-4, "optimizeTheta": True},
    )

    assert np.allclose(res["coefX"], data["coefX"])
    assert np.allclose(res["loadings"], data["loadings"])
    assert np.allclose(res["factors"], data["factors"])
    assert np.allclose(res["dev"], data["dev"])
    assert np.allclose(res["coefZ"], data["coefZ"])


def test_glmpca_with_init():
    data = np.load(test_dir / "glmpca_init.npz")
    res = glmpca(
        data["input"],
        L=50,
        penalty=1,
        fam="poi",
        init={"factors": data["i_factors"], "loadings": data["i_loadings"]},
        ctl={"maxIter": 10, "eps": 1e-4, "optimizeTheta": True},
    )
    assert np.allclose(res["coefX"], data["coefX"])
    assert np.allclose(res["loadings"], data["o_loadings"])
    assert np.allclose(res["factors"], data["o_factors"])
    assert np.allclose(res["dev"], data["dev"])
    assert res["coefZ"] is None
