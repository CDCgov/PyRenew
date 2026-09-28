"""
Unit tests for the pyrenew.math module.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.random import RandomState
from numpy.testing import (
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)

import pyrenew.math as pmath

rng = RandomState(5)


@pytest.mark.parametrize(
    "arr, arr_len",
    [
        ([3, 1, 2], 3),
        (np.ones(50), 50),
        ((jnp.nan * jnp.ones(250)).reshape((50, -1)), 250),
    ],
)
def test_positive_ints_like(arr, arr_len):
    """
    Test the _positive_ints_like helper function.
    """
    result = pmath._positive_ints_like(arr)
    expected = jnp.arange(1, arr_len + 1)
    assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "R, G",
    [
        (5, rng.dirichlet(np.ones(2))),
        (0.2, rng.dirichlet(np.ones(50))),
        (1, rng.dirichlet(np.ones(10))),
        (1.01, rng.dirichlet(np.ones(4))),
        (0.99, rng.dirichlet(np.ones(6))),
    ],
)
def test_r_approx(R, G):
    """
    Test that r_approx_from_R gives answers
    consistent with those gained from a Leslie
    matrix approach.
    """
    r_val = pmath.r_approx_from_R(R, G, n_newton_steps=5)
    e_val, stable_dist = pmath.get_asymptotic_growth_rate_and_age_dist(R, G)

    unnormed = r_val * stable_dist
    if r_val != 0:
        assert_array_almost_equal(unnormed / np.sum(unnormed), stable_dist)
    else:
        assert_almost_equal(e_val, 1, decimal=5)


def test_r_approx_vectorized_matches_scalar():
    """
    Test that r_approx_from_R with an array of R values
    gives the same answers as calling it once per scalar
    R value.
    """
    vec_rng = RandomState(11)
    G = vec_rng.dirichlet(np.ones(8))
    R_vec = jnp.array([0.5, 0.99, 1.0, 1.01, 1.5, 3.0])
    r_vec = pmath.r_approx_from_R(R_vec, G, n_newton_steps=8)
    r_expected = jnp.array(
        [pmath.r_approx_from_R(float(R), G, n_newton_steps=8) for R in R_vec]
    )
    assert r_vec.shape == R_vec.shape
    assert_array_almost_equal(r_vec, r_expected)


def test_r_approx_vectorized_satisfies_defining_equation():
    """
    Test that each entry of a vectorized r_approx_from_R
    result satisfies the defining equation
    R * M_-(r) - 1 == 0 for its own R value.
    """
    vec_rng = RandomState(12)
    G = vec_rng.dirichlet(np.ones(6))
    R_vec = jnp.array([0.7, 1.2, 2.5])
    r_vec = pmath.r_approx_from_R(R_vec, G, n_newton_steps=8)
    residuals = R_vec * pmath.neg_MGF(r_vec, G) - 1
    assert residuals.shape == R_vec.shape
    assert_array_almost_equal(residuals, jnp.zeros_like(residuals), decimal=5)


def test_r_approx_vectorized_multidimensional():
    """
    Test that r_approx_from_R preserves the shape of a
    multidimensional R array and matches scalar calls
    entry by entry.
    """
    vec_rng = RandomState(13)
    G = vec_rng.dirichlet(np.ones(5))
    R_mat = jnp.array([[0.8, 1.0, 1.4], [2.0, 0.9, 1.1]])
    r_mat = pmath.r_approx_from_R(R_mat, G, n_newton_steps=8)
    assert r_mat.shape == R_mat.shape
    for i in range(R_mat.shape[0]):
        for j in range(R_mat.shape[1]):
            r_scalar = pmath.r_approx_from_R(float(R_mat[i, j]), G, n_newton_steps=8)
            assert_almost_equal(float(r_mat[i, j]), float(r_scalar))


def test_neg_MGF_batched():
    """
    Test that neg_MGF and neg_MGF_del_r evaluate
    independently for each entry of an array-valued r.
    """
    vec_rng = RandomState(14)
    w = vec_rng.dirichlet(np.ones(7))
    r_vec = jnp.array([-0.1, 0.0, 0.2])
    mgf_vec = pmath.neg_MGF(r_vec, w)
    dmgf_vec = pmath.neg_MGF_del_r(r_vec, w)
    assert mgf_vec.shape == r_vec.shape
    assert dmgf_vec.shape == r_vec.shape
    for k in range(len(r_vec)):
        assert_almost_equal(float(mgf_vec[k]), float(pmath.neg_MGF(float(r_vec[k]), w)))
        assert_almost_equal(
            float(dmgf_vec[k]), float(pmath.neg_MGF_del_r(float(r_vec[k]), w))
        )


def test_r_approx_scalar_still_scalar():
    """
    Test backward compatibility: a scalar R still yields
    a scalar r.
    """
    G = np.array([0.2, 0.1, 0.2, 0.15, 0.05, 0.025, 0.025, 0.25])
    R = 1.2
    r_val = pmath.r_approx_from_R(R, G, n_newton_steps=5)
    assert jnp.asarray(r_val).shape == ()
    assert_almost_equal(float(R * pmath.neg_MGF(r_val, G) - 1), 0.0, decimal=5)


def test_asymptotic_properties():
    """
    Check that the calculated
    asymptotic growth rate and
    age distribution given by
    get_asymptotic_growth_rate()
    and get_stable_age_distribution()
    agree with simulated ones from
    just running a process for a
    while.
    """
    R = 1.2
    gi = np.array([0.2, 0.1, 0.2, 0.15, 0.05, 0.025, 0.025, 0.25])
    A = pmath.get_leslie_matrix(R, gi)

    # check via Leslie matrix multiplication
    x = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    for i in range(1000):
        x_new = A @ x
        rat_x = np.sum(x_new) / np.sum(x)
        x = x_new

    assert_almost_equal(rat_x, pmath.get_asymptotic_growth_rate(R, gi), decimal=5)
    assert_array_almost_equal(x / np.sum(x), pmath.get_stable_age_distribution(R, gi))

    # check via backward-looking convolution
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    for j in range(1000):
        new_pop = np.dot(y, R * gi)
        rat_y = new_pop / y[0]
        y = np.hstack([new_pop, y[:-1]])
    assert_almost_equal(rat_y, pmath.get_asymptotic_growth_rate(R, gi), decimal=5)
    assert_array_almost_equal(y / np.sum(x), pmath.get_stable_age_distribution(R, gi))


@pytest.mark.parametrize(
    "R, gi, expected",
    [
        (
            0.4,
            np.array([0.4, 0.2, 0.2, 0.1, 0.1]),
            np.array(
                [
                    [0.16, 0.08, 0.08, 0.04, 0.04],
                    [1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                ]
            ),
        ),
        (
            3,
            np.array([0.4, 0.2, 0.2, 0.1, 0.1]),
            np.array(
                [
                    [1.2, 0.6, 0.6, 0.3, 0.3],
                    [1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                ]
            ),
        ),
    ],
)
def test_get_leslie(R, gi, expected):
    """
    Test that get_leslie matrix
    returns expected Leslie matrices
    """
    assert_array_almost_equal(pmath.get_leslie_matrix(R, gi), expected)
