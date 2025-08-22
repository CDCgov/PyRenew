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
