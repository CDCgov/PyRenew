# -*- coding: utf-8 -*-
# numpydoc ignore=GL08

import jax.numpy as jnp
import pyrenew.math as pmath
from numpy.testing import assert_almost_equal, assert_array_almost_equal


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
    gi = jnp.array([0.2, 0.1, 0.2, 0.15, 0.05, 0.025, 0.025, 0.25])
    A = pmath.get_leslie_matrix(R, gi)

    # check via Leslie matrix multiplication
    x = jnp.array([1, 0, 0, 0, 0, 0, 0, 0])
    for i in range(1000):
        x_new = A @ x
        rat_x = jnp.sum(x_new) / jnp.sum(x)
        x = x_new

    assert_almost_equal(
        rat_x, pmath.get_asymptotic_growth_rate(R, gi), decimal=5
    )
    assert_array_almost_equal(
        x / jnp.sum(x), pmath.get_stable_age_distribution(R, gi)
    )

    # check via backward-looking convolution
    y = jnp.array([1, 0, 0, 0, 0, 0, 0, 0])
    for j in range(1000):
        new_pop = jnp.dot(y, R * gi)
        rat_y = new_pop / y[0]
        y = jnp.hstack([new_pop, y[:-1]])
    assert_almost_equal(
        rat_y, pmath.get_asymptotic_growth_rate(R, gi), decimal=5
    )
    assert_array_almost_equal(
        y / jnp.sum(x), pmath.get_stable_age_distribution(R, gi)
    )
