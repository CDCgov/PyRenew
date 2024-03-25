# -*- coding: utf-8 -*-

import jax.numpy as jnp
import pyrenew.math as pmath
from numpy.testing import assert_array_almost_equal


def test_get_leslie():
    """
    Test that get_leslie matrix
    returns expected Leslie matrices
    """

    gi = jnp.array([0.4, 0.2, 0.2, 0.1, 0.1])
    R_a = 0.4
    R_b = 3.0
    expected_a = jnp.array(
        [
            [0.16, 0.08, 0.08, 0.04, 0.04],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
        ]
    )
    expected_b = jnp.array(
        [
            [1.2, 0.6, 0.6, 0.3, 0.3],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
        ]
    )

    assert_array_almost_equal(pmath.get_leslie_matrix(R_a, gi), expected_a)
    assert_array_almost_equal(pmath.get_leslie_matrix(R_b, gi), expected_b)
