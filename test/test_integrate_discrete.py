"""
Test the integrate_discrete function
used in DifferencedProcess and elsewhere
"""

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from pyrenew.math import integrate_discrete


@pytest.mark.parametrize(
    ["order", "diff_shape"],
    [
        [1, (250,)],
        [2, (40, 40, 2)],
        [3, (10, 3, 5)],
        [4, (10, 1, 1, 2)],
        [5, (5, 1, 1, 1)],
    ],
)
def test_integrator_correctness(order, diff_shape):
    """
    Test that the scan-based integrate_discrete() function
    in pyrenew.math works equivalently to a manual
    implementation using a for-loop.
    """
    n_diffs = diff_shape[0]
    batch_shape = diff_shape[1:]
    init_shape = (order,) + batch_shape
    diffs = jax.random.normal(key=jax.random.key(54), shape=diff_shape)
    inits = jax.random.normal(key=jax.random.key(45), shape=init_shape)
    result_manual = diffs
    for init in jnp.flip(inits, axis=0):
        result_manual = jnp.cumsum(
            jnp.concatenate([init[jnp.newaxis, ...], result_manual], axis=0),
            axis=0,
        )

    result_integrator = integrate_discrete(inits, diffs)
    assert result_integrator.shape == (n_diffs + order,) + diff_shape[1:]
    assert_array_almost_equal(result_manual, result_integrator, decimal=4)
    assert_array_equal(result_integrator[0], inits[0])


@pytest.mark.parametrize(
    ["diffs", "inits", "expected_solution"],
    [
        [
            jnp.array([0.25, 0.5, 0.5]),
            jnp.array([0]),
            jnp.array([0, 0.25, 0.75, 1.25]),
        ],
        [jnp.array([1, 1, 1]), jnp.array([0, 2]), jnp.array([0, 2, 5, 9, 14])],
        [
            jnp.array([[1, 0], [1, 1], [1, 1]]),
            0.1,
            jnp.array([[0.1, 0.1], [1.1, 0.1], [2.1, 1.1], [3.1, 2.1]]),
        ],
        [
            jnp.array([[1, 0], [1, 1], [1, 1]]),
            jnp.array([[0.1, 0.2]]),
            jnp.array([[0.1, 0.2], [1.1, 0.2], [2.1, 1.2], [3.1, 2.2]]),
        ],
    ],
)
def test_manual_integrator_correctness(diffs, inits, expected_solution):
    """
    Test the integrator correctness with manually computed
    solutions.
    """
    result = integrate_discrete(inits, diffs)
    assert_array_almost_equal(result, expected_solution)


@pytest.mark.parametrize(
    ["diff_shape", "init_shape"],
    [
        [(50, 3), (2, 2)],
        [(12, 5, 10), (4, 5, 11)],
        [
            (
                36,
                12,
            ),
            (2, 12, 3),
        ],
    ],
)
def test_integrator_shape_validation(diff_shape, init_shape):
    """
    Test that appropriate ValueErrors
    are raised when input shapes to integrate_discrete
    are incompatible
    """
    diffs = jnp.zeros(diff_shape)
    inits = jnp.zeros(init_shape)
    with pytest.raises(ValueError, match="Non-time dimensions"):
        integrate_discrete(inits, diffs)
