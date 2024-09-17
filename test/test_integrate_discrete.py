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
    Test that the scan-based integrate function built in
    to DifferencedProcess works equivalently
    to a manual implementation.
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
    ],
)
def test_manual_integrator_correctness(diffs, inits, expected_solution):
    """
    Test the integrator correctness with manually computed
    solutions.
    """
    result = integrate_discrete(inits, diffs)
    assert_array_almost_equal(result, expected_solution)
