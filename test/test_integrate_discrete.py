"""
Test the integrate_discrete function
used in DifferencedProcess and elsewhere
"""
import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_almost_equal

from pyrenew.math import integrate_discrete


@pytest.mark.parametrize(
    ["order", "n_diffs"], [[1, 250], [2, 40], [3, 10], [4, 10], [5, 5]]
)
def test_integrator_correctness(order, n_diffs):
    """
    Test that the scan-based integrate function built in
    to DifferencedProcess works equivalently
    to a manual implementation.
    """
    diffs = jax.random.normal(key=jax.random.key(54), shape=(n_diffs,))
    inits = jax.random.normal(key=jax.random.key(45), shape=(order,))
    result_manual = diffs
    for init in jnp.flip(inits):
        result_manual = jnp.cumsum(jnp.hstack([init, result_manual]))

    result_integrator = integrate_discrete(inits, diffs)
    assert result_integrator.shape == (n_diffs + order,)
    assert_array_almost_equal(result_manual, result_integrator, decimal=4)
    assert result_integrator[0] == inits[0]


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
