"""
Unit tests for the DifferencedProcess class
"""

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_almost_equal

from pyrenew.deterministic import NullVariable
from pyrenew.process import DifferencedProcess


@pytest.mark.parametrize(
    "wrong_type_order", ["test", jnp.array([5.2]), 1.0, NullVariable()]
)
def test_differencing_order_type_validation(wrong_type_order):
    """
    Test that passing something other than an
    integer as the differencing_order raises
    an error via the dedicated assertion function,
    that valid types do pass validation, and
    that this function is correctly used for
    type validation at object instantiation.
    """
    err_match = "must be an integer"
    with pytest.raises(ValueError, match=err_match):
        DifferencedProcess.assert_valid_differencing_order(wrong_type_order)
    with pytest.raises(ValueError, match=err_match):
        _ = DifferencedProcess(
            "should_fail",
            fundamental_process=None,
            differencing_order=wrong_type_order,
        )
    DifferencedProcess.assert_valid_differencing_order(1)
    _ = DifferencedProcess(
        "should_succeed", fundamental_process=None, differencing_order=1
    )


@pytest.mark.parametrize(
    ["wrong_value", "right_value"], [[0, 1], [-5, 5], [-10325235, 300]]
)
def test_differencing_order_value_validation(wrong_value, right_value):
    """
    Test that passing an integer that is less than 1
    as the differencing_order raises a ValueError via
    the dedicated assertion function, that valid
    values do pass, and that the validation function
    is correctly used for value validation at
    object instantiation.
    """
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        DifferencedProcess.assert_valid_differencing_order(wrong_value)
        _ = DifferencedProcess(
            "should_fail",
            fundamental_process=None,
            differencing_order=wrong_value,
        )

    DifferencedProcess.assert_valid_differencing_order(right_value)
    _ = DifferencedProcess(
        "should_succeed",
        fundamental_process=None,
        differencing_order=right_value,
    )


@pytest.mark.parametrize(
    ["order", "diffs"],
    [
        [1, jnp.array([1.0, 2, -3])],
        [2, jnp.array([1.0, 2, -3])],
        [3, jnp.array([1.0, 2, -3])],
        [4, jnp.array([1.0, 2, -3])],
    ],
)
def test_integrator_init_validation(order, diffs):
    """
    Test that when the integrator is called,
    it succeeds if and only if the right number
    of initial values have been specified, and raises
    the appropriate ValueError otherwise.
    """
    inits_short = jnp.ones(order - 1)
    inits_correct = jnp.ones(order)
    inits_long = jnp.ones(order + 1)
    proc = DifferencedProcess(
        f"test_difference_process_of_order_{order}",
        fundamental_process=None,
        differencing_order=order,
    )
    with pytest.raises(
        ValueError, match="exactly as many initial difference values"
    ):
        proc.integrate(inits_short, diffs)
    with pytest.raises(
        ValueError, match="exactly as many initial difference values"
    ):
        proc.integrate(inits_long, diffs)
    proc.integrate(inits_correct, diffs)


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

    proc = DifferencedProcess(
        "test_process", fundamental_process=None, differencing_order=order
    )
    result_proc1 = proc.integrate(inits, diffs)
    assert result_proc1.shape == (n_diffs + order,)
    assert_array_almost_equal(result_manual, result_proc1, decimal=5)
    assert result_proc1[0] == inits[0]


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
    order = inits.size
    proc = DifferencedProcess(
        "test_proc", fundamental_process=None, differencing_order=order
    )
    result = proc.integrate(inits, diffs)
    assert_array_almost_equal(result, expected_solution)
