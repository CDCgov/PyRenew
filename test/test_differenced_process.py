"""
Unit tests for the DifferencedProcess class
"""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from numpy.testing import assert_array_almost_equal

from pyrenew.deterministic import DeterministicVariable, NullVariable
from pyrenew.process import (
    DifferencedProcess,
    IIDRandomSequence,
    StandardNormalSequence,
)
from pyrenew.randomvariable import DistributionalVariable


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
            name="test_diff",
            fundamental_process=None,
            differencing_order=wrong_type_order,
        )
    DifferencedProcess.assert_valid_differencing_order(1)
    _ = DifferencedProcess(
        name="test_diff", fundamental_process=None, differencing_order=1
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
            name="test_diff",
            fundamental_process=None,
            differencing_order=wrong_value,
        )

    DifferencedProcess.assert_valid_differencing_order(right_value)
    _ = DifferencedProcess(
        name="test_diff",
        fundamental_process=None,
        differencing_order=right_value,
    )


@pytest.mark.parametrize(
    ["fundamental_process", "differencing_order", "init_diff_vals"],
    [
        [
            IIDRandomSequence(
                name="test_iid_seq",
                element_rv=DistributionalVariable(
                    "element_dist", dist.Cauchy(0.02, 0.3)
                ),
            ),
            3,
            jnp.array([0.25, 0.67, 5]),
        ],
        [
            StandardNormalSequence(
                name="test_std_norm_seq", element_rv_name="test_stand_norm"
            ),
            5,
            jnp.array([0.23, 5.2, 1, 0.2, 3]),
        ],
    ],
)
def test_differenced_process_sample(
    fundamental_process, differencing_order, init_diff_vals
):
    """
    Test that differenced processes can be sampled,
    that they yield the correct sample shapes, and that
    they raise errors if non-feasible sample lengths are
    requested.
    """
    proc = DifferencedProcess(
        name="test_diff",
        differencing_order=differencing_order,
        fundamental_process=fundamental_process,
    )

    n_long = differencing_order + 1032
    n_long_alt = differencing_order + 235
    n_one_diff = differencing_order + 1
    n_no_diffs = differencing_order
    n_no_diffs_alt = differencing_order - 1
    n_fail = -1
    n_fail_alt = 0
    with numpyro.handlers.seed(rng_seed=6723):
        samp = proc.sample(n=n_long, init_vals=init_diff_vals)
        samp_alt = proc.sample(n=n_long_alt, init_vals=init_diff_vals)
        samp_one_diff = proc.sample(n=n_one_diff, init_vals=init_diff_vals)
        samp_no_diffs = proc.sample(n=n_no_diffs, init_vals=init_diff_vals)
        samp_no_diffs_alt = proc.sample(n=n_no_diffs_alt, init_vals=init_diff_vals)
    assert samp.shape == (n_long,)
    assert samp_alt.shape == (n_long_alt,)
    assert samp_one_diff.shape == (n_one_diff,)
    assert samp_no_diffs.shape == (n_no_diffs,)
    assert samp_no_diffs_alt.shape == (n_no_diffs_alt,)

    with numpyro.handlers.seed(rng_seed=7834):
        with pytest.raises(ValueError, match="must be positive"):
            proc.sample(n=n_fail, init_vals=init_diff_vals)
        with pytest.raises(ValueError, match="must be positive"):
            proc.sample(n=n_fail_alt, init_vals=init_diff_vals)
        with pytest.raises(
            ValueError,
            match=("Must have exactly as many initial difference values"),
        ):
            proc.sample(n=n_long, init_vals=jnp.atleast_2d(init_diff_vals))


@pytest.mark.parametrize(
    ["fundamental_process", "inits", "n", "expected_solution"],
    [
        [
            IIDRandomSequence(
                name="test_iid_zero",
                element_rv=DeterministicVariable("zero", jnp.array(0.0)),
            ),
            jnp.array([0.0, 0, 0, 0, 0]),
            3,
            jnp.array([0.0, 0.0, 0.0]),
        ],
        [
            IIDRandomSequence(
                name="test_iid_one",
                element_rv=DeterministicVariable("zero", jnp.array(1.0)),
            ),
            jnp.array([0]),
            5,
            jnp.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        ],
        [
            IIDRandomSequence(
                name="test_iid_one",
                element_rv=DeterministicVariable("zero", jnp.array(1.0)),
            ),
            jnp.array([0, 1]),
            7,
            jnp.array([0.0, 1.0, 3.0, 6.0, 10.0, 15.0, 21.0]),
        ],
        [
            IIDRandomSequence(
                name="test_iid_one",
                element_rv=DeterministicVariable("zero", jnp.array(1.0)),
            ),
            jnp.array([0, 1]),
            1,
            jnp.array([0.0]),
        ],
        [
            IIDRandomSequence(
                name="test_iid_one",
                element_rv=DeterministicVariable("zero", jnp.array(1.0)),
            ),
            jnp.array([0, 1]),
            2,
            jnp.array([0.0, 1.0]),
        ],
    ],
)
def test_manual_difference_process_sample(
    fundamental_process, inits, n, expected_solution
):
    """
    Test the correctness of DifferencedProcess.sample()
    with manually computed solutions
    """
    proc = DifferencedProcess(
        name="test_diff",
        differencing_order=len(inits),
        fundamental_process=fundamental_process,
    )
    result = proc.sample(n=n, init_vals=inits)
    assert_array_almost_equal(result, expected_solution)
