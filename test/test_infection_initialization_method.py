# numpydoc ignore=GL08
import numpy as np
import numpy.testing as testing
import pytest

from pyrenew.deterministic import DeterministicVariable
from pyrenew.latent import (
    InitializeInfectionsExponentialGrowth,
    InitializeInfectionsFromVec,
    InitializeInfectionsZeroPad,
)


def test_initialize_infections_exponential():
    """Check that the InitializeInfectionsExponentialGrowth class generates the correct number of infections at each time point."""
    n_timepoints = 10
    default_t_pre_init = n_timepoints - 1
    t_pre_init = 6

    rate_RV = DeterministicVariable(name="rate_RV", value=np.array([0.5, 0.1]))
    rate_scalar_RV = DeterministicVariable(name="rate_RV", value=0.5)

    rate = rate_RV()
    rate_scalar = rate_scalar_RV()

    I_pre_init = np.array([5.0, 10.0])
    I_pre_init_scalar = 5.0

    # both rate and I_pre_init are arrays with default t_pre_init
    result = InitializeInfectionsExponentialGrowth(
        n_timepoints, rate_rv=rate_RV
    ).initialize_infections(I_pre_init)

    manual_result = np.column_stack(
        [
            I_pre_init[0]
            * np.exp(rate[0] * (np.arange(n_timepoints) - default_t_pre_init)),
            I_pre_init[1]
            * np.exp(rate[1] * (np.arange(n_timepoints) - default_t_pre_init)),
        ]
    )

    ## check that the result is as expected
    testing.assert_array_almost_equal(result, manual_result)

    ## check that infections at default t_pre_init is I_pre_init
    testing.assert_array_equal(result[default_t_pre_init], I_pre_init)

    # both rate and I_pre_init are arrays with custom t_pre_init
    result = InitializeInfectionsExponentialGrowth(
        n_timepoints, rate_rv=rate_RV, t_pre_init=t_pre_init
    ).initialize_infections(I_pre_init)

    manual_result = np.column_stack(
        [
            I_pre_init[0] * np.exp(rate[0] * (np.arange(n_timepoints) - t_pre_init)),
            I_pre_init[1] * np.exp(rate[1] * (np.arange(n_timepoints) - t_pre_init)),
        ]
    )

    ## check that the result is as expected
    testing.assert_array_almost_equal(result, manual_result)

    ## check that infections at t_pre_init is I_pre_init
    testing.assert_array_equal(result[t_pre_init], I_pre_init)

    # rate is array, I_pre_init is scalar with default t_pre_init
    result = InitializeInfectionsExponentialGrowth(
        n_timepoints, rate_rv=rate_RV
    ).initialize_infections(I_pre_init_scalar)

    manual_result = np.column_stack(
        [
            I_pre_init_scalar
            * np.exp(rate[0] * (np.arange(n_timepoints) - default_t_pre_init)),
            I_pre_init_scalar
            * np.exp(rate[1] * (np.arange(n_timepoints) - default_t_pre_init)),
        ]
    )

    ## check that the result is as expected
    testing.assert_array_almost_equal(result, manual_result)

    ## check that infections at default t_pre_init is I_pre_init
    testing.assert_array_equal(result[default_t_pre_init], I_pre_init_scalar)

    # rate is scalar, I_pre_init is array with default t_pre_init
    result = InitializeInfectionsExponentialGrowth(
        n_timepoints, rate_rv=rate_scalar_RV
    ).initialize_infections(I_pre_init)

    manual_result = np.column_stack(
        [
            I_pre_init[0]
            * np.exp(rate_scalar * (np.arange(n_timepoints) - default_t_pre_init)),
            I_pre_init[1]
            * np.exp(rate_scalar * (np.arange(n_timepoints) - default_t_pre_init)),
        ]
    )

    ## check that the result is as expected
    testing.assert_array_almost_equal(result, manual_result)

    ## check that infections at default t_pre_init is I_pre_init
    testing.assert_array_equal(result[default_t_pre_init], I_pre_init)

    # both rate and I_pre_init are scalar with default t_pre_init
    result = InitializeInfectionsExponentialGrowth(
        n_timepoints, rate_rv=rate_scalar_RV
    ).initialize_infections(I_pre_init_scalar)

    manual_result = I_pre_init_scalar * np.exp(
        rate_scalar * (np.arange(n_timepoints) - default_t_pre_init)
    )

    ## check that the result is as expected
    testing.assert_array_almost_equal(result, manual_result)

    ## check that infections at default t_pre_init is I_pre_init
    testing.assert_array_equal(result[default_t_pre_init], I_pre_init_scalar)


def test_initialize_infections_zero_pad():
    """Check that the InitializeInfectionsZeroPad class generates the correct number of infections at each time point."""

    n_timepoints = 10
    I_pre_init_RV = DeterministicVariable(name="I_pre_init_RV", value=10.0)
    I_pre_init = I_pre_init_RV()
    I_pre_init = I_pre_init

    infections = InitializeInfectionsZeroPad(n_timepoints).initialize_infections(
        I_pre_init
    )

    manual_infections = np.pad(
        np.atleast_1d(I_pre_init),
        (n_timepoints - np.array(I_pre_init).size, 0),
    )

    testing.assert_array_equal(infections, manual_infections)

    I_pre_init_RV_2 = DeterministicVariable(
        name="I_pre_init_RV", value=np.array([10.0, 10.0])
    )

    I_pre_init_2 = I_pre_init_RV_2()
    I_pre_init_2 = I_pre_init_2

    infections_2 = InitializeInfectionsZeroPad(n_timepoints).initialize_infections(
        I_pre_init_2
    )
    testing.assert_array_equal(
        infections_2,
        np.pad(I_pre_init_2, (n_timepoints - I_pre_init_2.size, 0)),
    )

    # Check that the InitializeInfectionsZeroPad class raises an error when the length of I_pre_init is greater than n_timepoints.
    with pytest.raises(ValueError):
        InitializeInfectionsZeroPad(1).initialize_infections(I_pre_init_2)


def test_initialize_infections_from_vec():
    """Check that the InitializeInfectionsFromVec class generates the correct number of infections at each time point."""
    n_timepoints = 10
    I_pre_init = np.arange(n_timepoints)

    infections = InitializeInfectionsFromVec(n_timepoints).initialize_infections(
        I_pre_init
    )
    testing.assert_array_equal(infections, I_pre_init)

    I_pre_init_2 = np.arange(n_timepoints - 1)
    with pytest.raises(ValueError):
        InitializeInfectionsFromVec(n_timepoints).initialize_infections(I_pre_init_2)

    n_timepoints_float = 10.0
    with pytest.raises(TypeError):
        InitializeInfectionsFromVec(n_timepoints_float).initialize_infections(
            I_pre_init
        )

    n_timepoints_neg = -10
    with pytest.raises(ValueError):
        InitializeInfectionsFromVec(n_timepoints_neg).initialize_infections(I_pre_init)
