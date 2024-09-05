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
    rate_RV = DeterministicVariable(name="rate_RV", value=0.5)
    I_pre_init_RV = DeterministicVariable(name="I_pre_init_RV", value=10.0)
    default_t_pre_init = n_timepoints - 1

    (I_pre_init,) = I_pre_init_RV()
    (rate,) = rate_RV()

    I_pre_init = I_pre_init.value
    rate = rate.value
    infections_default_t_pre_init = InitializeInfectionsExponentialGrowth(
        n_timepoints, rate_rv=rate_RV
    ).initialize_infections(I_pre_init)
    infections_default_t_pre_init_manual = I_pre_init * np.exp(
        rate * (np.arange(n_timepoints)[:, np.newaxis] - default_t_pre_init)
    )

    testing.assert_array_almost_equal(
        infections_default_t_pre_init, infections_default_t_pre_init_manual
    )

    # assert that infections at default t_pre_init is I_pre_init
    assert infections_default_t_pre_init[default_t_pre_init] == I_pre_init

    # test non-default t_pre_init
    t_pre_init = 6
    infections_custom_t_pre_init = InitializeInfectionsExponentialGrowth(
        n_timepoints, rate_rv=rate_RV, t_pre_init=t_pre_init
    ).initialize_infections(I_pre_init)
    infections_custom_t_pre_init_manual = I_pre_init * np.exp(
        rate * (np.arange(n_timepoints)[:, np.newaxis] - t_pre_init)
    )
    testing.assert_array_almost_equal(
        infections_custom_t_pre_init,
        infections_custom_t_pre_init_manual,
        decimal=5,
    )

    assert infections_custom_t_pre_init[t_pre_init] == I_pre_init


def test_initialize_infections_zero_pad():
    """Check that the InitializeInfectionsZeroPad class generates the correct number of infections at each time point."""

    n_timepoints = 10
    I_pre_init_RV = DeterministicVariable(name="I_pre_init_RV", value=10.0)
    (I_pre_init,) = I_pre_init_RV()
    I_pre_init = I_pre_init.value

    infections = InitializeInfectionsZeroPad(
        n_timepoints
    ).initialize_infections(I_pre_init)

    manual_infections = np.pad(
        np.atleast_1d(I_pre_init),
        (n_timepoints - np.array(I_pre_init).size, 0),
    )

    testing.assert_array_equal(infections, manual_infections)

    I_pre_init_RV_2 = DeterministicVariable(
        name="I_pre_init_RV", value=np.array([10.0, 10.0])
    )

    (I_pre_init_2,) = I_pre_init_RV_2()
    I_pre_init_2 = I_pre_init_2.value

    infections_2 = InitializeInfectionsZeroPad(
        n_timepoints
    ).initialize_infections(I_pre_init_2)
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

    infections = InitializeInfectionsFromVec(
        n_timepoints
    ).initialize_infections(I_pre_init)
    testing.assert_array_equal(infections, I_pre_init)

    I_pre_init_2 = np.arange(n_timepoints - 1)
    with pytest.raises(ValueError):
        InitializeInfectionsFromVec(n_timepoints).initialize_infections(
            I_pre_init_2
        )

    n_timepoints_float = 10.0
    with pytest.raises(TypeError):
        InitializeInfectionsFromVec(n_timepoints_float).initialize_infections(
            I_pre_init
        )

    n_timepoints_neg = -10
    with pytest.raises(ValueError):
        InitializeInfectionsFromVec(n_timepoints_neg).initialize_infections(
            I_pre_init
        )
