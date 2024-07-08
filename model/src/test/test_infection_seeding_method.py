# numpydoc ignore=GL08
import numpy as np
import numpy.testing as testing
import pytest
from pyrenew.deterministic import DeterministicVariable
from pyrenew.latent import (
    SeedInfectionsExponentialGrowth,
    SeedInfectionsFromVec,
    SeedInfectionsZeroPad,
)


def test_seed_infections_exponential():
    """Check that the SeedInfectionsExponentialGrowth class generates the correct number of infections at each time point."""
    n_timepoints = 10
    rate_RV = DeterministicVariable(0.5, name="rate_RV")
    I_pre_seed_RV = DeterministicVariable(10.0, name="I_pre_seed_RV")
    default_t_pre_seed = n_timepoints - 1

    (I_pre_seed,) = I_pre_seed_RV.sample()
    (rate,) = rate_RV.sample()

    infections_default_t_pre_seed = SeedInfectionsExponentialGrowth(
        n_timepoints, rate=rate_RV
    ).seed_infections(I_pre_seed)
    infections_default_t_pre_seed_manual = I_pre_seed * np.exp(
        rate * (np.arange(n_timepoints) - default_t_pre_seed)
    )

    testing.assert_array_almost_equal(
        infections_default_t_pre_seed, infections_default_t_pre_seed_manual
    )

    # assert that infections at default t_pre_seed is I_pre_seed
    assert infections_default_t_pre_seed[default_t_pre_seed] == I_pre_seed

    # test for failure with non-scalar rate or I_pre_seed
    rate_RV_2 = DeterministicVariable(np.array([0.5, 0.5]), name="rate_RV")
    with pytest.raises(ValueError):
        SeedInfectionsExponentialGrowth(
            n_timepoints, rate=rate_RV_2
        ).seed_infections(I_pre_seed)

    I_pre_seed_RV_2 = DeterministicVariable(
        np.array([10.0, 10.0]), name="I_pre_seed_RV"
    )
    (I_pre_seed_2,) = I_pre_seed_RV_2.sample()

    with pytest.raises(ValueError):
        SeedInfectionsExponentialGrowth(
            n_timepoints, rate=rate_RV
        ).seed_infections(I_pre_seed_2)

    # test non-default t_pre_seed
    t_pre_seed = 6
    infections_custom_t_pre_seed = SeedInfectionsExponentialGrowth(
        n_timepoints, rate=rate_RV, t_pre_seed=t_pre_seed
    ).seed_infections(I_pre_seed)
    infections_custom_t_pre_seed_manual = I_pre_seed * np.exp(
        rate * (np.arange(n_timepoints) - t_pre_seed)
    )
    testing.assert_array_almost_equal(
        infections_custom_t_pre_seed,
        infections_custom_t_pre_seed_manual,
        decimal=5,
    )

    assert infections_custom_t_pre_seed[t_pre_seed] == I_pre_seed


def test_seed_infections_zero_pad():
    """Check that the SeedInfectionsZeroPad class generates the correct number of infections at each time point."""

    n_timepoints = 10
    I_pre_seed_RV = DeterministicVariable(10.0, name="I_pre_seed_RV")
    (I_pre_seed,) = I_pre_seed_RV.sample()

    infections = SeedInfectionsZeroPad(n_timepoints).seed_infections(
        I_pre_seed
    )
    testing.assert_array_equal(
        infections, np.pad(I_pre_seed, (n_timepoints - I_pre_seed.size, 0))
    )

    I_pre_seed_RV_2 = DeterministicVariable(
        np.array([10.0, 10.0]), name="I_pre_seed_RV"
    )
    (I_pre_seed_2,) = I_pre_seed_RV_2.sample()

    infections_2 = SeedInfectionsZeroPad(n_timepoints).seed_infections(
        I_pre_seed_2
    )
    testing.assert_array_equal(
        infections_2,
        np.pad(I_pre_seed_2, (n_timepoints - I_pre_seed_2.size, 0)),
    )

    # Check that the SeedInfectionsZeroPad class raises an error when the length of I_pre_seed is greater than n_timepoints.
    with pytest.raises(ValueError):
        SeedInfectionsZeroPad(1).seed_infections(I_pre_seed_2)


def test_seed_infections_from_vec():
    """Check that the SeedInfectionsFromVec class generates the correct number of infections at each time point."""
    n_timepoints = 10
    I_pre_seed = np.arange(n_timepoints)

    infections = SeedInfectionsFromVec(n_timepoints).seed_infections(
        I_pre_seed
    )
    testing.assert_array_equal(infections, I_pre_seed)

    I_pre_seed_2 = np.arange(n_timepoints - 1)
    with pytest.raises(ValueError):
        SeedInfectionsFromVec(n_timepoints).seed_infections(I_pre_seed_2)

    n_timepoints_float = 10.0
    with pytest.raises(TypeError):
        SeedInfectionsFromVec(n_timepoints_float).seed_infections(I_pre_seed)

    n_timepoints_neg = -10
    with pytest.raises(ValueError):
        SeedInfectionsFromVec(n_timepoints_neg).seed_infections(I_pre_seed)
