# numpydoc ignore=GL08
import numpy as np
import numpy.testing as testing
from pyrenew.deterministic import DeterministicVariable
from pyrenew.latent import (
    SeedInfectionsExponential,
    SeedInfectionsFromVec,
    SeedInfectionsZeroPad,
)


def test_seed_infections_exponential():
    """Check that the SeedInfectionsExponential class generates the correct number of infections at each time point."""
    n_timepoints = 10
    rate_RV = DeterministicVariable(0.5, name="rate_RV")
    I_pre_seed_RV = DeterministicVariable(10.0, name="I_pre_seed_RV")
    default_t_pre_seed = n_timepoints - 1

    (I_pre_seed,) = I_pre_seed_RV.sample()
    (rate,) = rate_RV.sample()

    infections_default_t_pre_seed = SeedInfectionsExponential(
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

    t_pre_seed = 6
    infections_custom_t_pre_seed = SeedInfectionsExponential(
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


def test_seed_infections_from_vec():
    """Check that the SeedInfectionsFromVec class generates the correct number of infections at each time point."""
    n_timepoints = 10
    I_pre_seed_RV = DeterministicVariable(np.arange(10), name="I_pre_seed_RV")
    (I_pre_seed,) = I_pre_seed_RV.sample()

    infections = SeedInfectionsFromVec(n_timepoints).seed_infections(
        I_pre_seed
    )
    testing.assert_array_equal(infections, I_pre_seed)
