"""
Shared pytest fixtures for PyRenew tests.

This module provides reusable fixtures for creating observation processes,
test data, and common configurations used across multiple test files.
"""

import jax.numpy as jnp
import pytest

from pyrenew.deterministic import DeterministicPMF, DeterministicVariable
from pyrenew.observation import Counts, NegativeBinomialNoise
from pyrenew.randomvariable import GammaGroupSdPrior, HierarchicalNormalPrior

# =============================================================================
# PMF Fixtures
# =============================================================================


@pytest.fixture
def simple_delay_pmf():
    """
    Simple 1-day delay PMF (no delay).

    Returns
    -------
    jnp.ndarray
        A single-element PMF array representing no delay.
    """
    return jnp.array([1.0])


@pytest.fixture
def short_delay_pmf():
    """
    Short 2-day delay PMF.

    Returns
    -------
    jnp.ndarray
        A 2-element PMF array.
    """
    return jnp.array([0.5, 0.5])


@pytest.fixture
def medium_delay_pmf():
    """
    Medium 4-day delay PMF.

    Returns
    -------
    jnp.ndarray
        A 4-element PMF array.
    """
    return jnp.array([0.1, 0.3, 0.4, 0.2])


@pytest.fixture
def realistic_delay_pmf():
    """
    Realistic 10-day delay PMF (shifted gamma-like).

    Returns
    -------
    jnp.ndarray
        A 10-element PMF array with gamma-like shape.
    """
    return jnp.array([0.01, 0.05, 0.10, 0.15, 0.20, 0.20, 0.15, 0.08, 0.04, 0.02])


@pytest.fixture
def long_delay_pmf():
    """
    Long 10-day delay PMF for edge case testing.

    Returns
    -------
    jnp.ndarray
        A 10-element PMF array.
    """
    return jnp.array([0.05, 0.1, 0.15, 0.2, 0.2, 0.15, 0.1, 0.03, 0.01, 0.01])


@pytest.fixture
def simple_shedding_pmf():
    """
    Simple 1-day shedding PMF (no delay).

    Returns
    -------
    jnp.ndarray
        A single-element PMF array representing no shedding delay.
    """
    return jnp.array([1.0])


@pytest.fixture
def short_shedding_pmf():
    """
    Short 3-day shedding PMF.

    Returns
    -------
    jnp.ndarray
        A 3-element PMF array.
    """
    return jnp.array([0.3, 0.4, 0.3])


@pytest.fixture
def medium_shedding_pmf():
    """
    Medium 5-day shedding PMF.

    Returns
    -------
    jnp.ndarray
        A 5-element PMF array.
    """
    return jnp.array([0.1, 0.3, 0.3, 0.2, 0.1])


# =============================================================================
# Hierarchical Prior Fixtures
# =============================================================================


@pytest.fixture
def sensor_mode_prior():
    """
    Standard hierarchical normal prior for sensor modes.

    Returns
    -------
    HierarchicalNormalPrior
        A hierarchical normal prior with standard deviation 0.5.
    """
    return HierarchicalNormalPrior(
        name="ww_sensor_mode", sd_rv=DeterministicVariable("mode_sd", 0.5)
    )


@pytest.fixture
def sensor_mode_prior_tight():
    """
    Tight hierarchical normal prior for deterministic-like behavior.

    Returns
    -------
    HierarchicalNormalPrior
        A hierarchical normal prior with small standard deviation 0.01.
    """
    return HierarchicalNormalPrior(
        name="ww_sensor_mode", sd_rv=DeterministicVariable("mode_sd_tight", 0.01)
    )


@pytest.fixture
def sensor_sd_prior():
    """
    Standard gamma prior for sensor standard deviations.

    Returns
    -------
    GammaGroupSdPrior
        A gamma prior for group standard deviations.
    """
    return GammaGroupSdPrior(
        name="ww_sensor_sd",
        sd_mean_rv=DeterministicVariable("sd_mean", 0.3),
        sd_concentration_rv=DeterministicVariable("sd_concentration", 4.0),
        sd_min=0.10,
    )


@pytest.fixture
def sensor_sd_prior_tight():
    """
    Tight gamma prior for deterministic-like behavior.

    Returns
    -------
    GammaGroupSdPrior
        A gamma prior with small mean for tight behavior.
    """
    return GammaGroupSdPrior(
        name="ww_sensor_sd",
        sd_mean_rv=DeterministicVariable("sd_mean_tight", 0.01),
        sd_concentration_rv=DeterministicVariable("sd_concentration_tight", 4.0),
        sd_min=0.005,
    )


# =============================================================================
# Counts Process Fixtures
# =============================================================================


@pytest.fixture
def counts_process(simple_delay_pmf):
    """
    Standard Counts observation process with simple delay.

    Returns
    -------
    Counts
        A Counts observation process with no delay.
    """
    return Counts(
        ascertainment_rate_rv=DeterministicVariable("ihr", 0.01),
        delay_distribution_rv=DeterministicPMF("delay", simple_delay_pmf),
        noise=NegativeBinomialNoise(DeterministicVariable("conc", 10.0)),
    )


@pytest.fixture
def counts_process_medium_delay(medium_delay_pmf):
    """
    Counts observation process with medium delay.

    Returns
    -------
    Counts
        A Counts observation process with 4-day delay.
    """
    return Counts(
        ascertainment_rate_rv=DeterministicVariable("ihr", 0.01),
        delay_distribution_rv=DeterministicPMF("delay", medium_delay_pmf),
        noise=NegativeBinomialNoise(DeterministicVariable("conc", 50.0)),
    )


@pytest.fixture
def counts_process_realistic(realistic_delay_pmf):
    """
    Counts observation process with realistic delay and ascertainment.

    Returns
    -------
    Counts
        A Counts observation process with realistic parameters.
    """
    return Counts(
        ascertainment_rate_rv=DeterministicVariable("ihr", 0.005),
        delay_distribution_rv=DeterministicPMF("delay", realistic_delay_pmf),
        noise=NegativeBinomialNoise(DeterministicVariable("conc", 100.0)),
    )


class CountsProcessFactory:
    """Factory for creating Counts processes with custom parameters."""

    @staticmethod
    def create(
        delay_pmf=None,
        ascertainment_rate=0.01,
        concentration=10.0,
    ):
        """
        Create a Counts process with specified parameters.

        Returns
        -------
        Counts
            A Counts observation process with the specified parameters.
        """
        if delay_pmf is None:
            delay_pmf = jnp.array([1.0])
        return Counts(
            ascertainment_rate_rv=DeterministicVariable("ihr", ascertainment_rate),
            delay_distribution_rv=DeterministicPMF("delay", delay_pmf),
            noise=NegativeBinomialNoise(DeterministicVariable("conc", concentration)),
        )


@pytest.fixture
def counts_factory():
    """
    Factory fixture for creating custom Counts processes.

    Returns
    -------
    CountsProcessFactory
        A factory for creating Counts processes.
    """
    return CountsProcessFactory()


# =============================================================================
# Infection Fixtures
# =============================================================================


@pytest.fixture
def constant_infections():
    """
    Constant infections array (30 days, 100 infections/day).

    Returns
    -------
    jnp.ndarray
        A 1D array of shape (30,) with constant value 100.
    """
    return jnp.ones(30) * 100


@pytest.fixture
def constant_infections_2d():
    """
    Constant infections array for 2 subpopulations.

    Returns
    -------
    jnp.ndarray
        A 2D array of shape (30, 2) with constant value 100.
    """
    return jnp.ones((30, 2)) * 100


def make_infections(n_days, n_subpops=None, value=100.0):
    """
    Create infection arrays for testing.

    Parameters
    ----------
    n_days : int
        Number of days
    n_subpops : int, optional
        Number of subpopulations (None for 1D array)
    value : float
        Constant infection value

    Returns
    -------
    jnp.ndarray
        Infections array
    """
    if n_subpops is None:
        return jnp.ones(n_days) * value
    return jnp.ones((n_days, n_subpops)) * value


def make_spike_infections(n_days, spike_day, spike_value=1000.0, n_subpops=None):
    """
    Create spike infection arrays for testing.

    Parameters
    ----------
    n_days : int
        Number of days
    spike_day : int
        Day of the spike
    spike_value : float
        Value at spike
    n_subpops : int, optional
        Number of subpopulations

    Returns
    -------
    jnp.ndarray
        Infections array with spike
    """
    if n_subpops is None:
        infections = jnp.zeros(n_days)
        return infections.at[spike_day].set(spike_value)
    infections = jnp.zeros((n_days, n_subpops))
    return infections.at[spike_day, :].set(spike_value)


def create_mock_infections(
    n_days: int,
    peak_day: int = 10,
    peak_value: float = 1000.0,
    shape: str = "spike",
) -> jnp.ndarray:
    """
    Create mock infection time series for testing.

    Parameters
    ----------
    n_days : int
        Number of days
    peak_day : int
        Day of peak infections
    peak_value : float
        Peak infection value
    shape : str
        Shape of the curve: "spike", "constant", or "decay"

    Returns
    -------
    jnp.ndarray
        Array of infections of shape (n_days,)
    """
    if shape == "spike":
        infections = jnp.zeros(n_days)
        infections = infections.at[peak_day].set(peak_value)
    elif shape == "constant":
        infections = jnp.ones(n_days) * peak_value
    elif shape == "decay":
        infections = peak_value * jnp.exp(-jnp.arange(n_days) / 20.0)
    else:
        raise ValueError(f"Unknown shape: {shape}")

    return infections
