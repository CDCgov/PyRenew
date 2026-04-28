"""
Shared pytest fixtures for PyRenew tests.

This module provides reusable fixtures for creating observation processes,
test data, and common configurations used across multiple test files.

The module also sets two JAX environment variables before any jax import so
that all tests (unit and integration) run with 64-bit precision and with
four logical host devices available for parallel MCMC chains. JAX reads
these variables at import time, so they must be set before the first
``import jax`` anywhere in the test process; placing them at the top of
this file (loaded by pytest before any test module) satisfies that
requirement. The ``setdefault`` form respects any value the caller already
set at the shell level (e.g., a CI with different configuration).
"""

import os

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from pyrenew.deterministic import DeterministicPMF, DeterministicVariable
from pyrenew.latent import (
    AR1,
    PopulationInfections,
    RandomWalk,
    SubpopulationInfections,
)
from pyrenew.observation import (
    HierarchicalNormalNoise,
    NegativeBinomialNoise,
    PoissonNoise,
    PopulationCounts,
    SubpopulationCounts,
)
from pyrenew.randomvariable import DistributionalVariable, VectorizedVariable
from pyrenew.time import MMWR_WEEK

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
def short_shedding_pmf():
    """
    Short 3-day shedding PMF.

    Returns
    -------
    jnp.ndarray
        A 3-element PMF array.
    """
    return jnp.array([0.3, 0.4, 0.3])


# =============================================================================
# Generation Interval Fixture
# =============================================================================


@pytest.fixture
def gen_int_rv():
    """
    COVID-like generation interval (7-day PMF).

    Returns
    -------
    DeterministicPMF
        Generation interval random variable.
    """
    pmf = jnp.array([0.16, 0.32, 0.25, 0.14, 0.07, 0.04, 0.02])
    return DeterministicPMF("gen_int", pmf)


# =============================================================================
# Noise Fixtures
# =============================================================================


@pytest.fixture
def hierarchical_normal_noise():
    """
    Standard HierarchicalNormalNoise with VectorizedVariable wrappers.

    Returns
    -------
    HierarchicalNormalNoise
        Noise model for continuous measurements.
    """
    sensor_mode_rv = VectorizedVariable(
        name="sensor_mode_rv",
        rv=DistributionalVariable("ww_sensor_mode", dist.Normal(0, 0.5)),
    )
    sensor_sd_rv = VectorizedVariable(
        name="sensor_sd_rv",
        rv=DistributionalVariable(
            "ww_sensor_sd", dist.TruncatedNormal(0.3, 0.15, low=0.10)
        ),
    )
    return HierarchicalNormalNoise(sensor_mode_rv, sensor_sd_rv)


@pytest.fixture
def hierarchical_normal_noise_tight():
    """
    Tight HierarchicalNormalNoise for near-deterministic testing.

    Returns
    -------
    HierarchicalNormalNoise
        Noise model with very small variance.
    """
    sensor_mode_rv = VectorizedVariable(
        name="sensor_mode_rv",
        rv=DistributionalVariable("ww_sensor_mode", dist.Normal(0, 0.01)),
    )
    sensor_sd_rv = VectorizedVariable(
        name="sensor_sd_rv",
        rv=DistributionalVariable(
            "ww_sensor_sd", dist.TruncatedNormal(0.01, 0.005, low=0.001)
        ),
    )
    return HierarchicalNormalNoise(sensor_mode_rv, sensor_sd_rv)


# =============================================================================
# Latent Infections Fixtures
# =============================================================================


@pytest.fixture
def subpopulation_infections(gen_int_rv):
    """
    Pre-configured SubpopulationInfections instance.

    Returns
    -------
    SubpopulationInfections
        Configured infection process with realistic parameters.
    """
    return SubpopulationInfections(
        name="subpopulation",
        gen_int_rv=gen_int_rv,
        I0_rv=DeterministicVariable("I0", 0.001),
        log_rt_time_0_rv=DeterministicVariable("log_rt_time_0", 0.0),
        baseline_rt_process=AR1(autoreg=0.9, innovation_sd=0.05),
        subpop_rt_deviation_process=RandomWalk(innovation_sd=0.025),
        n_initialization_points=7,
    )


@pytest.fixture
def population_infections(gen_int_rv):
    """
    Pre-configured PopulationInfections instance.

    Returns
    -------
    PopulationInfections
        Configured infection process with realistic parameters.
    """
    return PopulationInfections(
        name="population",
        gen_int_rv=gen_int_rv,
        I0_rv=DeterministicVariable("I0", 0.001),
        log_rt_time_0_rv=DeterministicVariable("log_rt_time_0", 0.0),
        single_rt_process=AR1(autoreg=0.9, innovation_sd=0.05),
        n_initialization_points=7,
    )


# =============================================================================
# PopulationCounts Process Fixtures
# =============================================================================


@pytest.fixture
def counts_process(simple_delay_pmf):
    """
    Standard PopulationCounts observation process with simple delay.

    Returns
    -------
    PopulationCounts
        A PopulationCounts observation process with no delay.
    """
    return PopulationCounts(
        name="test_counts",
        ascertainment_rate_rv=DeterministicVariable("ihr", 0.01),
        delay_distribution_rv=DeterministicPMF("delay", simple_delay_pmf),
        noise=NegativeBinomialNoise(DeterministicVariable("conc", 10.0)),
    )


class CountsProcessFactory:
    """Factory for creating PopulationCounts observation processes with custom parameters."""

    @staticmethod
    def create(
        name="test_counts",
        delay_pmf=None,
        ascertainment_rate=0.01,
        concentration=10.0,
    ):
        """
        Create a PopulationCounts observation process with specified parameters.

        Returns
        -------
        PopulationCounts
            A PopulationCounts observation process with the specified parameters.
        """
        if delay_pmf is None:
            delay_pmf = jnp.array([1.0])
        return PopulationCounts(
            name=name,
            ascertainment_rate_rv=DeterministicVariable("ihr", ascertainment_rate),
            delay_distribution_rv=DeterministicPMF("delay", delay_pmf),
            noise=NegativeBinomialNoise(DeterministicVariable("conc", concentration)),
        )


@pytest.fixture
def counts_factory():
    """
    Factory fixture for creating custom PopulationCounts observation processes.

    Returns
    -------
    CountsProcessFactory
        A factory for creating Counts processes.
    """
    return CountsProcessFactory()


@pytest.fixture
def weekly_regular_counts(simple_delay_pmf):
    """
    PopulationCounts with weekly aggregation and regular (dense) reporting.

    Uses MMWR Sunday-Saturday epiweeks (``start_dow=6``).

    Returns
    -------
    PopulationCounts
        A weekly-regular PopulationCounts process.
    """
    return PopulationCounts(
        name="hosp",
        ascertainment_rate_rv=DeterministicVariable("ihr", 0.01),
        delay_distribution_rv=DeterministicPMF("delay", simple_delay_pmf),
        noise=PoissonNoise(),
        aggregation="weekly",
        reporting_schedule="regular",
        start_dow=MMWR_WEEK,
    )


@pytest.fixture
def weekly_irregular_counts(simple_delay_pmf):
    """
    PopulationCounts with weekly aggregation and irregular (sparse) reporting.

    Uses MMWR Sunday-Saturday epiweeks (``start_dow=6``).

    Returns
    -------
    PopulationCounts
        A weekly-irregular PopulationCounts process.
    """
    return PopulationCounts(
        name="hosp",
        ascertainment_rate_rv=DeterministicVariable("ihr", 0.01),
        delay_distribution_rv=DeterministicPMF("delay", simple_delay_pmf),
        noise=PoissonNoise(),
        aggregation="weekly",
        reporting_schedule="irregular",
        start_dow=MMWR_WEEK,
    )


@pytest.fixture
def daily_irregular_counts(simple_delay_pmf):
    """
    PopulationCounts with daily scale and irregular (sparse) reporting.

    Returns
    -------
    PopulationCounts
        A daily-irregular PopulationCounts process.
    """
    return PopulationCounts(
        name="hosp",
        ascertainment_rate_rv=DeterministicVariable("ihr", 0.01),
        delay_distribution_rv=DeterministicPMF("delay", simple_delay_pmf),
        noise=PoissonNoise(),
        reporting_schedule="irregular",
    )


@pytest.fixture
def weekly_regular_subpop_counts(simple_delay_pmf):
    """
    SubpopulationCounts with weekly aggregation and regular (dense) reporting.

    Uses MMWR Sunday-Saturday epiweeks (``start_dow=6``).

    Returns
    -------
    SubpopulationCounts
        A weekly-regular SubpopulationCounts process.
    """
    return SubpopulationCounts(
        name="ed",
        ascertainment_rate_rv=DeterministicVariable("iedr", 0.01),
        delay_distribution_rv=DeterministicPMF("delay", simple_delay_pmf),
        noise=PoissonNoise(),
        aggregation="weekly",
        reporting_schedule="regular",
        start_dow=MMWR_WEEK,
    )


@pytest.fixture
def weekly_irregular_subpop_counts(simple_delay_pmf):
    """
    SubpopulationCounts with weekly aggregation and irregular (sparse) reporting.

    Uses MMWR Sunday-Saturday epiweeks (``start_dow=6``).

    Returns
    -------
    SubpopulationCounts
        A weekly-irregular SubpopulationCounts process.
    """
    return SubpopulationCounts(
        name="ed",
        ascertainment_rate_rv=DeterministicVariable("iedr", 0.01),
        delay_distribution_rv=DeterministicPMF("delay", simple_delay_pmf),
        noise=PoissonNoise(),
        aggregation="weekly",
        reporting_schedule="irregular",
        start_dow=MMWR_WEEK,
    )


@pytest.fixture
def daily_irregular_subpop_counts(simple_delay_pmf):
    """
    SubpopulationCounts with daily scale and irregular (sparse) reporting.

    Returns
    -------
    SubpopulationCounts
        A daily-irregular SubpopulationCounts process.
    """
    return SubpopulationCounts(
        name="ed",
        ascertainment_rate_rv=DeterministicVariable("iedr", 0.01),
        delay_distribution_rv=DeterministicPMF("delay", simple_delay_pmf),
        noise=PoissonNoise(),
        reporting_schedule="irregular",
    )


# =============================================================================
# Infection Fixtures
# =============================================================================


@pytest.fixture
def subpop_infections_28d():
    """
    Four-week ``(28, 3)`` infections array at 100/day/subpop.

    Used by weekly-aggregation tests on ``SubpopulationCounts``.

    Returns
    -------
    jnp.ndarray
        Shape ``(28, 3)`` filled with 100.0.
    """
    return jnp.ones((28, 3)) * 100.0


@pytest.fixture
def subpop_infections_30d():
    """
    Thirty-day ``(30, 3)`` infections array at 100/day/subpop.

    Used by daily-scale tests on ``SubpopulationCounts``.

    Returns
    -------
    jnp.ndarray
        Shape ``(30, 3)`` filled with 100.0.
    """
    return jnp.ones((30, 3)) * 100.0


@pytest.fixture
def mmwr_saturday_indices_first_three():
    """
    Daily-axis indices of the first three MMWR Saturdays for a Sunday-start axis.

    When element 0 of the daily axis is a Sunday (``first_day_dow=6``),
    Saturdays fall on indices 6, 13, 20, ...

    Returns
    -------
    jnp.ndarray
        Shape ``(3,)`` containing ``[6, 13, 20]``.
    """
    return jnp.array([6, 13, 20])


# =============================================================================
# Temporal Process Stubs
# =============================================================================


class WrongShapeTemporalProcess:
    """Temporal process stub that returns a fixed wrong-shaped array."""

    step_size = 1

    def __init__(self, value):
        """
        Store the wrong-shaped value to return from ``sample``.

        Parameters
        ----------
        value
            Array returned by ``sample`` regardless of requested shape.
        """
        self.value = value

    def sample(self, **kwargs):
        """
        Return the configured wrong-shaped value.

        Returns
        -------
        ArrayLike
            The array passed to ``__init__``.
        """
        return self.value


class ConstantTemporalProcess:
    """Temporal process stub that returns zeros with the requested shape."""

    step_size = 1

    def sample(self, n_timepoints, n_processes=1, **kwargs):
        """
        Return a correctly shaped zero trajectory.

        Parameters
        ----------
        n_timepoints
            Number of time points.
        n_processes
            Number of parallel processes.

        Returns
        -------
        jnp.ndarray
            Zeros of shape ``(n_timepoints, n_processes)``.
        """
        return jnp.zeros((n_timepoints, n_processes))


@pytest.fixture
def wrong_shape_temporal_process_cls():
    """
    Class producing a temporal-process stub that returns a fixed wrong shape.

    Returns
    -------
    type
        The ``WrongShapeTemporalProcess`` class. Instantiate with the array
        the stub should return from ``sample``.
    """
    return WrongShapeTemporalProcess


@pytest.fixture
def constant_temporal_process():
    """
    Temporal-process stub that returns a correctly shaped zero trajectory.

    Returns
    -------
    ConstantTemporalProcess
        Instance whose ``sample`` returns zeros of the requested shape.
    """
    return ConstantTemporalProcess()
